# AutoClip

Runs after each YouTube live stream ends and turns the VOD into 2–3 vertical Shorts, uploaded as
**private** for review, with a Discord message when they're ready.

| # | Stage | Command | Status |
|---|---|---|---|
| 1 | trigger: find finished streams | `uv run python -m src.trigger` | done |
| 2 | download: VOD + chat replay (yt-dlp) | `uv run python -m src.download VIDEO_ID` | done |
| 3 | transcribe: YouTube captions, Whisper fallback | `uv run python -m src.transcribe VIDEO_ID` | done |
| 4 | highlights: laughs + loudness + chat + Ollama | `uv run python -m src.highlights VIDEO_ID` | done |
| 5 | render: 9:16 stacked layout + word captions | `uv run python -m src.render VIDEO_ID` | done |
| 6 | upload: private upload + Discord | `uv run python -m src.upload VIDEO_ID` | |

Stages pass work to each other only through files in `work/<video_id>/`. State (what was
seen and which stages finished) lives in `state/autoclip.sqlite3`; logs go to `logs/pipeline.log`.

## Local setup

```bash
uv sync
```

1. Copy `.env.example` to `.env`. Secrets go there and nowhere else.
2. Set `channel.id` in `config.yaml`. Find your channel ID (starts with `UC`) at
   https://www.youtube.com/account_advanced while signed in.
3. Install Deno: yt-dlp needs a JavaScript runtime to download from YouTube. Open a new
   terminal afterwards so it's on PATH, or set `download.deno_path` in `config.yaml`.
   `winget install DenoLand.Deno`
4. Already present on this machine: FFmpeg full build (with libass) and Ollama with `qwen2.5:7b`.

## Google Cloud setup (manual, one time)

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project and enable
   **YouTube Data API v3**.
2. **OAuth consent screen**: User type *External*. Add yourself as a test user.
   Scopes: `.../auth/youtube.upload` and `.../auth/youtube.readonly`.
3. **Publish the app to "In production"** (it can stay unverified). In *Testing* mode, refresh
   tokens expire after 7 days and scheduled runs start failing. For personal use, unverified
   only means a warning screen the first time you sign in.
4. **Credentials > Create credentials > OAuth client ID**, type **Desktop app**. Download the
   JSON to `secrets/client_secret.json`.
5. Authorize once (opens a browser, writes `secrets/token.json`):
   `uv run python -m src.common.youtube_auth`

### Quota

The default is 10,000 units/day.

- Trigger with `method: ytdlp` (default) or `rss` costs nothing. `method: api` costs about 2
  units per run.
- Each upload costs about 1,600 units, so `clips_per_stream` is capped at 6 and
  `upload.max_uploads_per_day` defaults to 5.
- YouTube forces uploads from unaudited API projects to private. That suits this workflow, but
  those videos may stay locked to private. To publish clips from the pipeline itself, apply for
  the API audit (it takes weeks).

## Discord

Server Settings > Integrations > Webhooks > New Webhook > Copy Webhook URL, then put it in `.env`
as `DISCORD_WEBHOOK_URL`.

## YouTube settings to check

- Live stream archiving is on (VODs are saved).
- Live chat replay is on (needed for chat-activity scoring).

## Trigger usage

```bash
uv run python -m src.trigger              # check the feed, queue finished streams
uv run python -m src.trigger --dry-run    # show decisions without saving
uv run python -m src.trigger --list       # show everything in the state database
uv run python -m src.trigger --queue VIDEO_ID   # queue an older stream by hand
```

On the first run, streams that ended more than `trigger.lookback_hours` ago are marked
`skipped`, so old streams aren't backfilled. Live, upcoming and still-processing streams stay
`waiting` and are rechecked on every run until their VOD is ready.

## Download usage

```bash
uv run python -m src.download VIDEO_ID          # full VOD into work/<id>/
uv run python -m src.download https://youtube.com/live/VIDEO_ID --section 2:33:00-3:33:00
```

`--section` cuts that part into `samples/<id>_<start>-<end>/` for developing later stages. It
downloads the full VOD into `work/<id>/` first (yt-dlp fetching just a range stalls on long
streams) and keeps it, so the real pipeline won't download it again. Every stage accepts the
same folder layout:

| File | What it is |
|---|---|
| `vod.mp4` | the video (1080p max) |
| `live_chat.raw.jsonl` | chat replay as yt-dlp saved it (only if the stream has one) |
| `chat.json` | flat message list, times relative to `vod.mp4`; `available: false` if no replay |
| `audio.wav` | 16 kHz mono, for transcription and loudness scoring |
| `download.json` | summary, written last: its presence means the stage finished |

Re-running skips files that already exist; `--force` downloads again. YouTube doesn't seem
to keep a chat replay for streams where nobody chatted, so `available: false` is normal.

## Transcribe usage

```bash
uv run python -m src.transcribe VIDEO_ID                              # work/<id>/
uv run python -m src.transcribe samples/8fKUNhHQR_M_023300-033300     # a sample folder
uv run python -m src.transcribe VIDEO_ID --source whisper             # force Whisper
```

With `transcribe.source: auto` (the default) the stage uses **YouTube's automatic captions**
(Indonesian, free, about a second to download). They catch more speech than Whisper on this
channel and carry sound tags such as `[tertawa]` (laughing) and `[musik]`, which the highlights
stage uses. If YouTube hasn't made captions yet, the stage exits with code 3 and is retried on
the next run, for up to `captions_wait_hours` after the download; then **Whisper** (`large-v3`
on the GPU, about 40 s per hour of audio) takes over. `--no-wait` skips the wait.

Output, the same shape for both sources:

| File | What it is |
|---|---|
| `transcript.json` | sentences with word timings, `sounds` (tag + time), and a `dropped` list with reasons |
| `transcript.srt` | the sentences as subtitles: open it next to `vod.mp4` in VLC to check |
| `youtube_captions.json3` | YouTube's captions as downloaded (whole stream) |

Times are relative to `vod.mp4`. For a sample, `download.json` records where the cut really
starts (`offset_s`: the keyframe before the requested time, ~5 s earlier on these VODs), and
captions and chat are shifted by that, so they line up with the video to within ~0.2 s.

Clean-up, mostly for Whisper: phrases it invents over music ("Terima kasih telah menonton!"),
segments that are probably not speech, and repetition loops are dropped. YouTube's censored
swear words come through as `***`.

## Highlights usage

```bash
uv run python -m src.highlights VIDEO_ID                            # work/<id>/
uv run python -m src.highlights samples/8fKUNhHQR_M_023300-033300   # a sample folder
uv run python -m src.highlights VIDEO_ID --no-llm                   # signals only, for tuning
uv run python -m src.highlights VIDEO_ID --focus "ga tidur sebelum menang"   # favour a theme
```

`--focus THEME` favours moments about a theme: the LLM is told (an off-theme moment scores at
most 6, however funny), and the theme's words become a signal (`weights.focus`), so on-theme
stretches make the shortlist. The words are taken from the theme (`tidur`, `menang`) and match
inside longer words (`kemenangan`, `tidurnya`); name your own with `--focus-words
tidur,menang,ngantuk`.

1. Every 45 s window (5 s steps) is scored on **laughs** (`[tertawa]` tags in YouTube's
   captions), **loudness spikes** (how far you get above the surrounding minute, so a loud
   teamfight alone doesn't count) and **chat** (messages in the window). A signal a stream
   doesn't have (no chat replay, Whisper transcript without laugh tags) is left out and the
   other weights are rescaled.
2. The best 12 non-overlapping windows go to the local LLM (`qwen2.5:7b` in Ollama) with 15 s of
   transcript around them. It scores each 0-10, picks clean start and end lines, and writes a
   title and a one-line reason. The model is unloaded afterwards to free VRAM for rendering.
3. Each clip is snapped to whole sentences, keeps a laugh that follows the punchline, and is
   fitted to 30-60 s. The best `clips_per_stream` (3), at least 60 s apart, are kept; anything
   the LLM scored below `min_llm_score` is never picked.

4. **Still screens**: how much the gameplay area moves is measured once per folder (cached in
   `motion.json`; about 3.5 min per hour of video). A clip loses up to `still_screen_penalty`
   (50%) of its score by how much of it is still, so a paused game loses to live play. The
   post-game scoreboard is just as still, so banter over it competes poorly on a full-stream
   run; cut that stretch as its own section to get those clips.
5. **Glossary**: `highlights.glossary` explains your stream's nicknames and slang to the LLM
   (`kotol: Keeper of the Light`). Add your own.

`highlights.json` holds the clips (times in `vod.mp4` and in the full stream, scores per
signal, transcript text, music warnings for copyright) and every candidate that was judged,
for tuning the weights in `config.yaml`.

## Render usage

```bash
uv run python -m src.render VIDEO_ID                              # work/<id>/
uv run python -m src.render samples/8fKUNhHQR_M_023300-033300     # a sample folder
uv run python -m src.render VIDEO_ID --preview                    # one still, to tune the crops
```

Each clip in `highlights.json` becomes `clips/NN.mp4` (1080x1920, 60 fps, H.264 on the GPU via
NVENC, AAC 48 kHz) plus `NN.jpg` (thumbnail for the review message) and `NN.ass` (the captions):

- **Layout** (`render.layout: stacked`): the gameplay crop (`render.gameplay`, the middle of the
  screen) at full width on top, the avatar crop (`render.avatar`, bottom-right of the stream)
  enlarged below at its own shape, on a blurred copy of the stream. `--preview` writes
  `clips/layout_preview.png` so the boxes can be tuned without rendering whole clips.
  `layout: center` is a plain 9:16 crop instead.
- **Captions**: up to 4 words at a time, broken at pauses and punctuation, the word being said
  lit up (`captions.highlight_color`). `captions.y` sets the height (0.49 = just above the
  Dota HUD); `captions.style: sentence` turns the word highlight off.
- **Audio**: two-pass loudness normalization to YouTube's -14 LUFS (with a -1.5 dBTP peak
  limit, so loud reactions don't clip).

Rendering takes about a quarter of the clip's length on the RTX 5060 Ti. If NVENC ever fails,
set `render.video_encoder: libx264` to encode on the CPU.

## Tests

```bash
uv run pytest
```
