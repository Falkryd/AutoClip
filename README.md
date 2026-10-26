# AutoClip

Runs after each YouTube live stream ends and turns the VOD into 2–3 vertical Shorts, uploaded as
**private** for review, with a Discord message when they're ready.

| # | Stage | Command | Status |
|---|---|---|---|
| 1 | trigger: find finished streams | `uv run python -m src.trigger` | done |
| 2 | download: VOD + chat replay (yt-dlp) | `uv run python -m src.download VIDEO_ID` | next |
| 3 | transcribe: faster-whisper | `uv run python -m src.transcribe VIDEO_ID` | |
| 4 | highlights: chat + loudness + Ollama | `uv run python -m src.highlights VIDEO_ID` | |
| 5 | render: 9:16 stacked layout + captions | `uv run python -m src.render VIDEO_ID` | |
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
3. Install Deno: yt-dlp needs a JavaScript runtime to download from YouTube.
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

- Trigger with `method: rss` costs nothing. `method: api` costs about 2 units per run.
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

## Tests

```bash
uv run pytest
```
