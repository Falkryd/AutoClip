from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from src.common.config import CaptionConfig, RenderConfig, ShotsConfig
from src.render.captions import build_ass, split_at_spans
from src.render.layout import Box, filter_graph, plan_layout, zoompan
from src.render.shots import (
    SAMPLE_FPS,
    Moment,
    Shot,
    action,
    deaths,
    manual_shots,
    measure_voice,
    plan_shots,
    shouts,
)

BOTH = {"game", "face"}

# --- choosing shots ---------------------------------------------------------------------------


def test_death_then_laugh_reads_as_cause_then_reaction() -> None:
    # Died at 10 s, laughed at 10.8 s: the game view shows the death, the avatar takes over
    # where it ends (the laugh's shot starts later than it would on its own).
    shots = plan_shots([Moment(10.8, "laugh"), Moment(10.0, "death")], 40.0, 0.5, BOTH)
    assert shots == [Shot(8.5, 11.0, "game", "death"), Shot(11.0, 12.5, "face", "laugh")]


def test_laughs_close_together_become_one_shot() -> None:
    shots = plan_shots([Moment(5.0, "laugh"), Moment(6.5, "laugh")], 40.0, 0.5, BOTH)
    assert shots == [Shot(4.7, 8.2, "face", "laugh")]


def test_shots_stay_within_the_budget_and_after_the_lead_in() -> None:
    laughs = [Moment(t, "laugh") for t in (0.5, 10.0, 20.0, 30.0)]
    shots = plan_shots(laughs, 40.0, 0.1, BOTH)  # 4 s of 40 s: room for two
    assert [s.start for s in shots] == [1.0, 9.7]  # the first waits for the opening second
    assert sum(s.end - s.start for s in shots) <= 4.0


def test_an_action_stretch_takes_the_free_part_before_a_laugh() -> None:
    fight = Moment(4.5, "action", start=1.0, end=6.0)
    shots = plan_shots([fight, Moment(3.7, "laugh")], 40.0, 0.5, BOTH)
    # the laugh wins 3.4-5.4; the fight keeps 1.0-3.4, which runs straight into it
    assert shots == [Shot(1.0, 3.4, "game", "action"), Shot(3.4, 5.4, "face", "laugh")]


def test_short_flash_of_the_stacked_view_is_bridged() -> None:
    shots = plan_shots([Moment(10.0, "death"), Moment(11.7, "laugh")], 40.0, 0.5, BOTH)
    # game 8.5-11.0, face 11.4-13.4: a 0.4 s gap closes
    assert shots[0].end == shots[1].start == 11.4


def test_no_face_shots_without_an_avatar() -> None:
    assert plan_shots([Moment(5.0, "laugh")], 40.0, 0.5, {"game"}) == []


def test_manual_shots() -> None:
    spec = [{"start": 5, "end": 7, "view": "face"}, {"start": 1, "end": 99, "view": "game"}]
    assert manual_shots(spec, 20.0) == [  # type: ignore[arg-type]
        Shot(1.0, 20.0, "game", "manual"), Shot(5.0, 7.0, "face", "manual")
    ]  # fmt: skip
    assert manual_shots([], 20.0) == []
    with pytest.raises(ValueError, match="view"):
        manual_shots([{"start": 1, "end": 2, "view": "chat"}], 20.0)
    with pytest.raises(ValueError, match="overlap"):
        manual_shots([{"start": 1, "end": 3, "view": "face"},
                      {"start": 2, "end": 4, "view": "face"}], 20.0)  # fmt: skip
    with pytest.raises(ValueError, match="stretch"):
        manual_shots([{"start": 5, "end": 4, "view": "face"}], 20.0)


# --- what happens on screen and in the voice --------------------------------------------------


def series(*parts: tuple[float, float]) -> np.ndarray:
    """(value, seconds) pieces, sampled at SAMPLE_FPS."""
    return np.concatenate([np.full(int(s * SAMPLE_FPS), v) for v, s in parts])


def test_screen_turning_grey_is_a_death() -> None:
    colour = series((70, 5), (38, 4))
    assert deaths(colour, series((100, 9))) == [5.0]


def test_menus_fades_and_loading_screens_are_not_deaths() -> None:
    colour = series((70, 5), (38, 4))
    assert deaths(colour, series((100, 5), (70, 4))) == []  # shop or scoreboard: darker
    assert deaths(series((27, 5), (16, 4)), series((25, 5), (34, 4))) == []  # black, then logo


def test_action_is_a_long_enough_busy_stretch() -> None:
    motion = series((5, 4), (30, 3), (5, 4), (30, 0.4), (5, 4))
    (fight,) = action(motion)
    assert fight.kind == "action" and 3.5 <= fight.start <= 4.2 and 6.8 <= fight.end <= 7.4


def test_shouts_need_a_real_jump() -> None:
    db = series((-30, 5), (-16, 0.6), (-30, 5), (-25, 1))
    assert shouts(db, 12.0) == [5.0]


def test_measure_voice(tmp_path: Path) -> None:
    rate = 16_000
    pcm = np.concatenate([np.zeros(rate), np.full(rate, 3000)]).astype("<i2")
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm.tobytes())
    db = measure_voice(path, 0.5, 1.0)
    assert len(db) == SAMPLE_FPS
    assert db[0] < -100 and db[-1] == pytest.approx(20 * np.log10(3000 / 32768), abs=0.1)


# --- drawing them -----------------------------------------------------------------------------


def test_views_in_the_layout() -> None:
    layout = plan_layout(RenderConfig(), 1920, 1080)
    assert layout.game_view_src == Box(698, 0, 522, 928)  # middle of the screen, above the HUD
    assert layout.face_out == Box(0, 240, 1080, 1280)  # full width, below the hook line
    assert plan_layout(RenderConfig(layout="center"), 1920, 1080).face_out is None


def test_filter_graph_with_shots() -> None:
    layout = plan_layout(RenderConfig(), 1920, 1080)
    shots = [Shot(2.0, 3.5, "game", "death"), Shot(3.5, 5.0, "face", "laugh"),
             Shot(9.0, 10.0, "face", "laugh")]  # fmt: skip
    graph = filter_graph(layout, "01.ass", None, shots, ShotsConfig(), 60.0)
    assert graph.startswith("[0:v]setpts=N/FRAME_RATE/TB,split=5[bg][game_in][avatar_in]"
                            "[game_view_in][face_view_in]")  # fmt: skip
    assert "[game_view_in]crop=522:928:698:0,zoompan=" in graph
    assert "[face_view_in]crop=346:410:1574:670,zoompan=" in graph and ":s=1080x1280:" in graph
    assert "[face_base][face]overlay=0:240[face_view]" in graph
    assert "[with_game_view][face_view]overlay=0:0:enable='between(t,3.500,5.000)+between(t," \
           "9.000,10.000)'" in graph  # fmt: skip
    assert graph.endswith("subtitles=01.ass,format=yuv420p[vout]")
    # without shots the graph is the plain stacked one
    assert "zoompan" not in filter_graph(layout, None)


def test_zoompan_ramps_in_each_shot() -> None:
    z = zoompan([(2.0, 4.0)], 1.3, 0.2, 0.5, 0.35, 1080, 1280, 60.0)
    assert z == ("zoompan=z='1+0.300*(between(it,2.000,4.000)*min(1,(it-2.000)/0.200))'"
                 ":x='(iw-iw/zoom)*0.5':y='(ih-ih/zoom)*0.35':d=1:s=1080x1280:fps=60")  # fmt: skip
    assert "/2.000))" in zoompan([(2.0, 4.0)], 1.1, None, 0.5, 0.5, 1080, 1920, 60.0)  # whole shot


def test_captions_drop_during_shots() -> None:
    assert split_at_spans(1.0, 3.0, [(2.0, 5.0)]) == [(1.0, 2.0, False), (2.0, 3.0, True)]
    assert split_at_spans(1.0, 3.0, []) == [(1.0, 3.0, False)]
    words = [{"start": 1.0, "end": 1.5, "word": "halo"}, {"start": 6.0, "end": 6.5, "word": "x"}]
    ass = build_ass(words, CaptionConfig(), 1080, 1920, low_spans=[(5.0, 8.0)], low_y=0.8)
    dialogue = [line for line in ass.splitlines() if line.startswith("Dialogue")]
    assert dialogue[0].split(",")[7] == "0"  # the style's height
    assert dialogue[1].split(",")[7] == "384"  # 20 % from the bottom
