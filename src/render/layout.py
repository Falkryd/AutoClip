"""Where the gameplay and avatar go in the vertical frame, as an FFmpeg filter graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.common.config import Rect, RenderConfig, ShotsConfig
from src.render.shots import Shot

FACE_TOP = 0.125  # the full-screen avatar starts here, below the hook line


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    gameplay_src: Box  # crop of the source frame
    gameplay_out: Box  # where it lands in the output
    avatar_src: Box | None
    avatar_out: Box | None
    game_view_src: Box | None = None  # full-screen game shots: this crop, at 9:16
    face_out: Box | None = None  # full-screen avatar shots: the avatar here, full width


def _even(value: float) -> int:
    """Video sizes and offsets must be even for yuv420p."""
    return max(2, int(round(value / 2)) * 2)


def source_box(rect: Rect, src_w: int, src_h: int) -> Box:
    w, h = min(_even(rect.w * src_w), src_w), min(_even(rect.h * src_h), src_h)
    x = min(_even(rect.x * src_w) if rect.x else 0, src_w - w)
    y = min(_even(rect.y * src_h) if rect.y else 0, src_h - h)
    return Box(x, y, w, h)


def fit_aspect(box: Box, aspect: float) -> Box:
    """The biggest part of box with this width/height ratio, around its centre."""
    if box.w / box.h > aspect:
        w = min(_even(box.h * aspect), box.w)
        return Box(box.x + (box.w - w) // 2 // 2 * 2, box.y, w, box.h)
    h = min(_even(box.w / aspect), box.h)
    return Box(box.x, box.y + (box.h - h) // 2 // 2 * 2, box.w, h)


def plan_layout(rc: RenderConfig, src_w: int, src_h: int) -> Layout:
    width, height = rc.width, rc.height
    game_view = fit_aspect(source_box(rc.shots.game_view, src_w, src_h), width / height)
    if rc.layout == "center":
        crop_w = min(_even(src_h * width / height), src_w)
        crop_h = min(_even(crop_w * height / width), src_h)
        crop = Box(_even((src_w - crop_w) / 2) if src_w > crop_w else 0, 0, crop_w, crop_h)
        out = Box(0, 0, width, height)
        return Layout(width, height, crop, out, None, None, game_view)

    # stacked: gameplay on top at full width, avatar below at its own aspect ratio
    game, avatar = source_box(rc.gameplay, src_w, src_h), source_box(rc.avatar, src_w, src_h)
    avatar_h = _even(height * rc.avatar_panel_fraction)
    avatar_w = _even(avatar_h * avatar.w / avatar.h)
    if avatar_w > width:
        avatar_w, avatar_h = width, _even(width * avatar.h / avatar.w)
    room = height - avatar_h
    game_w, game_h = width, _even(width * game.h / game.w)
    if game_h > room:  # too tall for the space above the avatar: shrink, keep the shape
        game_w, game_h = _even(room * game.w / game.h), room
    game_out = Box((width - game_w) // 2 // 2 * 2, 0, game_w, game_h)
    avatar_out = Box((width - avatar_w) // 2 // 2 * 2, height - avatar_h, avatar_w, avatar_h)
    face_w, face_h = width, _even(width * avatar.h / avatar.w)
    if face_h > height:
        face_w, face_h = _even(height * avatar.w / avatar.h), height
    face_out = Box((width - face_w) // 2 // 2 * 2, min(_even(height * FACE_TOP), height - face_h),
                   face_w, face_h)  # fmt: skip
    return Layout(width, height, game, game_out, avatar, avatar_out, game_view, face_out)


def filter_graph(
    layout: Layout,
    captions_file: str | None,
    select: str | None = None,
    shots: Sequence[Shot] = (),
    sc: ShotsConfig | None = None,
    fps: float = 60.0,
) -> str:
    """Input 0's video to [vout]. captions_file is a bare file name, resolved from FFmpeg's
    working directory (Windows paths inside filter arguments need awkward escaping).
    select: a select expression for jump cuts, applied before anything else.
    shots: full-screen views laid over the stacked layout while they run (cut timeline)."""
    w, h = layout.width, layout.height
    g, go = layout.gameplay_src, layout.gameplay_out
    has_avatar = layout.avatar_src is not None and layout.avatar_out is not None
    sc = sc or ShotsConfig()
    views = [v for v in ("game", "face") if any(s.view == v for s in shots)]
    if layout.game_view_src is None and "game" in views:
        views.remove("game")
    if (not has_avatar or layout.face_out is None) and "face" in views:
        views.remove("face")
    # Every branch is computed for every frame. Dropping frames outside the shots would be
    # cheaper, but overlay would then hold every frame of the main picture until the next shot.
    retime = "setpts=N/FRAME_RATE/TB,"  # frame-exact times, matching zoompan's output
    steps = [
        "[0:v]"
        + (f"select={select},{retime}" if select else retime if views else "")
        + f"split={2 + has_avatar + len(views)}[bg][game_in]"
        + ("[avatar_in]" if has_avatar else "")
        + "".join(f"[{v}_view_in]" for v in views),
        # Background: blurred at quarter size (much faster), then scaled back up and darkened.
        f"[bg]scale={w // 4}:{h // 4}:force_original_aspect_ratio=increase,"
        f"crop={w // 4}:{h // 4},boxblur=8:1,scale={w}:{h},eq=brightness=-0.15"
        + (",split[base][face_base]" if "face" in views else "[base]"),
        f"[game_in]crop={g.w}:{g.h}:{g.x}:{g.y},scale={go.w}:{go.h}:flags=lanczos[game]",
        f"[base][game]overlay={go.x}:{go.y}[with_game]",
    ]
    last = "with_game"
    if has_avatar:
        a, ao = layout.avatar_src, layout.avatar_out
        assert a is not None and ao is not None
        steps.append(
            f"[avatar_in]crop={a.w}:{a.h}:{a.x}:{a.y},scale={ao.w}:{ao.h}:flags=lanczos[avatar]"
        )
        steps.append(f"[{last}][avatar]overlay={ao.x}:{ao.y}[with_avatar]")
        last = "with_avatar"
    for view in views:
        spans = [(s.start, s.end) for s in shots if s.view == view]
        if view == "game":
            v = layout.game_view_src
            assert v is not None
            zoom = zoompan(spans, sc.game_zoom, None, 0.5, 0.5, w, h, fps)
            steps.append(f"[game_view_in]crop={v.w}:{v.h}:{v.x}:{v.y},{zoom}[game_view]")
        else:
            a, fo = layout.avatar_src, layout.face_out
            assert a is not None and fo is not None
            zoom = zoompan(spans, sc.face_zoom, sc.zoom_seconds, 0.5, sc.face_focus_y,
                           fo.w, fo.h, fps)  # fmt: skip
            steps.append(f"[face_view_in]crop={a.w}:{a.h}:{a.x}:{a.y},{zoom}[face]")
            steps.append(f"[face_base][face]overlay={fo.x}:{fo.y}[face_view]")
        on = "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in spans)
        steps.append(f"[{last}][{view}_view]overlay=0:0:enable='{on}'[with_{view}_view]")
        last = f"with_{view}_view"
    captions = f"subtitles={captions_file}," if captions_file else ""
    steps.append(f"[{last}]{captions}format=yuv420p[vout]")
    return ";".join(steps)


def zoompan(
    spans: list[tuple[float, float]],
    zoom: float,
    ramp_s: float | None,
    focus_x: float,
    focus_y: float,
    out_w: int,
    out_h: int,
    fps: float,
) -> str:
    """Scale to out_w x out_h, zooming in from 1 to `zoom` over ramp_s from the start of each
    span (over the whole span when None), towards the focus point (fractions of the picture).
    The input must already have the output's shape: zoompan stretches it otherwise."""
    terms = "+".join(
        f"between(it,{a:.3f},{b:.3f})*min(1,(it-{a:.3f})/{(ramp_s or b - a):.3f})" for a, b in spans
    )
    return (
        f"zoompan=z='1+{zoom - 1:.3f}*({terms})'"
        f":x='(iw-iw/zoom)*{focus_x:g}':y='(ih-ih/zoom)*{focus_y:g}'"
        f":d=1:s={out_w}x{out_h}:fps={fps:g}"
    )
