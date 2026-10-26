"""Where the gameplay and avatar go in the vertical frame, as an FFmpeg filter graph."""

from __future__ import annotations

from dataclasses import dataclass

from src.common.config import Rect, RenderConfig


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


def _even(value: float) -> int:
    """Video sizes and offsets must be even for yuv420p."""
    return max(2, int(round(value / 2)) * 2)


def source_box(rect: Rect, src_w: int, src_h: int) -> Box:
    w, h = min(_even(rect.w * src_w), src_w), min(_even(rect.h * src_h), src_h)
    x = min(_even(rect.x * src_w) if rect.x else 0, src_w - w)
    y = min(_even(rect.y * src_h) if rect.y else 0, src_h - h)
    return Box(x, y, w, h)


def plan_layout(rc: RenderConfig, src_w: int, src_h: int) -> Layout:
    width, height = rc.width, rc.height
    if rc.layout == "center":
        crop_w = min(_even(src_h * width / height), src_w)
        crop_h = min(_even(crop_w * height / width), src_h)
        crop = Box(_even((src_w - crop_w) / 2) if src_w > crop_w else 0, 0, crop_w, crop_h)
        out = Box(0, 0, width, height)
        return Layout(width, height, crop, out, None, None)

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
    return Layout(width, height, game, game_out, avatar, avatar_out)


def filter_graph(layout: Layout, captions_file: str | None, select: str | None = None) -> str:
    """Input 0's video to [vout]. captions_file is a bare file name, resolved from FFmpeg's
    working directory (Windows paths inside filter arguments need awkward escaping).
    select: a select expression for jump cuts, applied before anything else."""
    w, h = layout.width, layout.height
    g, go = layout.gameplay_src, layout.gameplay_out
    has_avatar = layout.avatar_src is not None and layout.avatar_out is not None
    steps = [
        "[0:v]"
        + (f"select={select},setpts=N/FRAME_RATE/TB," if select else "")
        + f"split={3 if has_avatar else 2}[bg][game_in]"
        + ("[avatar_in]" if has_avatar else ""),
        # Background: blurred at quarter size (much faster), then scaled back up and darkened.
        f"[bg]scale={w // 4}:{h // 4}:force_original_aspect_ratio=increase,"
        f"crop={w // 4}:{h // 4},boxblur=8:1,scale={w}:{h},eq=brightness=-0.15[base]",
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
    captions = f"subtitles={captions_file}," if captions_file else ""
    steps.append(f"[{last}]{captions}format=yuv420p[vout]")
    return ";".join(steps)
