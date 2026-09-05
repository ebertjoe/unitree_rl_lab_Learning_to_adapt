"""Annotated mp4 recording of a sweep, plus a seekable manifest.

The sweep runs thousands of episodes, so recording every one of them is neither
affordable nor watchable. Instead the writer produces a single continuous mp4
that shows *one* episode per parallel batch, chosen so the gaits stay balanced,
with the gait and the commanded twist burned into the frame. ``manifest.csv``
maps the timestamp of every segment back to its row in ``episodes.csv``, so a
segment of interest can be found by seeking rather than by scrubbing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:  # pillow ships with Isaac Lab, but keep the writer usable without it
    from PIL import Image, ImageDraw, ImageFont

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

MANIFEST_FIELDS = [
    "segment", "video_start_s", "video_end_s", "episode_id", "seed",
    "gait_id", "gait_name", "vx_cmd", "vy_cmd", "wz_cmd", "batch", "env_slot",
]


def _load_font(size: int):
    if not _HAS_PIL:
        return None
    for name in (
        "DejaVuSansMono-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class SweepVideoWriter:
    """Writes one continuous, annotated mp4 for a sweep run."""

    def __init__(
        self,
        path: Path | str,
        fps: int = 25,
        manifest_path: Path | str | None = None,
        font_size: int = 22,
    ):
        import imageio.v2 as imageio  # imported lazily: only needed with --video

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer = imageio.get_writer(
            str(self.path),
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=8,
            ffmpeg_log_level="error",
        )
        self._frames = 0
        self._font = _load_font(font_size)
        self._font_small = _load_font(max(12, font_size - 6))

        self.manifest_path = Path(manifest_path) if manifest_path else self.path.with_name("manifest.csv")
        self._manifest_fh = open(self.manifest_path, "w", newline="")
        self._manifest = csv.DictWriter(self._manifest_fh, fieldnames=MANIFEST_FIELDS)
        self._manifest.writeheader()
        self._segment = 0
        self._segment_start = 0

    # -- segments -----------------------------------------------------------

    def begin_segment(self) -> None:
        self._segment_start = self._frames

    def end_segment(self, *, episode_id: int, episode, batch: int, env_slot: int) -> None:
        if self._frames == self._segment_start:
            return
        self._manifest.writerow({
            "segment": self._segment,
            "video_start_s": round(self._segment_start / self.fps, 2),
            "video_end_s": round(self._frames / self.fps, 2),
            "episode_id": episode_id,
            "seed": episode.seed,
            "gait_id": episode.gait_id,
            "gait_name": episode.gait_name,
            "vx_cmd": round(episode.vx, 3),
            "vy_cmd": round(episode.vy, 3),
            "wz_cmd": round(episode.wz, 3),
            "batch": batch,
            "env_slot": env_slot,
        })
        self._manifest_fh.flush()
        self._segment += 1

    # -- frames -------------------------------------------------------------

    def add_frame(self, rgb: np.ndarray, header: str = "", lines: list[str] | None = None) -> None:
        if rgb is None:
            return
        frame = np.asarray(rgb)
        if frame.ndim != 3:
            return
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = self._annotate(frame, header, lines or [])
        # libx264 needs even dimensions
        h, w = frame.shape[:2]
        if h % 2 or w % 2:
            frame = frame[: h - h % 2, : w - w % 2]
        self._writer.append_data(frame)
        self._frames += 1

    def _annotate(self, frame: np.ndarray, header: str, lines: list[str]) -> np.ndarray:
        if not _HAS_PIL or (not header and not lines):
            return frame
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, "RGBA")
        pad = 10
        n_lines = (1 if header else 0) + len(lines)
        line_h = 28
        box_h = pad * 2 + n_lines * line_h
        draw.rectangle([0, 0, img.width, box_h], fill=(0, 0, 0, 165))
        y = pad
        if header:
            draw.text((pad, y), header, fill=(255, 235, 120), font=self._font)
            y += line_h
        for line in lines:
            draw.text((pad, y), line, fill=(235, 235, 235), font=self._font_small)
            y += line_h
        return np.asarray(img)

    # -- lifecycle ----------------------------------------------------------

    @property
    def n_segments(self) -> int:
        return self._segment

    @property
    def n_frames(self) -> int:
        return self._frames

    @property
    def duration_s(self) -> float:
        return self._frames / self.fps

    def close(self) -> None:
        try:
            self._writer.close()
        finally:
            self._manifest_fh.close()


class GaitBalancedSelector:
    """Picks which episode of a batch is filmed, keeping the gait counts even."""

    def __init__(self):
        self._counts: dict[int, int] = {}

    def pick(self, episodes) -> int:
        """Index of the least-filmed gait in ``episodes`` (ties broken by order)."""
        best_idx, best_key = 0, None
        for i, ep in enumerate(episodes):
            key = (self._counts.get(ep.gait_id, 0), i)
            if best_key is None or key < best_key:
                best_idx, best_key = i, key
        self._counts[episodes[best_idx].gait_id] = self._counts.get(episodes[best_idx].gait_id, 0) + 1
        return best_idx

    @property
    def counts(self) -> dict[int, int]:
        return dict(self._counts)
