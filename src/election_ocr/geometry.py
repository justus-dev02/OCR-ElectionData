"""Pixel-space geometry primitives (page image coordinates at render DPI)."""

from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel


class Box(BaseModel):
    """Axis-aligned box in page-image pixels: x0,y0 top-left; x1,y1 bottom-right (exclusive)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def as_list(self) -> List[float]:
        return [round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2)]

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x0)), int(round(self.y0)), int(round(self.x1)), int(round(self.y1)))

    def pad(self, px: float, max_w: float, max_h: float) -> Box:
        return Box(
            x0=max(0.0, self.x0 - px),
            y0=max(0.0, self.y0 - px),
            x1=min(max_w, self.x1 + px),
            y1=min(max_h, self.y1 + px),
        )

    def union(self, other: Box) -> Box:
        return Box(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def horizontal_overlap(self, other: Box) -> float:
        return max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))

    def vertical_overlap(self, other: Box) -> float:
        return max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def to_pdf_points(self, dpi: int) -> List[float]:
        s = 72.0 / dpi
        return [round(self.x0 * s, 3), round(self.y0 * s, 3), round(self.x1 * s, 3), round(self.y1 * s, 3)]

    @classmethod
    def from_pdf_points(cls, x0: float, y0: float, x1: float, y1: float, dpi: int) -> Box:
        s = dpi / 72.0
        return cls(x0=x0 * s, y0=y0 * s, x1=x1 * s, y1=y1 * s)


def union_all(boxes: List[Box]) -> Box:
    if not boxes:
        raise ValueError("union_all() requires at least one box")
    out = boxes[0]
    for b in boxes[1:]:
        out = out.union(b)
    return out
