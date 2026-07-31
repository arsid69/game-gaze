"""3D pointing: smoothing, camera rays, ray/shape intersection, scene registry.

The pipeline is: hand landmark -> 2D normalised device coordinate -> smoothing
-> camera ray -> nearest intersected scene object. That is the same thing a
mouse cursor does, except the cursor is a fingertip.

Coordinates
-----------
* **NDC**   x, y in [-1, 1], +x right, +y up, (0, 0) = screen centre.
* **World** right-handed, whatever units the game uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

_EPS = 1e-9


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else v


# --------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------
class OneEuroFilter:
    """1-D One Euro filter (Casiez et al., 2012).

    A plain low-pass filter forces a choice between jitter and lag. This one
    adapts: it filters hard when the hand is still (killing tremor) and barely
    at all when the hand moves fast (keeping the pointer responsive).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, _EPS))
        return 1.0 / (1.0 + tau / max(dt, _EPS))

    def reset(self) -> None:
        self._x_prev = self._t_prev = None
        self._dx_prev = 0.0

    def __call__(self, x: float, timestamp: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev, self._t_prev = x, timestamp
            return x

        dt = timestamp - self._t_prev
        if dt <= 0:
            return self._x_prev

        dx = (x - self._x_prev) / dt
        dx_hat = self._dx_prev + self._alpha(self.d_cutoff, dt) * (dx - self._dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self._x_prev + self._alpha(cutoff, dt) * (x - self._x_prev)

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, timestamp
        return x_hat


class OneEuroFilter2D:
    """Two independent One Euro filters for a screen-space point."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0):
        self.x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.y = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def __call__(self, x: float, y: float, timestamp: float) -> tuple[float, float]:
        return self.x(x, timestamp), self.y(y, timestamp)

    def reset(self) -> None:
        self.x.reset()
        self.y.reset()


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.direction = _unit(np.asarray(self.direction, dtype=np.float64))

    def at(self, t: float) -> np.ndarray:
        """Point ``t`` units along the ray."""
        return self.origin + self.direction * t


def intersect_sphere(ray: Ray, center: np.ndarray, radius: float) -> Optional[float]:
    """Distance to the nearest hit in front of the ray origin, else None."""
    oc = ray.origin - np.asarray(center, dtype=np.float64)
    b = 2.0 * float(np.dot(oc, ray.direction))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    for t in ((-b - root) * 0.5, (-b + root) * 0.5):
        if t > _EPS:
            return t
    return None


def intersect_aabb(ray: Ray, center: np.ndarray, half_extents: np.ndarray) -> Optional[float]:
    """Slab-method ray/axis-aligned-box test."""
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half_extents, dtype=np.float64)
    lo, hi = center - half, center + half

    t_min, t_max = -np.inf, np.inf
    for axis in range(3):
        d = ray.direction[axis]
        o = ray.origin[axis]
        if abs(d) < _EPS:
            if o < lo[axis] or o > hi[axis]:
                return None            # parallel and outside this slab
            continue
        t1, t2 = (lo[axis] - o) / d, (hi[axis] - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min, t_max = max(t_min, t1), min(t_max, t2)
        if t_min > t_max:
            return None

    if t_max < _EPS:
        return None
    return t_min if t_min > _EPS else t_max


@dataclass
class Camera:
    """Pinhole camera, enough for ray casting and the demo's renderer."""

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    forward: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    fov_y_deg: float = 60.0
    aspect: float = 4.0 / 3.0
    near: float = 0.1

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.forward = _unit(np.asarray(self.forward, dtype=np.float64))
        up = np.asarray(self.up, dtype=np.float64)
        self.right = _unit(np.cross(self.forward, up))
        self.up = np.cross(self.right, self.forward)   # re-orthogonalised

    @property
    def _tan_half_fov(self) -> float:
        return math.tan(math.radians(self.fov_y_deg) * 0.5)

    def look_at(self, target: Iterable[float]) -> None:
        self.forward = _unit(np.asarray(target, dtype=np.float64) - self.position)
        self.right = _unit(np.cross(self.forward, np.array([0.0, 1.0, 0.0])))
        self.up = np.cross(self.right, self.forward)

    def ray_from_ndc(self, ndc_x: float, ndc_y: float) -> Ray:
        """Ray through a normalised device coordinate; this is the 'laser'."""
        t = self._tan_half_fov
        direction = (self.forward
                     + self.right * (ndc_x * t * self.aspect)
                     + self.up * (ndc_y * t))
        return Ray(self.position.copy(), direction)

    def world_to_ndc(self, point: Iterable[float]) -> Optional[tuple[float, float, float]]:
        """Project a world point. Returns ``(ndc_x, ndc_y, depth)`` or None if behind."""
        d = np.asarray(point, dtype=np.float64) - self.position
        z = float(np.dot(d, self.forward))
        if z <= self.near:
            return None
        t = self._tan_half_fov
        x = float(np.dot(d, self.right)) / (z * t * self.aspect)
        y = float(np.dot(d, self.up)) / (z * t)
        return x, y, z


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------
@dataclass
class SceneObject:
    """A thing the player can point at.

    Give it a ``radius`` for a sphere collider or ``half_extents`` for a box.
    ``payload`` is free for the game to hang its own object off.
    """

    id: str
    position: np.ndarray
    radius: float = 0.5
    half_extents: Optional[np.ndarray] = None
    selectable: bool = True
    grabbable: bool = True
    color: tuple[int, int, int] = (200, 200, 200)
    payload: object = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.half_extents is not None:
            self.half_extents = np.asarray(self.half_extents, dtype=np.float64)

    @property
    def bounding_radius(self) -> float:
        """Radius of a sphere that encloses the object, used by grab assist."""
        if self.half_extents is not None:
            return float(np.linalg.norm(self.half_extents))
        return float(self.radius)

    def intersect(self, ray: Ray) -> Optional[float]:
        if self.half_extents is not None:
            return intersect_aabb(ray, self.position, self.half_extents)
        return intersect_sphere(ray, self.position, self.radius)


@dataclass
class Hit:
    object: SceneObject
    distance: float
    point: np.ndarray


class Scene:
    """Registry of selectable objects plus the current selection."""

    def __init__(self, objects: Iterable[SceneObject] = ()):
        self._objects: dict[str, SceneObject] = {o.id: o for o in objects}
        self.selected: set[str] = set()

    # -- contents ----------------------------------------------------------
    def add(self, obj: SceneObject) -> SceneObject:
        self._objects[obj.id] = obj
        return obj

    def remove(self, object_id: str) -> None:
        self._objects.pop(object_id, None)
        self.selected.discard(object_id)

    def get(self, object_id: str) -> Optional[SceneObject]:
        return self._objects.get(object_id)

    @property
    def objects(self) -> list[SceneObject]:
        return list(self._objects.values())

    def ids(self) -> list[str]:
        return list(self._objects.keys())

    def __len__(self) -> int:
        return len(self._objects)

    # -- picking -----------------------------------------------------------
    def raycast(self, ray: Ray, max_distance: float = float("inf")) -> Optional[Hit]:
        """Nearest selectable object along ``ray``."""
        best: Optional[Hit] = None
        for obj in self._objects.values():
            if not obj.selectable:
                continue
            t = obj.intersect(ray)
            if t is None or t > max_distance:
                continue
            if best is None or t < best.distance:
                best = Hit(obj, t, ray.at(t))
        return best

    def pick(self, ray: Ray, assist: float = 1.0) -> Optional[Hit]:
        """Like :meth:`raycast`, but forgives a near miss.

        Hand tracking is a few pixels noisier than a mouse, so requiring the ray
        to land exactly on an object makes grabbing feel unreliable. With
        ``assist > 1`` an object also counts as picked when the ray passes
        within that multiple of its bounding radius. A true hit always wins.
        """
        exact = self.raycast(ray)
        if exact is not None or assist <= 1.0:
            return exact

        best: Optional[Hit] = None
        for obj in self._objects.values():
            if not obj.selectable:
                continue
            to_obj = obj.position - ray.origin
            along = float(np.dot(to_obj, ray.direction))
            if along <= _EPS:
                continue                       # behind the camera
            perpendicular = float(np.linalg.norm(to_obj - ray.direction * along))
            if perpendicular > obj.bounding_radius * assist:
                continue
            if best is None or along < best.distance:
                best = Hit(obj, along, ray.at(along))
        return best

    # -- selection ---------------------------------------------------------
    def select(self, object_id: str, exclusive: bool = True) -> None:
        if exclusive:
            self.selected.clear()
        if object_id in self._objects:
            self.selected.add(object_id)

    def deselect(self, object_id: str) -> None:
        self.selected.discard(object_id)

    def clear_selection(self) -> None:
        self.selected.clear()

    def is_selected(self, object_id: str) -> bool:
        return object_id in self.selected


def image_to_ndc(x: float, y: float) -> tuple[float, float]:
    """MediaPipe image coords ([0,1], y down) -> NDC ([-1,1], y up)."""
    return 2.0 * x - 1.0, 1.0 - 2.0 * y


def ndc_to_pixel(ndc_x: float, ndc_y: float, width: int, height: int) -> tuple[int, int]:
    """NDC -> integer pixel coordinates, for drawing overlays."""
    return int(round((ndc_x + 1.0) * 0.5 * width)), \
           int(round((1.0 - ndc_y) * 0.5 * height))
