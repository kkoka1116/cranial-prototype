"""Interactive landmark annotation tool.

    uv run python -m anatomy.annotate <scan.stl> --out landmarks.json

Opens a PyVista window with the scan rendered. The text panel always shows
(a) the landmark to pick next, (b) how many remain, (c) the keyboard
shortcuts. Click the mesh to place the current landmark (snapped to the
surface); keys: n = skip/next, b = back/revise, r = restart, s = save+quit.

Landmarks captured here are in the *source* (raw-scan) frame — registration
to the canonical frame happens downstream, not in this tool (invariant #7).

PyVista/VTK is imported lazily inside run_annotation_tool so the headless
path (annotate_programmatic, used by tests and any non-GUI caller) never
needs a display.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import ArrayLike

from anatomy.config import LANDMARK_HINTS, LANDMARK_ORDER, REGISTRATION_LANDMARKS, LandmarkName
from anatomy.errors import AnatomyError
from anatomy.io import load_scan
from anatomy.landmarks import Landmark, LandmarkSet, snap_to_surface

logger = logging.getLogger("anatomy")


class AnnotationState:
    """Mutable cursor over LANDMARK_ORDER plus the picks collected so far.

    Pure logic, no PyVista — unit-testable without a display.
    """

    def __init__(self) -> None:
        self._order: tuple[LandmarkName, ...] = LANDMARK_ORDER
        self._index: int = 0
        self.picks: dict[LandmarkName, Landmark] = {}

    @property
    def done(self) -> bool:
        return self._index >= len(self._order)

    @property
    def current(self) -> LandmarkName | None:
        if self.done:
            return None
        return self._order[self._index]

    def remaining(self) -> int:
        return len(self._order) - self._index

    def record(self, landmark: Landmark) -> None:
        """Store a pick for the current landmark and advance."""
        if self.done:
            return
        self.picks[self._order[self._index]] = landmark
        self._index += 1

    def skip(self) -> None:
        if not self.done:
            self._index += 1

    def back(self) -> None:
        if self._index > 0:
            self._index -= 1
            self.picks.pop(self._order[self._index], None)

    def restart(self) -> None:
        self._index = 0
        self.picks.clear()

    def missing_required(self) -> list[LandmarkName]:
        return [n for n in REGISTRATION_LANDMARKS if n not in self.picks]

    def to_landmark_set(self) -> LandmarkSet:
        """Build a source-frame LandmarkSet. Raises if a required landmark
        (the four registration landmarks) was skipped."""
        missing = self.missing_required()
        if missing:
            raise AnatomyError(
                "cannot save: missing required registration landmarks: "
                + ", ".join(m.value for m in missing)
            )
        return LandmarkSet(frame="source", landmarks=dict(self.picks))

    def status_text(self) -> str:
        if self.done:
            miss = self.missing_required()
            tail = (
                "ALL PICKED — press [s] to save+quit"
                if not miss
                else f"MISSING REQUIRED: {', '.join(m.value for m in miss)}"
            )
            return f"Done ({len(self.picks)}/12 picked). {tail}\n[b]ack  [r]estart  [s]ave"
        name = self.current
        assert name is not None
        return (
            f"Click: {name.value.upper()} — {LANDMARK_HINTS[name]}\n"
            f"{self.remaining()} landmark(s) remaining "
            f"({len(self.picks)}/12 picked)\n"
            f"[click]=place  [n]=skip  [b]=back  [r]=restart  [s]=save+quit"
        )


def annotate_programmatic(
    mesh: trimesh.Trimesh,
    picks: Sequence[tuple[LandmarkName, ArrayLike]],
) -> LandmarkSet:
    """Headless annotation: snap each (name, raw_point) to the surface and
    build a source-frame LandmarkSet. Used by tests and any non-GUI caller.
    No PyVista / display required.
    """
    state = AnnotationState()
    by_name = {name: np.asarray(pt, dtype=np.float64) for name, pt in picks}
    for name in LANDMARK_ORDER:
        if name not in by_name:
            state.skip()
            continue
        snap = snap_to_surface(by_name[name], mesh)
        state.record(
            Landmark(
                name=name,
                position_mm=snap.position_mm,
                method="manual",
                confidence=1.0,
                notes=f"snapped (d={snap.distance_mm:.3f} mm) from picked point",
            )
        )
    return state.to_landmark_set()


def run_annotation_tool(scan_path: Path, out_path: Path) -> int:  # pragma: no cover
    """Interactive PyVista session. Returns a process exit code.

    Not unit-tested (requires a display); the pure logic lives in
    AnnotationState and annotate_programmatic, which are tested headlessly.
    """
    import pyvista as pv  # lazy: no display needed for the headless path

    mesh = load_scan(scan_path)
    pv_mesh = pv.wrap(mesh)
    state = AnnotationState()

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color="lightgray", show_edges=False)
    panel = plotter.add_text(state.status_text(), font_size=11, name="panel")

    def refresh() -> None:
        plotter.remove_actor(panel)
        plotter.add_text(state.status_text(), font_size=11, name="panel")
        plotter.render()

    def on_pick(point, *_args) -> None:
        if state.done or point is None:
            return
        snap = snap_to_surface(np.asarray(point, dtype=np.float64), mesh)
        name = state.current
        assert name is not None
        state.record(
            Landmark(
                name=name,
                position_mm=snap.position_mm,
                method="manual",
                confidence=1.0,
                notes=f"snapped (d={snap.distance_mm:.3f} mm)",
            )
        )
        plotter.add_mesh(
            pv.PolyData(np.asarray([snap.position_mm])),
            color="red",
            point_size=14,
            render_points_as_spheres=True,
            name=f"lm_{name.value}",
        )
        refresh()

    plotter.enable_point_picking(
        callback=on_pick, show_message=False, use_mesh=True, left_clicking=True
    )

    def do_skip() -> None:
        state.skip()
        refresh()

    def do_back() -> None:
        state.back()
        refresh()

    def do_restart() -> None:
        state.restart()
        for name in LANDMARK_ORDER:
            try:
                plotter.remove_actor(f"lm_{name.value}")
            except Exception:
                pass
        refresh()

    def do_save() -> None:
        try:
            lm_set = state.to_landmark_set()
        except AnatomyError as exc:
            plotter.remove_actor(panel)
            plotter.add_text(f"SAVE BLOCKED: {exc}", font_size=11, name="panel", color="red")
            plotter.render()
            return
        Path(out_path).write_text(lm_set.to_json(), encoding="utf-8")
        logger.info(
            "landmarks saved",
            extra={
                "anatomy_event": {
                    "operation": "annotate_save",
                    "out_path": str(out_path),
                    "n_landmarks": len(lm_set.landmarks),
                }
            },
        )
        plotter.close()

    plotter.add_key_event("n", do_skip)
    plotter.add_key_event("b", do_back)
    plotter.add_key_event("r", do_restart)
    plotter.add_key_event("s", do_save)
    plotter.show()
    return 0 if Path(out_path).is_file() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anatomy.annotate",
        description="Interactively annotate the twelve cranial landmarks on a scan.",
    )
    parser.add_argument("scan", type=Path, help="Path to the scan (STL/PLY/OBJ).")
    parser.add_argument("--out", required=True, type=Path, help="Output landmarks JSON path.")
    args = parser.parse_args(argv)
    try:
        return run_annotation_tool(args.scan, args.out)
    except AnatomyError as exc:  # pragma: no cover
        logger.error(
            "annotation failed",
            extra={"anatomy_event": {"operation": "annotate", "error": str(exc)}},
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
