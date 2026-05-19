"""Round-trip + determinism for ScanRecord composing real Week-2 anatomy types.

Builds a LandmarkSet with all 12 LandmarkName members (mixed methods), two
MeasurementResults, a populated CleanupReport, and a non-identity 4x4
registered_transform. Asserts the restored landmark_set is exactly a
LandmarkSet and a measurement value survives byte-for-byte.
"""

from __future__ import annotations

from anatomy.cleanup import CleanupReport
from anatomy.config import LANDMARK_ORDER, LandmarkName
from anatomy.landmarks import Landmark, LandmarkSet
from anatomy.measurements import MeasurementResult
from datamodel.scan_record import ScanRecord

_METHODS: tuple[str, ...] = ("manual", "snapped", "derived")


def _landmark_set() -> LandmarkSet:
    lms: dict[LandmarkName, Landmark] = {}
    for i, name in enumerate(LANDMARK_ORDER):
        lms[name] = Landmark(
            name=name,
            position_mm=(float(i), float(i) + 0.5, float(i) - 1.25),
            method=_METHODS[i % 3],
            confidence=1.0 - (i % 5) * 0.1,
            notes=f"landmark {name.value}",
        )
    return LandmarkSet(frame="canonical", landmarks=lms)


def _scan_record() -> ScanRecord:
    return ScanRecord(
        mesh_hash="sha256:" + "cd" * 32,
        source_filename="patient_0042_headscan.ply",
        landmark_set=_landmark_set(),
        measurements=[
            MeasurementResult(
                name="cephalic_index",
                value=82.375,
                unit="index",
                inputs_used=["euryon_left", "euryon_right", "glabella"],
                quality="high",
                notes="auto-computed in canonical frame",
            ),
            MeasurementResult(
                name="cvai",
                value=9.125,
                unit="percent",
                inputs_used=["diagonal_a", "diagonal_b"],
                quality="medium",
                notes="moderate asymmetry",
            ),
        ],
        cleanup_report=CleanupReport(
            was_already_manifold=False,
            duplicate_vertices_merged=314,
            degenerate_faces_removed=12,
            holes_filled=3,
            faces_removed=27,
            faces_added=58,
            normals_flipped=True,
            notes="v 50000->49686, f 100000->100031",
        ),
        registered_transform=[
            [0.998, -0.0349, 0.0523, 12.5],
            [0.0349, 0.999, 0.0, -3.25],
            [-0.0523, 0.0018, 0.998, 7.75],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def test_scan_record_round_trips_lossless() -> None:
    rec = _scan_record()
    restored = ScanRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec
    assert type(restored.landmark_set) is LandmarkSet
    assert len(restored.landmark_set.landmarks) == 12
    assert restored.measurements[0].value == 82.375
    assert restored.measurements[1].value == 9.125
    assert restored.registered_transform[0][3] == 12.5
    assert restored.landmark_set.landmarks[LandmarkName.GLABELLA].method == "manual"


def test_scan_record_serialization_is_deterministic() -> None:
    rec = _scan_record()
    assert rec.model_dump_json() == rec.model_dump_json()
    once = rec.model_dump_json()
    assert once == ScanRecord.model_validate_json(once).model_dump_json()
