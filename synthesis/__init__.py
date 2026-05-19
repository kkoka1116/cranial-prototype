"""Synthesis keystone: Design + ScanRecord -> patient-specific helmet.

Three separated stages — resolve (anchor names -> patient-frame geometry),
translate (semantic object -> KernelOperation list), execute (phase-
structured kernel execution) — plus a top-level `synthesize` pipeline.

Top layer: imports from `datamodel/`, `kernel/`, `anatomy/`. Nothing
imports `synthesis/` (CLAUDE.md package boundaries / dependency direction).
"""
