"""Anatomy package: scan ingestion and anatomical analysis.

This package sits next to `kernel/` (geometry). It owns scan loading,
mesh cleanup, landmark annotation, registration to the canonical
reference frame, and anatomical measurements.

Package boundary (CLAUDE.md): `anatomy/` may import from `kernel/`
read-only; `kernel/` must never import from `anatomy/`.
"""
