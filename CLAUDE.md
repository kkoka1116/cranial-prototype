# Cranial Prototype — `CLAUDE.md`

## Mission

This repository is the deterministic geometric **kernel** for an agentic CAD
system that designs custom cranial remolding helmets. The full system will
eventually take a clinician's natural-language case description plus a 3D head
scan and produce a manufacturable helmet design with a complete audit trail.
This codebase is **not** that full system. It is the layer underneath.

**Week 1 scope:** build the deterministic kernel and nothing else. When the AI
layer arrives in week 5+, it will propose typed parameters and this kernel will
execute them. The kernel itself contains zero AI, makes zero decisions, and is
bit-for-bit reproducible.

The architecture invariant that defines this project for the next twelve
months: **the LLM proposes parameters; the deterministic kernel produces
geometry; never the reverse.** Week 1 builds the kernel side of that contract.

---

## Critical architecture invariants (non-negotiable)

1. **The kernel produces geometry; the LLM produces parameters. Never the
   reverse.** No LLM call exists in any module under `kernel/`. No string
   interpolation generating geometric code. No exceptions, ever. Enforced by a
   pre-commit hook that fails if anything under `kernel/` imports from
   `anthropic`, `openai`, or any LLM SDK.

2. **Determinism is mandatory.** Same inputs always produce bit-identical
   outputs. No random seeds without explicit seed parameters. No floating-point
   drift from non-deterministic library iteration order. Hash every STL output
   and verify reproducibility in tests.

3. **Units are millimeters everywhere.** No silent unit conversions, no inches,
   no meters. If a value is not in mm, the variable name says so
   (`volume_mm3`, `density_g_per_cm3`).

4. **The reference frame is fixed and explicit.** Origin between the ear canals
   at skull base. +X is patient's right (lateral). +Y is anterior. +Z is
   superior. Every mesh leaving any kernel function is in this frame, full
   stop.

5. **Pydantic v2 for all parameter inputs.** Even in week 1, kernel functions
   take Pydantic-validated config objects, not loose kwargs. This foreshadows
   the typed-operations pattern that arrives week 4.

6. **Manifold and watertight are hard invariants.** Every mesh leaving the
   kernel passes manifold + watertight checks or the kernel raises
   `KernelError`. No silently broken geometry ever escapes.

7. **Scans are always processed in the canonical reference frame.** The frame
   transformation happens exactly once, at load time. The transformation
   matrix is logged. No measurement, no analysis, no kernel call ever runs on
   source-frame coordinates. If you find yourself writing code that does math
   on a scan before it has been registered, stop and register first.

8. **Landmarks are anchored to mesh geometry, not free space.** When a user
   picks a landmark, snap to the nearest vertex (or barycentric coordinates on
   the nearest face). Free-floating 3D points are not valid landmarks. This
   guarantees that landmarks survive mesh transformations and remain on the
   patient surface.

9. **The Case is the single source of truth.** Every artifact the system
   produces — a mesh, a measurement, a design version, an audit event — is
   either stored inside the `Case` object or content-addressed by hash and
   referenced from the `Case`. Nothing the system produces lives outside a
   `Case`. If you find yourself writing a file to disk that isn't either a
   content-addressed blob or part of a serialized `Case`, stop.

10. **Serialization is lossless and deterministic.** A `Case` round-trips
    through JSON with zero information loss:
    `Case.model_validate_json(case.model_dump_json())` equals the original in
    every field. Serializing the same `Case` twice produces byte-identical
    JSON. This is the literal foundation of the audit trail and the eventual
    regulatory story. Every model gets a round-trip test, no exceptions.

---

## Package boundaries

- `kernel/` does **geometry only** — synthetic head, shell, trim, validation,
  STL export, CLI. No scan ingestion, no anatomical analysis.
- `anatomy/` does **scan ingestion and anatomical analysis only** — load,
  cleanup, landmarks, registration, measurements, annotation.
- `datamodel/` is the **top compositional layer** — `Case`, `Design`,
  provenance/audit/clinical/semantic types. It composes `anatomy/` and
  `kernel/` types; it never redefines them.
- `storage/` is **persistence only** — content-addressed blob store and the
  SQLite `CaseRepository`. It depends on `datamodel/`.
- **Dependency arrows point one way only:**
  `storage → datamodel → {anatomy, kernel}`, and `anatomy → kernel`.
  `kernel/` imports from none of the others. `anatomy/` imports `kernel/`
  read-only and nothing above it. `kernel/` and `anatomy/` must **never**
  import from `datamodel/` or `storage/`. Adapters that bridge layers live in
  the higher layer, never retrofitted into the lower one. This keeps the
  graph acyclic and the lower layers independently reusable.

---

## Tech stack (versions are pinned in `pyproject.toml`)

- Python 3.11+
- `uv` for package management and virtual envs
- `cadquery` 2.4 — installed; not used in week 1
- `manifold3d` 2.5 — boolean operations on meshes
- `trimesh` 4.5 — primary mesh data type and I/O
- `libigl` 2.5 — mesh distance queries (wall-thickness analysis)
- `open3d` 0.18 — installed; pinned for determinism. Not yet used in code
  (reserved for future mesh-query work).
- `pyvista` 0.44+ — interactive 3D mesh visualization + point picking
  (week-2 annotation tool; pulls in VTK transitively)
- `pymeshfix` 0.17+ — robust hole filling / non-manifold repair for raw scans
- `scipy` 1.13 — pinned for determinism (transitive dep of trimesh).
  Not directly used: week-2 registration is a hand-rolled cross-product
  orthonormal basis, not an SVD.
- `pydantic` v2 — typed configuration; week-3 also uses generic models
  (`TracedValue[T]`) and discriminated unions
- `sqlite3` — Python standard library; the week-3 case database. No ORM,
  no migrations framework — by design (see Prototype data lifecycle).
- `pytest` + `pytest-cov` — tests
- `ruff` — linting and formatting

Week 3 adds **no new third-party dependencies** — storage uses stdlib
`sqlite3` only.

Determinism note: every numerical / mesh library above is **pinned to an exact
version**. The golden STL hashes in `tests/fixtures/golden/` are valid only for
the pinned versions. Any version bump must be accompanied by regenerated
golden hashes and a code-version bump. `pyvista`/VTK is used only by the
interactive annotation tool, which is outside the deterministic pipeline; the
landmark-derivation and measurement paths must stay bit-deterministic.

---

## Commands

```bash
# Set up
uv sync                                # install deps + create .venv
uv sync --extra dev                    # include dev tools

# Test
uv run pytest                          # full test suite (kernel + anatomy)
uv run pytest --cov=kernel             # kernel coverage
uv run pytest tests/anatomy            # anatomy package only
uv run pytest --cov=anatomy            # anatomy coverage
uv run pytest -k determinism           # only determinism tests

# Lint / format
uv run ruff check .
uv run ruff format .

# Generate a helmet
uv run python -m kernel.generate --config examples/baseline.yaml --out /tmp/helmet.stl

# Annotate a scan with the twelve cranial landmarks (interactive PyVista)
uv run python -m anatomy.annotate <scan.stl> --out landmarks.json

# Data model + storage
uv run pytest tests/datamodel tests/storage    # week-3 suites
uv run python scripts/build_demo_case.py       # assemble + persist a full Case
```

---

## Prototype data lifecycle

There is **no migrations framework, by design**. During the prototype,
schema changes are handled by deleting the dev SQLite database and the blob
directory and rebuilding from `scripts/build_demo_case.py`. SQLite stores
each `Case` as serialized Pydantic JSON in one column plus metadata columns
for indexing; meshes live as content-addressed blobs on disk, referenced by
`sha256:` hash from the `Case`. Do not add an ORM or Alembic — an ORM would
muddy the determinism story and a prototype does not need migrations.

---

## Coding standards

- **Type hints required on every public function.** Internal helpers may omit
  them when the type is locally obvious.
- **Pydantic v2 models for all configs.** Kernel functions take a typed config
  object, not loose `**kwargs`.
- **No global state.** Everything flows through function arguments.
- **No LLM imports anywhere under `kernel/`.** Enforced by pre-commit hook.
- **No bare `print()`** — use the `logging` module. Ruff rule `T20` enforces
  this on `kernel/` (tests are allowed to print for debugging).
- **Tests colocated under `tests/kernel/`** mirroring the source tree.
- **Determinism:** anywhere there could be nondeterminism (set iteration, dict
  order, hash randomization), be explicit. Sort before iterating. Pass seeds.
- **Units in names.** `wall_thickness_mm`, not `wall_thickness`. `mass_g`, not
  `mass`. `density_g_per_cm3`, not `density`.

---

## Out-of-scope guardrail (week 1 is NOT building)

When tempted to start any of the below because it would "only take an hour,"
**stop**. Week 1 is small on purpose.

- ❌ No LLM calls. No `anthropic`, `openai`, or any LLM SDK imports.
- ❌ No semantic objects (`CorrectionZone`, `ReliefZone`, etc.) — week 4.
- ❌ No real scan loading — week 2.
- ❌ No `Case` / `Design` / `TracedValue` / `AuditEvent` data model — week 3.
- ❌ No anatomical landmark detection — week 2.
- ❌ No CVAI or cephalic-index calculation — week 2.
- ❌ No vent patterns or strap mounts — week 4.
- ❌ No web API, no viewer, no frontend.
- ❌ No multi-user state, no auth, no database.

The structured logging in `generate.py` emits events whose shape forward-ports
to the week-3 `AuditEvent`, but it is **not** the audit trail yet — do not
build the audit trail this week.

### Week 2 out-of-scope (anatomy ingestion only)

Week 2 builds the `anatomy/` package (scan ingestion, cleanup, landmarks,
registration, measurements, annotation). The week-1 deferrals above marked
"— week 2" for scan loading / landmark annotation / CVAI now become in-scope.
Still **not** building in week 2:

- ❌ No learned landmark detection. No PointNet, no deep learning, no
  fine-tuned models. Manual annotation only this week.
- ❌ No ICP fine alignment. Landmark-based rigid registration only
  (ICP introduces nondeterminism and is unnecessary now).
- ❌ No Pydantic `Case` / `Design` data model integration — week 3.
  `anatomy/` models are scoped to the anatomy domain only.
- ❌ No anatomy → kernel connection. The kernel still takes raw
  `(x, y, z)` correction-zone coordinates as in week 1; landmark-anchored
  semantic objects arrive week 4.
- ❌ No DICOM / CT / MRI import. 3D scan formats (STL/PLY/OBJ) only.
- ❌ No mobile/photo capture, no multi-scan time-series, no real patient
  data, no web API/frontend beyond the annotation tool.

### Week 3 out-of-scope (data model + persistence only)

Week 3 builds `datamodel/` (Case, Design, provenance/audit/clinical/semantic
types) and `storage/` (content-addressed blobs + SQLite `CaseRepository`).
The job is: define the types, persist them losslessly, prove the round-trip
is deterministic. Still **not** building in week 3:

- ❌ No LLM. No Anthropic SDK, no narrative parsing, no parameter proposal.
  `clinical_narrative` is an inert string; semantic objects are hand-authored.
- ❌ No semantic→kernel translation. `KernelOperation` is a data type only;
  the `CorrectionZone`→`offset_surface` translator is week 4.
- ❌ No constraint propagation engine. `Constraint` is a data type; recording
  a Week-1 validator result as a `Constraint` is recording, not propagation.
- ❌ No ORM, no migrations framework, no Alembic — **ever**, in the
  prototype. Schema changes = delete the dev DB + blob dir and rebuild.
- ❌ No refactor of `kernel/` or `anatomy/` internals. Additive only; all
  prior tests pass unchanged. Adapters live in `datamodel/`.
- ❌ No web API, UI, viewer, multi-user, auth, or concurrency control beyond
  SQLite defaults. No performance optimization — correctness over speed.

---

## Reference frame (mandatory for every mesh)

```
       +Z (superior)
        |
        |
        |________ +Y (anterior)
       /
      /
    +X (patient's right / lateral)

Origin: between the ear canals, at skull base.
```

Every public kernel function returns meshes in this frame. Inputs are assumed
in this frame. There is no implicit conversion anywhere.
