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

---

## Tech stack (versions are pinned in `pyproject.toml`)

- Python 3.11+
- `uv` for package management and virtual envs
- `cadquery` 2.4 — installed; not used in week 1
- `manifold3d` 2.5 — boolean operations on meshes
- `trimesh` 4.5 — primary mesh data type and I/O
- `libigl` 2.5 — mesh distance queries (wall-thickness analysis)
- `open3d` 0.18 — installed; not used until week 2
- `pydantic` v2 — typed configuration
- `pytest` + `pytest-cov` — tests
- `ruff` — linting and formatting

Determinism note: every numerical / mesh library above is **pinned to an exact
version**. The golden STL hashes in `tests/fixtures/golden/` are valid only for
the pinned versions. Any version bump must be accompanied by regenerated
golden hashes and a code-version bump.

---

## Commands

```bash
# Set up
uv sync                                # install deps + create .venv
uv sync --extra dev                    # include dev tools

# Test
uv run pytest                          # full test suite
uv run pytest --cov=kernel             # with coverage
uv run pytest -k determinism           # only determinism tests

# Lint / format
uv run ruff check .
uv run ruff format .

# Generate a helmet
uv run python -m kernel.generate --config examples/baseline.yaml --out /tmp/helmet.stl
```

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
