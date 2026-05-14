# cranial-prototype

[![CI](https://github.com/kkoka1116/cranial-prototype/actions/workflows/ci.yml/badge.svg)](https://github.com/kkoka1116/cranial-prototype/actions/workflows/ci.yml)

Deterministic geometric kernel for custom cranial remolding helmets.
Week 1 prototype — kernel only, no AI layer yet.

See [CLAUDE.md](CLAUDE.md) for full project invariants and conventions.

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pre-commit install      # install the LLM-import guard hook (invariant #1)
```

## Generate a helmet

```bash
uv run python -m kernel.generate \
    --config examples/baseline.yaml \
    --out /tmp/helmet.stl
```

Three example configs are included:
- `examples/baseline.yaml` — symmetric, no correction
- `examples/moderate_right_plagio.yaml` — right-side occipital flattening with correction
- `examples/brachycephaly.yaml` — AP-compressed head with frontal/posterior correction

The CLI exits non-zero with a structured error if the resulting mesh is non-manifold,
non-watertight, has wall thickness below the configured minimum, or exceeds the mass limit.

## Run tests

```bash
uv run pytest                     # full suite
uv run pytest --cov=kernel        # with coverage
```

## What this is (and isn't)

This is the **kernel** layer. It takes typed Pydantic config in and produces a
manifold, watertight STL out. It contains no AI, no decision-making, no
landmark detection, no semantic objects. Those layers arrive in later weeks.

The architectural invariant for the whole project: **the LLM proposes
parameters; the kernel produces geometry; never the reverse.**

## Repo conventions

- `main` is the default branch. CI runs on every push to `main` and on every
  PR. **Branch protection / rulesets are not currently enforced** — the
  feature requires GitHub Pro on private repos. Practically: keep direct
  pushes to `main` to CI-passing commits only. If we go public or upgrade,
  the ruleset payload is staged in [`docs/branch-protection.json`](docs/branch-protection.json).
- Conventional Commits (`feat:`, `fix:`, `chore:`, `test:`, `docs:`).
- The pre-commit hook ([`scripts/check_no_llm_imports.py`](scripts/check_no_llm_imports.py))
  enforces invariant #1 (no LLM SDK imports under `kernel/`). CI also runs
  the same check independently.
- The three example helmet STLs are produced fresh on every CI run and
  attached to the workflow run as artifacts (`example-helmets`).
