"""Storage package: content-addressed blob store + SQLite CaseRepository.

Persistence only. Depends on `datamodel/`; nothing below it imports from
here (CLAUDE.md package boundaries). No ORM, no migrations — by design.
"""
