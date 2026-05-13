"""Mesh-level operations: shell construction, trim cuts, etc."""

from kernel.operations.shell import build_shell
from kernel.operations.trim import apply_trim

__all__ = ["build_shell", "apply_trim"]
