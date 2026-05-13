"""Deterministic binary STL writer.

Binary STL format:
    80-byte header (we zero it — trimesh sometimes writes the version
                    string into the header, which would break determinism
                    across trimesh versions)
    uint32 triangle count
    Per triangle:
        3 × float32 normal
        3 × float32 vertex 0
        3 × float32 vertex 1
        3 × float32 vertex 2
        uint16 attribute byte count (always 0)

We write this directly with numpy/struct so the byte layout is fully under
our control. This guarantees byte-identical output across runs for the same
input mesh.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import trimesh

STL_HEADER_SIZE = 80
STL_TRIANGLE_DTYPE = np.dtype(
    [
        ("normal", "<f4", 3),
        ("v0", "<f4", 3),
        ("v1", "<f4", 3),
        ("v2", "<f4", 3),
        ("attr", "<u2"),
    ]
)


def write_stl_bytes(mesh: trimesh.Trimesh) -> bytes:
    """Serialize a mesh to deterministic binary STL bytes."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    n_tri = faces.shape[0]

    triangles = np.zeros(n_tri, dtype=STL_TRIANGLE_DTYPE)
    tri_verts = verts[faces]  # (n_tri, 3, 3)

    # Compute normals from vertices — using the mesh's stored normals would
    # introduce a dependency on however trimesh decided to compute them.
    # Doing it ourselves makes the output completely self-contained.
    v0 = tri_verts[:, 0]
    v1 = tri_verts[:, 1]
    v2 = tri_verts[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    # Avoid division by zero on degenerate triangles (should not occur in a
    # validated mesh, but be safe).
    norm_len = np.where(norm_len < 1e-20, 1.0, norm_len)
    normals = (normals / norm_len).astype(np.float32)

    triangles["normal"] = normals
    triangles["v0"] = v0.astype(np.float32)
    triangles["v1"] = v1.astype(np.float32)
    triangles["v2"] = v2.astype(np.float32)
    # attr left as zero.

    header = b"\x00" * STL_HEADER_SIZE
    count = struct.pack("<I", n_tri)
    return header + count + triangles.tobytes(order="C")


def write_stl_to_path(mesh: trimesh.Trimesh, path: Path | str) -> str:
    """Write the STL to disk and return its SHA-256 hex digest."""
    data = write_stl_bytes(mesh)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def sha256_of_mesh_stl(mesh: trimesh.Trimesh) -> str:
    """Compute the SHA-256 hex digest of the mesh's STL serialization."""
    return hashlib.sha256(write_stl_bytes(mesh)).hexdigest()
