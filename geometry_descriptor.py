from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ============================================================
# Data models
# ============================================================

@dataclass(frozen=True)
class GeometryDescriptor:
    volume: float
    area: float
    bbox_dims: tuple[float, float, float]
    num_faces: int
    num_edges: int
    inertia: tuple[float, float, float]
    centroid_offset: float
    face_hist: tuple[int, ...]
    edge_hist: tuple[int, ...]
    face_type_counts: tuple[int, int, int, int, int, int, int]  # plane, cylinder, cone, sphere, torus, bspline, other
    edge_type_counts: tuple[int, int, int, int, int]            # line, circle, ellipse, bspline, other
    tri_area_hist: tuple[int, ...]
    mesh_edge_hist: tuple[int, ...]
    mesh_hash: str

    def to_ordered_dict(self) -> dict[str, object]:
        data = asdict(self)
        return {
            "volume": data["volume"],
            "area": data["area"],
            "bbox": list(self.bbox_dims),
            "faces": data["num_faces"],
            "edges": data["num_edges"],
            "inertia": list(self.inertia),
            "centroid_offset": data["centroid_offset"],
            "face_hist": list(self.face_hist),
            "edge_hist": list(self.edge_hist),
            "face_type_counts": list(self.face_type_counts),
            "edge_type_counts": list(self.edge_type_counts),
            "tri_area_hist": list(self.tri_area_hist),
            "mesh_edge_hist": list(self.mesh_edge_hist),
            "mesh_hash": self.mesh_hash,
        }


@dataclass(frozen=True)
class GeometrySignature:
    path: str
    descriptor: GeometryDescriptor
    hash_hex: str


# ============================================================
# Geometry Hasher
# ============================================================

class GeometryHasher:
    def __init__(self, tolerance: float = 0.001, hist_bins: int = 24, mesh_bins: int = 32) -> None:
        self.tolerance = tolerance if tolerance > 0 else 0.001
        self.hist_bins = max(hist_bins, 1)
        self.mesh_bins = max(mesh_bins, 1)

    # --------------------------------------------------------
    # FreeCAD loader (for 1.0.2 conda bundle or similar)
    # --------------------------------------------------------
    def _require_part_module(self):
        """
        Ensure FreeCAD's Part module is importable.

        Works with:
          - FreeCAD 1.0.2 conda bundle
          - Any FreeCAD where python.exe lives in <root>/bin
        """
        if importlib.util.find_spec("Part") is not None:
            import Part
            return Part

        exe = Path(sys.executable).resolve()
        freecad_root = exe.parent.parent  # bin -> root

        bin_dir = freecad_root / "bin"
        lib_dir = freecad_root / "lib"
        mod_dir = freecad_root / "Mod"

        if not bin_dir.is_dir() or not lib_dir.is_dir() or not mod_dir.is_dir():
            raise RuntimeError(
                "FreeCAD installation not found.\n"
                f"Tried:\n  {bin_dir}\n  {lib_dir}\n  {mod_dir}\n"
                "Run this script using FreeCAD's python.exe or adjust paths."
            )

        # Add to sys.path
        for p in (bin_dir, lib_dir, mod_dir):
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.append(p_str)

        # Add Mod subfolders
        for sub in mod_dir.iterdir():
            if sub.is_dir():
                sub_str = str(sub)
                if sub_str not in sys.path:
                    sys.path.append(sub_str)

        # Add DLL dirs to PATH
        os.environ["PATH"] = f"{bin_dir};{lib_dir};" + os.environ.get("PATH", "")

        try:
            import FreeCAD  # noqa: F401
            import Part     # noqa: F401
        except Exception as e:
            raise RuntimeError(f"Failed to import FreeCAD Part module: {e}")

        import Part
        return Part

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def signature_for_file(self, path: str | Path) -> GeometrySignature:
        descriptor = self.build_descriptor(path)
        return GeometrySignature(str(path), descriptor, self.hash_descriptor(descriptor))

    def build_descriptor(self, path: str | Path) -> GeometryDescriptor:
        Part = self._require_part_module()

        shape = self._read_shape(Part, path)
        normalized_shape = self._normalize_solids(Part, shape)

        # Basic global invariants
        volume = self._quantize_value(normalized_shape.Volume)
        area = self._quantize_value(normalized_shape.Area)
        bbox_dims = self._quantize_bbox(normalized_shape)

        num_faces = len(normalized_shape.Faces)
        num_edges = len(normalized_shape.Edges)

        inertia = self._inertia(normalized_shape)
        centroid_offset = self._centroid_offset(normalized_shape)

        # Histograms of analytic face areas and edge lengths
        face_hist = self._histogram(
            (face.Area for face in normalized_shape.Faces),
            bins=self.hist_bins,
        )
        edge_hist = self._histogram(
            (edge.Length for edge in normalized_shape.Edges),
            bins=self.hist_bins,
        )

        # Topological type counts
        face_type_counts = self._face_type_counts(normalized_shape)
        edge_type_counts = self._edge_type_counts(normalized_shape)

        # Tessellation-based features
        tri_area_hist, mesh_edge_hist, mesh_hash = self._mesh_features(normalized_shape)

        return GeometryDescriptor(
            volume=volume,
            area=area,
            bbox_dims=tuple(bbox_dims),
            num_faces=num_faces,
            num_edges=num_edges,
            inertia=tuple(inertia),
            centroid_offset=centroid_offset,
            face_hist=tuple(face_hist),
            edge_hist=tuple(edge_hist),
            face_type_counts=tuple(face_type_counts),
            edge_type_counts=tuple(edge_type_counts),
            tri_area_hist=tuple(tri_area_hist),
            mesh_edge_hist=tuple(mesh_edge_hist),
            mesh_hash=mesh_hash,
        )

    # --------------------------------------------------------
    # Hashing
    # --------------------------------------------------------
    def hash_descriptor(self, descriptor: GeometryDescriptor) -> str:
        payload = json.dumps(
            descriptor.to_ordered_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # --------------------------------------------------------
    # FreeCAD geometry helpers
    # --------------------------------------------------------
    def _read_shape(self, part_module, path: str | Path):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)

        shape = part_module.Shape()
        shape.read(str(p))
        return shape

    def _normalize_solids(self, part_module, shape):
        solids = shape.Solids
        if not solids:
            return shape
        if len(solids) == 1:
            return solids[0]
        return part_module.makeCompound(solids)

    # --------------------------------------------------------
    # Numeric utilities
    # --------------------------------------------------------
    def _quantize_value(self, value: float) -> float:
        scale = 1 / self.tolerance
        return round(value * scale) / scale

    def _quantize_bbox(self, shape) -> list[float]:
        bb = shape.BoundBox
        dims = sorted([bb.XLength, bb.YLength, bb.ZLength])
        return [self._quantize_value(d) for d in dims]

    def _inertia(self, shape) -> list[float]:
        m = shape.MatrixOfInertia
        matrix = np.array(
            [
                [m.A11, m.A12, m.A13],
                [m.A21, m.A22, m.A23],
                [m.A31, m.A32, m.A33],
            ]
        )
        eigenvalues = np.linalg.eigvalsh(matrix)
        return [self._quantize_value(v) for v in sorted(eigenvalues)]

    def _centroid_offset(self, shape) -> float:
        c = shape.CenterOfMass
        bb = shape.BoundBox
        center = (
            (bb.XMin + bb.XMax) / 2,
            (bb.YMin + bb.YMax) / 2,
            (bb.ZMin + bb.ZMax) / 2,
        )
        dx = c.x - center[0]
        dy = c.y - center[1]
        dz = c.z - center[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        return self._quantize_value(dist)

    def _histogram(self, values: Iterable[float], bins: int) -> list[int]:
        vals = [v for v in values if v is not None]
        if not vals:
            return [0] * bins

        max_val = max(vals)
        if max_val <= 0:
            return [0] * bins

        bin_width = max(max_val / bins, self.tolerance)
        hist = [0] * bins

        for v in vals:
            idx = min(int(v / bin_width), bins - 1)
            hist[idx] += 1

        return hist

    # --------------------------------------------------------
    # Face / edge type counts
    # --------------------------------------------------------
    def _face_type_counts(self, shape) -> list[int]:
        """
        Count face surface types by Surface.__class__.__name__.

        Returns [plane, cylinder, cone, sphere, torus, bspline, other]
        """
        counts = [0, 0, 0, 0, 0, 0, 0]
        for face in shape.Faces:
            try:
                surf = face.Surface
                name = surf.__class__.__name__.lower()
            except Exception:
                counts[6] += 1
                continue

            if "plane" in name:
                counts[0] += 1
            elif "cylind" in name:
                counts[1] += 1
            elif "cone" in name:
                counts[2] += 1
            elif "sphere" in name:
                counts[3] += 1
            elif "torus" in name:
                counts[4] += 1
            elif "bspline" in name or "bezier" in name:
                counts[5] += 1
            else:
                counts[6] += 1

        return counts

    def _edge_type_counts(self, shape) -> list[int]:
        """
        Count edge curve types by Curve.__class__.__name__.

        Returns [line, circle, ellipse, bspline, other]
        """
        counts = [0, 0, 0, 0, 0]
        for edge in shape.Edges:
            try:
                curve = edge.Curve
                name = curve.__class__.__name__.lower()
            except Exception:
                counts[4] += 1
                continue

            if "line" in name:
                counts[0] += 1
            elif "circle" in name:
                counts[1] += 1
            elif "ellipse" in name:
                counts[2] += 1
            elif "bspline" in name or "bezier" in name:
                counts[3] += 1
            else:
                counts[4] += 1

        return counts

    # --------------------------------------------------------
    # Tessellation-based features
    # --------------------------------------------------------
    def _mesh_features(self, shape) -> tuple[list[int], list[int], str]:
        """
        Tessellate the shape and compute:
          - histogram of triangle areas
          - histogram of mesh edge lengths
          - mesh hash based on quantized areas + edge lengths
        """
        deflection = max(self.tolerance, 0.05)

        try:
            pts, tris = shape.tessellate(deflection)
        except Exception:
            empty_hist_tri = [0] * self.mesh_bins
            empty_hist_edge = [0] * self.mesh_bins
            return empty_hist_tri, empty_hist_edge, "mesh_tessellation_failed"

        if not pts or not tris:
            empty_hist_tri = [0] * self.mesh_bins
            empty_hist_edge = [0] * self.mesh_bins
            return empty_hist_tri, empty_hist_edge, "mesh_empty"

        pts_arr = np.array(pts, dtype=float)

        # Triangle areas
        tri_areas: list[float] = []
        for i1, i2, i3 in tris:
            p1 = pts_arr[i1]
            p2 = pts_arr[i2]
            p3 = pts_arr[i3]
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(cross)
            tri_areas.append(area)

        tri_area_hist = self._histogram(tri_areas, bins=self.mesh_bins)

        # Mesh edges (unique)
        edges = set()
        for i1, i2, i3 in tris:
            edges.add(tuple(sorted((i1, i2))))
            edges.add(tuple(sorted((i2, i3))))
            edges.add(tuple(sorted((i3, i1))))

        edge_lengths: list[float] = []
        for i, j in edges:
            p1 = pts_arr[i]
            p2 = pts_arr[j]
            length = float(np.linalg.norm(p2 - p1))
            edge_lengths.append(length)

        mesh_edge_hist = self._histogram(edge_lengths, bins=self.mesh_bins)

        # Build a mesh hash from quantized tri areas + edge lengths
        tokens: list[str] = []

        def quantize_vals(values: Sequence[float]) -> list[int]:
            if not values:
                return []
            scale = 1.0 / self.tolerance
            return sorted(int(round(v * scale)) for v in values)

        q_tri = quantize_vals(tri_areas)
        q_edge = quantize_vals(edge_lengths)

        for v in q_tri:
            tokens.append(f"A{v}")
        for v in q_edge:
            tokens.append(f"E{v}")

        mesh_hash = self._hash_tokens(tokens)

        return tri_area_hist, mesh_edge_hist, mesh_hash

    def _hash_tokens(self, tokens: Iterable[str]) -> str:
        digest = hashlib.sha256()
        for tok in sorted(tokens):
            digest.update(tok.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    # --------------------------------------------------------
    # Strict verification using boolean differences
    # --------------------------------------------------------
    def verify_exact_match(
        self,
        path_a: str | Path,
        path_b: str | Path,
        volume_tol: float | None = None,
        area_tol: float | None = None,
    ) -> tuple[bool, dict[str, float]]:
        """
        Slow but strict check: load A and B, normalize solids,
        do boolean cuts A-B and B-A, and inspect remaining volumes.

        Returns (is_exact_match, metrics_dict).

        If boolean operations fail, we err on the SAFE side and
        return (False, metrics).
        """
        Part = self._require_part_module()

        shape_a = self._read_shape(Part, path_a)
        shape_b = self._read_shape(Part, path_b)

        solid_a = self._normalize_solids(Part, shape_a)
        solid_b = self._normalize_solids(Part, shape_b)

        vol_a = float(solid_a.Volume)
        vol_b = float(solid_b.Volume)
        area_a = float(solid_a.Area)
        area_b = float(solid_b.Area)

        if volume_tol is None:
            volume_tol = max(self.tolerance**3, 1e-9)
        if area_tol is None:
            area_tol = max(self.tolerance**2, 1e-8)

        try:
            cut_ab = solid_a.cut(solid_b)
            cut_ba = solid_b.cut(solid_a)
            vol_ab = float(cut_ab.Volume)
            vol_ba = float(cut_ba.Volume)
        except Exception:
            metrics = {
                "vol_a": vol_a,
                "vol_b": vol_b,
                "area_a": area_a,
                "area_b": area_b,
                "vol_ab": float("nan"),
                "vol_ba": float("nan"),
            }
            return False, metrics

        metrics = {
            "vol_a": vol_a,
            "vol_b": vol_b,
            "area_a": area_a,
            "area_b": area_b,
            "vol_ab": vol_ab,
            "vol_ba": vol_ba,
        }

        same_volume = abs(vol_a - vol_b) <= volume_tol
        no_diff_ab = abs(vol_ab) <= volume_tol
        no_diff_ba = abs(vol_ba) <= volume_tol

        is_exact = same_volume and no_diff_ab and no_diff_ba
        return is_exact, metrics

    # --------------------------------------------------------
    # Export difference shapes for visual inspection
    # --------------------------------------------------------
    def export_difference_shapes(
        self,
        path_a: str | Path,
        path_b: str | Path,
        out_dir: Path,
    ) -> dict[str, Path]:
        """
        Export:
          - A_clean.step     (normalized solid A)
          - B_clean.step     (normalized solid B)
          - A_minus_B.step   (A - B)
          - B_minus_A.step   (B - A)
        """
        Part = self._require_part_module()

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        shape_a = self._read_shape(Part, path_a)
        shape_b = self._read_shape(Part, path_b)

        solid_a = self._normalize_solids(Part, shape_a)
        solid_b = self._normalize_solids(Part, shape_b)

        cut_ab = solid_a.cut(solid_b)
        cut_ba = solid_b.cut(solid_a)

        files: dict[str, Path] = {
            "A_clean": out_dir / "A_clean.step",
            "B_clean": out_dir / "B_clean.step",
            "A_minus_B": out_dir / "A_minus_B.step",
            "B_minus_A": out_dir / "B_minus_A.step",
        }

        solid_a.exportStep(str(files["A_clean"]))
        solid_b.exportStep(str(files["B_clean"]))
        cut_ab.exportStep(str(files["A_minus_B"]))
        cut_ba.exportStep(str(files["B_minus_A"]))

        return files
