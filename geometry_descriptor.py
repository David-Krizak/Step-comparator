from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


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
        }


@dataclass(frozen=True)
class GeometrySignature:
    path: str
    descriptor: GeometryDescriptor
    hash_hex: str


class GeometryHasher:
    def __init__(self, tolerance: float = 0.001, hist_bins: int = 12) -> None:
        self.tolerance = tolerance if tolerance > 0 else 0.001
        self.hist_bins = max(hist_bins, 1)

    def signature_for_file(self, path: str | Path) -> GeometrySignature:
        descriptor = self.build_descriptor(path)
        return GeometrySignature(str(path), descriptor, self.hash_descriptor(descriptor))

    def build_descriptor(self, path: str | Path) -> GeometryDescriptor:
        part_module = self._require_part_module()
        shape = self._read_shape(part_module, path)
        normalized_shape = self._normalize_solids(part_module, shape)

        volume = self._quantize_value(normalized_shape.Volume)
        area = self._quantize_value(normalized_shape.Area)
        bbox_dims = self._quantize_bbox(normalized_shape)
        num_faces = len(normalized_shape.Faces)
        num_edges = len(normalized_shape.Edges)
        inertia = self._inertia(normalized_shape)
        centroid_offset = self._centroid_offset(normalized_shape)
        face_hist = self._histogram([face.Area for face in normalized_shape.Faces])
        edge_hist = self._histogram([edge.Length for edge in normalized_shape.Edges])

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
        )

    def hash_descriptor(self, descriptor: GeometryDescriptor) -> str:
        payload = json.dumps(descriptor.to_ordered_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _require_part_module(self):
        if importlib.util.find_spec("Part") is None:
            raise RuntimeError(
                "FreeCAD Part module not found. Ensure FreeCAD libraries are available alongside the application."
            )
        import Part

        return Part

    def _read_shape(self, part_module, path: str | Path):
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        shape = part_module.Shape()
        shape.read(str(resolved))
        return shape

    def _normalize_solids(self, part_module, shape):
        solids = shape.Solids
        if not solids:
            return shape
        if len(solids) == 1:
            return solids[0]
        return part_module.makeCompound(solids)

    def _quantize_value(self, value: float) -> float:
        scale = 1 / self.tolerance
        return round(value * scale) / scale

    def _quantize_bbox(self, shape) -> list[float]:
        bbox = shape.BoundBox
        dims = sorted([bbox.XLength, bbox.YLength, bbox.ZLength])
        return [self._quantize_value(dim) for dim in dims]

    def _inertia(self, shape) -> list[float]:
        matrix = shape.MatrixOfInertia
        data = np.array(
            [
                [matrix.A11, matrix.A12, matrix.A13],
                [matrix.A21, matrix.A22, matrix.A23],
                [matrix.A31, matrix.A32, matrix.A33],
            ],
            dtype=float,
        )
        eigenvalues = np.linalg.eigvalsh(data)
        quantized = [self._quantize_value(val) for val in sorted(eigenvalues)]
        return quantized

    def _centroid_offset(self, shape) -> float:
        centroid = shape.CenterOfMass
        bbox = shape.BoundBox
        bbox_center = (
            (bbox.XMin + bbox.XMax) / 2,
            (bbox.YMin + bbox.YMax) / 2,
            (bbox.ZMin + bbox.ZMax) / 2,
        )
        offset = math.sqrt(
            (centroid.x - bbox_center[0]) ** 2
            + (centroid.y - bbox_center[1]) ** 2
            + (centroid.z - bbox_center[2]) ** 2
        )
        return self._quantize_value(offset)

    def _histogram(self, values: Iterable[float]) -> list[int]:
        values_list = list(values)
        if not values_list:
            return [0 for _ in range(self.hist_bins)]
        max_value = max(values_list)
        bin_width = max(max_value / self.hist_bins, self.tolerance)
        bins = [0 for _ in range(self.hist_bins)]
        for value in values_list:
            index = min(int(value / bin_width), self.hist_bins - 1)
            bins[index] += 1
        return bins
