import json
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, scrolledtext

from geometry_descriptor import GeometryHasher, GeometrySignature


@dataclass
class ComparisonResult:
    summary_a: GeometrySignature
    summary_b: GeometrySignature


class StepComparator:
    def __init__(self, tolerance: float) -> None:
        self.hasher = GeometryHasher(tolerance)

    def compare(self, path_a: str, path_b: str) -> ComparisonResult:
        summary_a = self.hasher.signature_for_file(path_a)
        summary_b = self.hasher.signature_for_file(path_b)
        return ComparisonResult(summary_a, summary_b)


class ComparatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("STEP Comparator")
        self.geometry("820x700")
        self._build_widgets()

    def _build_widgets(self) -> None:
        header = tk.Label(self, text="STEP Geometry Comparator", font=("Segoe UI", 18, "bold"))
        header.pack(pady=(12, 6))

        description = tk.Label(
            self,
            text=(
                "Compare two STEP files using geometry descriptors (volume, area, inertia, and more). "
                "Descriptors are quantized using the tolerance to build a stable geometry hash."
            ),
            wraplength=760,
            justify="left",
        )
        description.pack(pady=(0, 12))

        form = tk.Frame(self)
        form.pack(fill="x", padx=14, pady=(0, 10))

        # File A
        tk.Label(form, text="File A:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.file_a_var = tk.StringVar()
        tk.Entry(form, textvariable=self.file_a_var, width=70).grid(row=0, column=1, padx=4, pady=4)
        tk.Button(form, text="Browse", command=lambda: self._choose_file(self.file_a_var)).grid(row=0, column=2, padx=4, pady=4)

        # File B
        tk.Label(form, text="File B:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        self.file_b_var = tk.StringVar()
        tk.Entry(form, textvariable=self.file_b_var, width=70).grid(row=1, column=1, padx=4, pady=4)
        tk.Button(form, text="Browse", command=lambda: self._choose_file(self.file_b_var)).grid(row=1, column=2, padx=4, pady=4)

        # Tolerance
        tk.Label(form, text="Tolerance:").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        self.tolerance_var = tk.StringVar(value="0.001")
        tk.Entry(form, textvariable=self.tolerance_var, width=15).grid(row=2, column=1, sticky="w", padx=4, pady=4)

        tk.Button(self, text="Compare", command=self._compare).pack(pady=6)
        tk.Button(self, text="Save report as JSON", command=self._save_report).pack(pady=2)

        self.output = scrolledtext.ScrolledText(self, wrap="word", width=100, height=24)
        self.output.pack(padx=12, pady=10, fill="both", expand=True)

    def _choose_file(self, variable: tk.StringVar) -> None:
        filename = filedialog.askopenfilename(filetypes=[("STEP files", "*.step *.stp"), ("All files", "*.*")])
        if filename:
            variable.set(filename)

    def _parse_tolerance(self) -> float | None:
        try:
            tol = float(self.tolerance_var.get())
            if tol <= 0:
                raise ValueError
            return tol
        except ValueError:
            messagebox.showerror("Invalid tolerance", "Please enter a positive numeric tolerance.")
            return None

    def _compare(self) -> None:
        file_a = self.file_a_var.get().strip()
        file_b = self.file_b_var.get().strip()
        tol = self._parse_tolerance()
        if not tol:
            return
        if not file_a or not file_b:
            messagebox.showwarning("Files missing", "Please select both STEP files before comparing.")
            return

        comparator = StepComparator(tol)
        try:
            result = comparator.compare(file_a, file_b)
        except FileNotFoundError as exc:
            messagebox.showerror("File not found", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface errors to the user
            messagebox.showerror("Comparison failed", f"Unexpected error: {exc}")
            return

        self._display_result(result)
        self.latest_result = result

    def _display_result(self, result: ComparisonResult) -> None:
        self.output.delete("1.0", tk.END)

        def fmt_descriptor(summary: GeometrySignature) -> str:
            descriptor = summary.descriptor
            inertia = ", ".join(f"{v:.3f}" for v in descriptor.inertia)
            bbox = ", ".join(f"{v:.3f}" for v in descriptor.bbox_dims)
            face_hist = ", ".join(str(v) for v in descriptor.face_hist)
            edge_hist = ", ".join(str(v) for v in descriptor.edge_hist)
            return (
                f"Path: {summary.path}\n"
                f"Geometry hash: {summary.hash_hex}\n"
                f"Volume: {descriptor.volume}\n"
                f"Surface area: {descriptor.area}\n"
                f"BBox dims (sorted): {bbox}\n"
                f"Faces: {descriptor.num_faces}, Edges: {descriptor.num_edges}\n"
                f"Principal inertia: {inertia}\n"
                f"Centroid offset (from bbox center): {descriptor.centroid_offset}\n"
                f"Face area histogram: [{face_hist}]\n"
                f"Edge length histogram: [{edge_hist}]\n"
            )

        lines = ["== Geometry summaries ==\n", "File A:\n", fmt_descriptor(result.summary_a), "File B:\n", fmt_descriptor(result.summary_b)]

        if result.summary_a.hash_hex == result.summary_b.hash_hex:
            lines.append("\nGeometries match (hashes are identical).\n")
        else:
            lines.append("\nGeometries differ (hashes are not identical).\n")

        lines.append("Hashes are derived from quantized geometry descriptors for robust deduplication.")

        self.output.insert(tk.END, "\n".join(lines))

    def _save_report(self) -> None:
        if not hasattr(self, "latest_result"):
            messagebox.showinfo("Nothing to save", "Run a comparison first.")
            return
        result: ComparisonResult = self.latest_result  # type: ignore[attr-defined]
        data = {
            "tolerance": self.tolerance_var.get(),
            "file_a": {
                "path": result.summary_a.path,
                "hash": result.summary_a.hash_hex,
                "descriptor": result.summary_a.descriptor.to_ordered_dict(),
            },
            "file_b": {
                "path": result.summary_b.path,
                "hash": result.summary_b.hash_hex,
                "descriptor": result.summary_b.descriptor.to_ordered_dict(),
            },
            "hash_match": result.summary_a.hash_hex == result.summary_b.hash_hex,
        }
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Saved", f"Report saved to {filename}")


def main() -> int:
    app = ComparatorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
