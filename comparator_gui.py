import hashlib
import json
import re
import sys
import tkinter as tk
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext
from typing import Iterable, List

FLOAT_PATTERN = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+")


@dataclass
class StepSummary:
    path: str
    tolerance: float
    float_count: int
    min_value: float | None
    max_value: float | None
    hash_hex: str


@dataclass
class ComparisonResult:
    summary_a: StepSummary
    summary_b: StepSummary
    unmatched_from_a: List[float]
    unmatched_from_b: List[float]


class StepComparator:
    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance

    def _extract_numbers(self, content: str) -> List[float]:
        numbers = [float(match) for match in FLOAT_PATTERN.findall(content)]
        return numbers

    def _normalized_tokens(self, numbers: Iterable[float]) -> List[str]:
        # Normalize numbers by rounding to tolerance and using scientific notation for stability
        normalized = []
        tol = self.tolerance if self.tolerance > 0 else 1e-6
        for value in numbers:
            scaled = round(value / tol)
            normalized.append(f"{scaled}")
        return normalized

    def _hash_tokens(self, tokens: Iterable[str]) -> str:
        digest = hashlib.sha256()
        for token in tokens:
            digest.update(token.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _summarize(self, path: str, numbers: List[float]) -> StepSummary:
        tokens = self._normalized_tokens(numbers)
        digest = self._hash_tokens(tokens)
        min_value = min(numbers) if numbers else None
        max_value = max(numbers) if numbers else None
        return StepSummary(
            path=path,
            tolerance=self.tolerance,
            float_count=len(numbers),
            min_value=min_value,
            max_value=max_value,
            hash_hex=digest,
        )

    def compare(self, path_a: str, path_b: str) -> ComparisonResult:
        content_a = Path(path_a).read_text(errors="ignore")
        content_b = Path(path_b).read_text(errors="ignore")

        numbers_a = self._extract_numbers(content_a)
        numbers_b = self._extract_numbers(content_b)

        tokens_a = self._normalized_tokens(numbers_a)
        tokens_b = self._normalized_tokens(numbers_b)

        counter_a = Counter(tokens_a)
        counter_b = Counter(tokens_b)

        diff_a = []
        diff_b = []

        for token, count in (counter_a - counter_b).items():
            diff_a.extend([float(token) * self.tolerance for _ in range(count)])
        for token, count in (counter_b - counter_a).items():
            diff_b.extend([float(token) * self.tolerance for _ in range(count)])

        summary_a = self._summarize(path_a, numbers_a)
        summary_b = self._summarize(path_b, numbers_b)

        return ComparisonResult(summary_a, summary_b, diff_a, diff_b)


class ComparatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("STEP Comparator")
        self.geometry("820x700")
        self._build_widgets()

    def _build_widgets(self) -> None:
        header = tk.Label(self, text="STEP Comparator", font=("Segoe UI", 18, "bold"))
        header.pack(pady=(12, 6))

        description = tk.Label(
            self,
            text=(
                "Compare two STEP files by their numeric contents. "
                "Values are normalized using a tolerance to generate comparable hashes."
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

        def fmt_summary(summary: StepSummary) -> str:
            return (
                f"Path: {summary.path}\n"
                f"Tolerance: {summary.tolerance}\n"
                f"Numbers found: {summary.float_count}\n"
                f"Min value: {summary.min_value}\n"
                f"Max value: {summary.max_value}\n"
                f"Hash: {summary.hash_hex}\n"
            )

        lines = ["== Summary ==\n", "File A:\n", fmt_summary(result.summary_a), "File B:\n", fmt_summary(result.summary_b)]

        if result.unmatched_from_a or result.unmatched_from_b:
            lines.append("Differences detected (normalized by tolerance):\n")
            if result.unmatched_from_a:
                sample = ", ".join(f"{v:.6f}" for v in result.unmatched_from_a[:10])
                lines.append(f"Values only in A (first 10): {sample}\n")
            if result.unmatched_from_b:
                sample = ", ".join(f"{v:.6f}" for v in result.unmatched_from_b[:10])
                lines.append(f"Values only in B (first 10): {sample}\n")
        else:
            lines.append("No numeric differences detected within tolerance.\n")

        lines.append(
            "Hashes are generated from normalized numeric tokens and can be stored for future comparisons."
        )

        self.output.insert(tk.END, "".join(lines))

    def _save_report(self) -> None:
        if not hasattr(self, "latest_result"):
            messagebox.showinfo("Nothing to save", "Run a comparison first.")
            return
        result: ComparisonResult = self.latest_result  # type: ignore[attr-defined]
        data = {
            "tolerance": result.summary_a.tolerance,
            "file_a": result.summary_a.__dict__,
            "file_b": result.summary_b.__dict__,
            "unmatched_from_a": result.unmatched_from_a,
            "unmatched_from_b": result.unmatched_from_b,
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
