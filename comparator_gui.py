import json
import sys
from dataclasses import dataclass
from pathlib import Path

from geometry_descriptor import GeometryHasher, GeometrySignature

# -------- Qt imports (PySide6 first, fallback to PySide2) --------
try:
    from PySide6.QtWidgets import (
        QApplication,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QPushButton,
        QLabel,
        QLineEdit,
        QFileDialog,
        QTextEdit,
        QMessageBox,
    )
    from PySide6.QtCore import Qt
except ImportError:
    from PySide2.QtWidgets import (
        QApplication,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QPushButton,
        QLabel,
        QLineEdit,
        QFileDialog,
        QTextEdit,
        QMessageBox,
    )
    from PySide2.QtCore import Qt


# -------- Data model --------

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


# -------- GUI --------

class ComparatorWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STEP Geometry Comparator")
        self.resize(900, 700)

        self.file_a_input: QLineEdit
        self.file_b_input: QLineEdit
        self.tolerance_input: QLineEdit
        self.output: QTextEdit
        self.latest_result: ComparisonResult | None = None

        self._build_widgets()

    def _build_widgets(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel("STEP Geometry Comparator")
        header.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(header)

        description = QLabel(
            "Compare two STEP files using geometry descriptors "
            "(volume, area, inertia, histograms, etc.). "
            "Descriptors are quantized using the tolerance to build "
            "a stable geometry hash for deduplication."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form_layout = QGridLayout()
        layout.addLayout(form_layout)

        # File A
        self.file_a_input = QLineEdit()
        browse_a = QPushButton("Browse")
        browse_a.clicked.connect(lambda: self._choose_file(self.file_a_input))

        # File B
        self.file_b_input = QLineEdit()
        browse_b = QPushButton("Browse")
        browse_b.clicked.connect(lambda: self._choose_file(self.file_b_input))

        # Tolerance
        self.tolerance_input = QLineEdit("0.001")

        form_layout.addWidget(QLabel("File A:"), 0, 0, alignment=Qt.AlignRight)
        form_layout.addWidget(self.file_a_input, 0, 1)
        form_layout.addWidget(browse_a, 0, 2)

        form_layout.addWidget(QLabel("File B:"), 1, 0, alignment=Qt.AlignRight)
        form_layout.addWidget(self.file_b_input, 1, 1)
        form_layout.addWidget(browse_b, 1, 2)

        form_layout.addWidget(QLabel("Tolerance:"), 2, 0, alignment=Qt.AlignRight)
        form_layout.addWidget(self.tolerance_input, 2, 1)

        # Buttons
        button_row = QHBoxLayout()
        compare_button = QPushButton("Compare")
        compare_button.clicked.connect(self._compare)
        save_button = QPushButton("Save report as JSON")
        save_button.clicked.connect(self._save_report)
        button_row.addWidget(compare_button)
        button_row.addWidget(save_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        # Output box
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output, stretch=1)

    # -------- Helpers --------

    def _choose_file(self, widget: QLineEdit) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select STEP file",
            str(Path.home()),
            "STEP files (*.step *.stp);;All files (*.*)",
        )
        if filename:
            widget.setText(filename)

    def _parse_tolerance(self) -> float | None:
        try:
            tol = float(self.tolerance_input.text())
            if tol <= 0:
                raise ValueError
            return tol
        except ValueError:
            QMessageBox.critical(self, "Invalid tolerance", "Please enter a positive numeric tolerance.")
            return None

    def _compare(self) -> None:
        file_a = self.file_a_input.text().strip()
        file_b = self.file_b_input.text().strip()
        tol = self._parse_tolerance()
        if tol is None:
            return
        if not file_a or not file_b:
            QMessageBox.warning(self, "Files missing", "Please select both STEP files before comparing.")
            return

        comparator = StepComparator(tol)
        try:
            result = comparator.compare(file_a, file_b)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "File not found", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Comparison failed", f"Unexpected error: {exc}")
            return

        self.latest_result = result
        self._display_result(result)

    def _display_result(self, result: ComparisonResult) -> None:
        def fmt_descriptor(summary: GeometrySignature) -> str:
            d = summary.descriptor
            inertia = ", ".join(f"{v:.3f}" for v in d.inertia)
            bbox = ", ".join(f"{v:.3f}" for v in d.bbox_dims)
            face_hist = ", ".join(str(v) for v in d.face_hist)
            edge_hist = ", ".join(str(v) for v in d.edge_hist)
            return (
                f"Path: {summary.path}\n"
                f"Geometry hash: {summary.hash_hex}\n"
                f"Volume: {d.volume}\n"
                f"Surface area: {d.area}\n"
                f"BBox dims (sorted): {bbox}\n"
                f"Faces: {d.num_faces}, Edges: {d.num_edges}\n"
                f"Principal inertia: {inertia}\n"
                f"Centroid offset (from bbox center): {d.centroid_offset}\n"
                f"Face area histogram: [{face_hist}]\n"
                f"Edge length histogram: [{edge_hist}]\n"
            )

        lines = [
            "== Geometry summaries ==\n\n",
            "File A:\n",
            fmt_descriptor(result.summary_a),
            "\nFile B:\n",
            fmt_descriptor(result.summary_b),
            "\n",
        ]

        if result.summary_a.hash_hex == result.summary_b.hash_hex:
            lines.append("Geometries match (hashes are identical).\n")
        else:
            lines.append("Geometries differ (hashes are not identical).\n")

        lines.append(
            "\nHashes are derived from quantized geometry descriptors for robust deduplication.\n"
        )

        self.output.setPlainText("".join(lines))

    def _save_report(self) -> None:
        if self.latest_result is None:
            QMessageBox.information(self, "Nothing to save", "Run a comparison first.")
            return

        result = self.latest_result
        d_a = result.summary_a.descriptor
        d_b = result.summary_b.descriptor

        data = {
            "tolerance": self.tolerance_input.text(),
            "file_a": {
                "path": result.summary_a.path,
                "hash": result.summary_a.hash_hex,
                "descriptor": d_a.to_ordered_dict(),
            },
            "file_b": {
                "path": result.summary_b.path,
                "hash": result.summary_b.hash_hex,
                "descriptor": d_b.to_ordered_dict(),
            },
            "hash_match": result.summary_a.hash_hex == result.summary_b.hash_hex,
        }

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON report",
            str(Path.home()),
            "JSON files (*.json);;All files (*.*)",
        )
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "Saved", f"Report saved to {filename}")


def main() -> int:
    app = QApplication(sys.argv)

    window = ComparatorWindow()
    window.show()

    # PySide6 has exec(), PySide2 has exec_()
    if hasattr(app, "exec"):
        return app.exec()
    else:
        return app.exec_()



if __name__ == "__main__":
    sys.exit(main())
