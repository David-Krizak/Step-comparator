import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
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
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
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
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
    )
    from PySide2.QtCore import Qt


# -------- Simple JSON "DB" for known models --------

DEFAULT_DB_PATH = Path.home() / "step_compare_db.json"


def _build_hash_index(data: dict) -> dict[str, list[dict]]:
    """Builds {hash_hex: [model_record, ...]}."""
    index: dict[str, list[dict]] = {}
    models = data.get("models", [])
    if not isinstance(models, list):
        return index
    for m in models:
        h = m.get("hash")
        if not h:
            continue
        index.setdefault(h, []).append(m)
    return index


def _load_db(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    if not path.is_file():
        data = {"version": 1, "models": []}
        return data, _build_hash_index(data)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {"version": 1, "models": []}
        if "models" not in data or not isinstance(data["models"], list):
            data["models"] = []
        return data, _build_hash_index(data)
    except Exception:
        # Corrupt DB → start fresh
        data = {"version": 1, "models": []}
        return data, _build_hash_index(data)


def _save_db(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _find_by_hash(index: dict[str, list[dict]], hash_hex: str) -> list[dict]:
    return index.get(hash_hex, [])


def _record_model(
    data: dict,
    index: dict[str, list[dict]],
    sig: GeometrySignature,
) -> None:
    """Add model to DB if not already recorded with same hash+path."""
    models = data.setdefault("models", [])
    h = sig.hash_hex
    p = sig.path

    # Quick check via index
    existing_list = index.get(h, [])
    for m in existing_list:
        if m.get("path") == p:
            return

    d = sig.descriptor
    record = {
        "hash": h,
        "name": Path(p).name,
        "path": p,
        "added": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "volume": d.volume,
        "area": d.area,
        "bbox": list(d.bbox_dims),
    }
    models.append(record)
    index.setdefault(h, []).append(record)


# -------- Data model for comparison result --------

@dataclass
class ComparisonResult:
    summary_a: GeometrySignature
    summary_b: GeometrySignature
    exact_match: bool
    metrics: dict[str, float]


class StepComparator:
    def __init__(self, tolerance: float) -> None:
        self.hasher = GeometryHasher(tolerance)

    def compare(self, path_a: str, path_b: str) -> ComparisonResult:
        # 1) Descriptor + hash
        summary_a = self.hasher.signature_for_file(path_a)
        summary_b = self.hasher.signature_for_file(path_b)

        # 2) Strict boolean verification
        exact_match, metrics = self.hasher.verify_exact_match(path_a, path_b)

        return ComparisonResult(summary_a, summary_b, exact_match, metrics)

    def export_differences(self, path_a: str, path_b: str, out_dir: Path) -> dict[str, Path]:
        return self.hasher.export_difference_shapes(path_a, path_b, out_dir)


# -------- GUI --------

class ComparatorWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STEP Geometry Comparator (Hard Mode)")
        self.resize(1100, 820)

        self.file_a_input: QLineEdit
        self.file_b_input: QLineEdit
        self.tolerance_input: QLineEdit

        self.db_path_edit: QLineEdit

        self.tabs: QTabWidget
        self.summary_label: QLabel
        self.known_badge: QLabel
        self.summary_text: QTextEdit
        self.details_text: QTextEdit
        self.details_table: QTableWidget
        self.diff_info_text: QTextEdit

        self.latest_result: ComparisonResult | None = None

        # DB state
        self.db_path: Path = DEFAULT_DB_PATH
        self.db: dict
        self.db_index: dict[str, list[dict]]

        self.db, self.db_index = _load_db(self.db_path)

        self._build_widgets()
        self._update_db_path_display()

    def _build_widgets(self) -> None:
        outer = QVBoxLayout(self)

        # DB selector row
        db_row = QHBoxLayout()
        db_row.addWidget(QLabel("DB JSON:"))
        self.db_path_edit = QLineEdit(str(self.db_path))
        self.db_path_edit.setReadOnly(True)
        db_row.addWidget(self.db_path_edit, stretch=1)
        db_button = QPushButton("Odaberi DB…")
        db_button.clicked.connect(self._choose_db_file)
        db_row.addWidget(db_button)
        outer.addLayout(db_row)

        # Top: file selection + tolerance
        form_layout = QGridLayout()

        self.file_a_input = QLineEdit()
        browse_a = QPushButton("Browse")
        browse_a.clicked.connect(lambda: self._choose_file(self.file_a_input))

        self.file_b_input = QLineEdit()
        browse_b = QPushButton("Browse")
        browse_b.clicked.connect(lambda: self._choose_file(self.file_b_input))

        self.tolerance_input = QLineEdit("0.001")

        form_layout.addWidget(QLabel("File A:"), 0, 0, alignment=Qt.AlignRight)
        form_layout.addWidget(self.file_a_input, 0, 1)
        form_layout.addWidget(browse_a, 0, 2)

        form_layout.addWidget(QLabel("File B:"), 1, 0, alignment=Qt.AlignRight)
        form_layout.addWidget(self.file_b_input, 1, 1)
        form_layout.addWidget(browse_b, 1, 2)

        form_layout.addWidget(QLabel("Tolerance:"), 2, 0, alignment=Qt.AlignRight)
        form_layout.addWidget(self.tolerance_input, 2, 1)

        outer.addLayout(form_layout)

        # Buttons
        button_row = QHBoxLayout()
        compare_button = QPushButton("Compare")
        compare_button.clicked.connect(self._compare)
        single_check_button = QPushButton("Check model vs DB")
        single_check_button.clicked.connect(self._check_single_against_db)
        save_button = QPushButton("Save report as JSON")
        save_button.clicked.connect(self._save_report)
        button_row.addWidget(compare_button)
        button_row.addWidget(single_check_button)
        button_row.addWidget(save_button)
        button_row.addStretch()
        outer.addLayout(button_row)

        # Tabs
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, stretch=1)

        # Tab 1: Sažetak
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)

        self.summary_label = QLabel("Nema rezultata")
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setStyleSheet(
            "font-size: 26pt; font-weight: 800; color: gray;"
            "padding: 12px; border-radius: 12px;"
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #f7f7f7, stop:1 #e1e1e1);"
        )
        summary_layout.addWidget(self.summary_label)

        self.known_badge = QLabel("")
        self.known_badge.setAlignment(Qt.AlignCenter)
        self.known_badge.setStyleSheet(
            "font-size: 12pt; color: #1f3b57;"
            "padding: 6px 10px; border: 1px solid #7ab0d4;"
            "border-radius: 10px; background: #d9ecfa;"
        )
        self.known_badge.hide()
        summary_layout.addWidget(self.known_badge)

        info_label = QLabel(
            "Kratki rezultat: 'ISTI MODELI' ili 'RAZLIČITI MODELI',\n"
            "osnovne metrike i informacija iz interne baze (ranije viđeni modeli)."
        )
        info_label.setWordWrap(True)
        summary_layout.addWidget(info_label)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text, stretch=1)

        self.tabs.addTab(summary_tab, "Sažetak")

        # Tab 2: Detalji
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)

        self.details_table = QTableWidget(0, 3)
        self.details_table.setHorizontalHeaderLabels(["Metrika", "Model A", "Model B"])
        self.details_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.details_table.verticalHeader().setVisible(False)
        self.details_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.details_table.setAlternatingRowColors(True)
        details_layout.addWidget(self.details_table)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("font-family: Consolas, monospace;")
        details_layout.addWidget(self.details_text, stretch=1)

        self.tabs.addTab(details_tab, "Detalji")

        # Tab 3: 3D / razlike
        diff_tab = QWidget()
        diff_layout = QVBoxLayout(diff_tab)

        diff_label = QLabel(
            "Ovdje možeš otvoriti oba modela i eksportirati razlike (A−B, B−A)\n"
            "u STEP datoteke za pregled u FreeCAD-u."
        )
        diff_label.setWordWrap(True)
        diff_layout.addWidget(diff_label)

        diff_button_row = QHBoxLayout()
        open_a_btn = QPushButton("Otvori model A u FreeCAD")
        open_a_btn.clicked.connect(self._open_a_in_freecad)
        diff_button_row.addWidget(open_a_btn)

        open_b_btn = QPushButton("Otvori model B u FreeCAD")
        open_b_btn.clicked.connect(self._open_b_in_freecad)
        diff_button_row.addWidget(open_b_btn)

        export_diff_btn = QPushButton("Eksportiraj razlike (A−B, B−A)")
        export_diff_btn.clicked.connect(self._export_diffs)
        diff_button_row.addWidget(export_diff_btn)

        diff_button_row.addStretch()
        diff_layout.addLayout(diff_button_row)

        self.diff_info_text = QTextEdit()
        self.diff_info_text.setReadOnly(True)
        diff_layout.addWidget(self.diff_info_text, stretch=1)

        self.tabs.addTab(diff_tab, "3D / razlike")

    # -------- DB path handling --------

    def _update_db_path_display(self) -> None:
        self.db_path_edit.setText(str(self.db_path))

    def _choose_db_file(self) -> None:
        """
        Let user select or create a DB JSON.

        - If file exists → load it.
        - If not → create new empty DB in memory and use that path.
        """
        initial = str(self.db_path if self.db_path else DEFAULT_DB_PATH)
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Odaberi ili kreiraj DB JSON datoteku",
            initial,
            "JSON files (*.json);;All files (*.*)",
        )
        if not filename:
            return

        new_path = Path(filename)
        self.db_path = new_path
        self.db, self.db_index = _load_db(self.db_path)
        self._update_db_path_display()

        QMessageBox.information(
            self,
            "DB postavljena",
            f"Korištenje DB:\n{self.db_path}\n"
            f"(trenutno zabilježenih modela: {len(self.db.get('models', []))})",
        )

    # -------- Helpers --------

    def _get_existing_matches(self, sig: GeometrySignature) -> list[dict]:
        """Copy of matches before we possibly add the current signature."""
        return [dict(m) for m in _find_by_hash(self.db_index, sig.hash_hex)]

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
            QMessageBox.critical(self, "Comparison failed", f"Unexpected error:\n{exc}")
            return

        self.latest_result = result

        # Snapshot known matches before we add the current models to the DB index
        existing_a = self._get_existing_matches(result.summary_a)
        existing_b = self._get_existing_matches(result.summary_b)

        # Update DB with A and B
        self._update_db_with_signatures(result.summary_a, result.summary_b)

        # Refresh UI
        self._update_ui_with_result(result, existing_a, existing_b)

    def _check_single_against_db(self) -> None:
        tol = self._parse_tolerance()
        if tol is None:
            return

        file_a = self.file_a_input.text().strip()
        file_b = self.file_b_input.text().strip()
        if not file_a and not file_b:
            QMessageBox.warning(self, "No file", "Select at least one STEP file to check against the DB.")
            return

        target_label, target_path = ("A", file_a) if file_a else ("B", file_b)

        hasher = GeometryHasher(tol)
        try:
            sig = hasher.signature_for_file(target_path)
        except FileNotFoundError:
            QMessageBox.critical(self, "File not found", f"File does not exist:\n{target_path}")
            return
        except Exception as exc:
            QMessageBox.critical(self, "Hashing failed", f"Unexpected error while reading file:\n{exc}")
            return

        matches_before = self._get_existing_matches(sig)
        _record_model(self.db, self.db_index, sig)
        try:
            _save_db(self.db_path, self.db)
        except Exception as exc:
            QMessageBox.warning(self, "DB warning", f"Could not save DB:\n{exc}")

        if matches_before:
            text = self._build_known_models_text(f"Model {target_label}", matches_before)
        else:
            text = (
                f"Model {target_label}: geometrija nije ranije zabilježena u ovoj bazi.\n"
                f"Dodano je novo pojavljivanje ({Path(target_path).name})."
            )

        QMessageBox.information(self, "Provjera u bazi", text)

    def _update_db_with_signatures(self, sig_a: GeometrySignature, sig_b: GeometrySignature) -> None:
        _record_model(self.db, self.db_index, sig_a)
        _record_model(self.db, self.db_index, sig_b)
        try:
            _save_db(self.db_path, self.db)
        except Exception as exc:
            QMessageBox.warning(self, "DB warning", f"Could not save DB:\n{exc}")

    def _build_known_models_text(self, label: str, matches: list[dict]) -> str:
        if not matches:
            return f"{label}: geometrija nije ranije zabilježena u ovoj bazi.\n"
        lines = [f"{label}: geometrija već postoji u ovoj bazi, raniji modeli:\n"]
        for m in matches:
            lines.append(
                f"  - {m.get('name')}  (putanja: {m.get('path')}, dodano: {m.get('added')})\n"
            )
        return "".join(lines)

    def _update_ui_with_result(
        self,
        result: ComparisonResult,
        known_matches_a: list[dict],
        known_matches_b: list[dict],
    ) -> None:
        # ----- 1) Summary label: ISTI / RAZLIČITI MODELI -----
        if result.exact_match:
            self.summary_label.setText("ISTI MODELI")
            self.summary_label.setStyleSheet(
                "font-size: 26pt; font-weight: 800; color: white;"
                "padding: 12px; border-radius: 12px;"
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                " stop:0 #3ba55d, stop:1 #2f8c4d);"
            )
        else:
            self.summary_label.setText("RAZLIČITI MODELI")
            self.summary_label.setStyleSheet(
                "font-size: 26pt; font-weight: 800; color: white;"
                "padding: 12px; border-radius: 12px;"
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                " stop:0 #d64b4b, stop:1 #b23333);"
            )

        a = result.summary_a
        b = result.summary_b
        m = result.metrics

        same_hash = (a.hash_hex == b.hash_hex)

        # Short summary text (Sažetak tab)
        summary_lines: list[str] = []
        summary_lines.append(f"DB: {self.db_path}\n\n")
        summary_lines.append(f"File A: {a.path}\nFile B: {b.path}\n\n")
        summary_lines.append(f"Hash A: {a.hash_hex}\nHash B: {b.hash_hex}\n")
        summary_lines.append(f"Hash isti: {'DA' if same_hash else 'NE'}\n\n")

        known_a = self._build_known_models_text("Model A", known_matches_a)
        known_b = self._build_known_models_text("Model B", known_matches_b)
        summary_lines.append(known_a)
        summary_lines.append("\n")
        summary_lines.append(known_b)
        summary_lines.append("\n")

        summary_lines.append("Boolean metrike:\n")
        summary_lines.append(f"  Volumen A:       {m.get('vol_a')}\n")
        summary_lines.append(f"  Volumen B:       {m.get('vol_b')}\n")
        summary_lines.append(f"  Volumen (A − B): {m.get('vol_ab')}\n")
        summary_lines.append(f"  Volumen (B − A): {m.get('vol_ba')}\n\n")
        summary_lines.append(
            "Konačni zaključak:\n"
            f"  {'ISTI MODELI' if result.exact_match else 'RAZLIČITI MODELI'}\n"
        )

        self.summary_text.setPlainText("".join(summary_lines))

        # ----- 2) Detailed report (Detalji tab) -----
        self._populate_details_table(result)
        self.details_text.setPlainText(self._build_details_report(result))

        # ----- 3) Diff tab info -----
        self.diff_info_text.setPlainText(
            "Klikni na 'Eksportiraj razlike (A−B, B−A)' za generiranje STEP datoteka "
            "koje možeš otvoriti u FreeCAD-u za vizualnu provjeru.\n\n"
            "Ako boolean operacije ne uspiju, aplikacija će ići na sigurnu stranu i "
            "prijavit će 'RAZLIČITI MODELI'."
        )

        # Focus summary tab so user immediately sees ISTI/RAZLIČITI
        self.tabs.setCurrentIndex(0)

        self._update_known_badge(result, known_matches_a, known_matches_b)

    def _update_known_badge(
        self,
        result: ComparisonResult,
        known_matches_a: list[dict],
        known_matches_b: list[dict],
    ) -> None:
        has_known_a = bool(known_matches_a)
        has_known_b = bool(known_matches_b)
        same_hash = result.summary_a.hash_hex == result.summary_b.hash_hex

        if has_known_a or has_known_b:
            if same_hash:
                text = "Modeli su poznati i imaju identičnu geometriju prema bazi."
            elif has_known_a and has_known_b:
                text = "Oba modela su ranije viđena (različite geometrije)."
            elif has_known_a:
                text = "Model A je ranije zabilježen u bazi."
            else:
                text = "Model B je ranije zabilježen u bazi."
            self.known_badge.setText(text)
            self.known_badge.show()
        else:
            self.known_badge.hide()

    def _populate_details_table(self, result: ComparisonResult) -> None:
        a = result.summary_a.descriptor
        b = result.summary_b.descriptor

        def fmt_list(values) -> str:
            return ", ".join(f"{v:.6g}" for v in values)

        rows = [
            ("Volumen", f"{a.volume:.6g}", f"{b.volume:.6g}"),
            ("Površina", f"{a.area:.6g}", f"{b.area:.6g}"),
            ("BBox dimenzije", fmt_list(a.bbox_dims), fmt_list(b.bbox_dims)),
            ("Broj lica", str(a.num_faces), str(b.num_faces)),
            ("Broj bridova", str(a.num_edges), str(b.num_edges)),
            ("Centroid offset", f"{a.centroid_offset:.6g}", f"{b.centroid_offset:.6g}"),
            ("Inercija eigenvalues", fmt_list(a.inertia), fmt_list(b.inertia)),
            ("Face hash", a.mesh_hash, b.mesh_hash),
        ]

        self.details_table.setRowCount(len(rows))
        for r, (metric, val_a, val_b) in enumerate(rows):
            for c, value in enumerate((metric, val_a, val_b)):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.details_table.setItem(r, c, item)

        self.details_table.resizeRowsToContents()

    def _build_details_report(self, result: ComparisonResult) -> str:
        a = result.summary_a
        b = result.summary_b

        def fmt_descriptor(label: str, sig: GeometrySignature) -> str:
            d = sig.descriptor

            inertia = ", ".join(f"{v:.6g}" for v in d.inertia)
            bbox = ", ".join(f"{v:.6g}" for v in d.bbox_dims)
            face_hist = ", ".join(str(v) for v in d.face_hist)
            edge_hist = ", ".join(str(v) for v in d.edge_hist)

            ft_labels = ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "other"]
            et_labels = ["line", "circle", "ellipse", "bspline", "other"]
            ft_counts = ", ".join(f"{name}={cnt}" for name, cnt in zip(ft_labels, d.face_type_counts))
            et_counts = ", ".join(f"{name}={cnt}" for name, cnt in zip(et_labels, d.edge_type_counts))

            tri_hist = ", ".join(str(v) for v in d.tri_area_hist)
            mesh_edge_hist = ", ".join(str(v) for v in d.mesh_edge_hist)

            return (
                f"{label} path: {sig.path}\n"
                f"{label} hash: {sig.hash_hex}\n"
                f"{label} volume: {d.volume}\n"
                f"{label} area:   {d.area}\n"
                f"{label} bbox (sorted): [{bbox}]\n"
                f"{label} faces: {d.num_faces}, edges: {d.num_edges}\n"
                f"{label} inertia eigenvalues: [{inertia}]\n"
                f"{label} centroid offset: {d.centroid_offset}\n"
                f"{label} face area hist: [{face_hist}]\n"
                f"{label} edge length hist: [{edge_hist}]\n"
                f"{label} face type counts: {ft_counts}\n"
                f"{label} edge type counts: {et_counts}\n"
                f"{label} tri area hist (mesh): [{tri_hist}]\n"
                f"{label} mesh edge hist: [{mesh_edge_hist}]\n"
                f"{label} mesh hash: {d.mesh_hash}\n"
            )

        lines: list[str] = []
        lines.append("== Geometry summaries ==\n\n")
        lines.append(fmt_descriptor("A", a))
        lines.append("\n")
        lines.append(fmt_descriptor("B", b))
        lines.append("\n")

        same_hash = (a.hash_hex == b.hash_hex)
        m = result.metrics

        lines.append("== Match decision ==\n")
        lines.append(f"Descriptor hashes identical: {'YES' if same_hash else 'NO'}\n")
        if result.exact_match:
            lines.append("Boolean A−B / B−A: NO remaining volume → exact.\n")
            lines.append("FINAL: Geometries are considered EXACT MATCH.\n")
        else:
            lines.append("Boolean A−B / B−A: difference detected or boolean failed.\n")
            lines.append("FINAL: Geometries are considered DIFFERENT.\n")

        lines.append("\n== Boolean metrics ==\n")
        lines.append(f"Volume A: {m.get('vol_a')}\n")
        lines.append(f"Volume B: {m.get('vol_b')}\n")
        lines.append(f"Area   A: {m.get('area_a')}\n")
        lines.append(f"Area   B: {m.get('area_b')}\n")
        lines.append(f"Volume(A − B): {m.get('vol_ab')}\n")
        lines.append(f"Volume(B − A): {m.get('vol_ba')}\n")

        lines.append(
            "\nDescriptor + mesh hash + boolean difference make false positives "
            "extremely unlikely. If boolean fails, tool errs on the SAFE side "
            "(reports DIFFERENT).\n"
        )
        return "".join(lines)

    def _save_report(self) -> None:
        if self.latest_result is None:
            QMessageBox.information(self, "Nothing to save", "Run a comparison first.")
            return

        result = self.latest_result
        da = result.summary_a.descriptor
        db = result.summary_b.descriptor

        data = {
            "tolerance": self.tolerance_input.text(),
            "db_path": str(self.db_path),
            "file_a": {
                "path": result.summary_a.path,
                "hash": result.summary_a.hash_hex,
                "descriptor": da.to_ordered_dict(),
            },
            "file_b": {
                "path": result.summary_b.path,
                "hash": result.summary_b.hash_hex,
                "descriptor": db.to_ordered_dict(),
            },
            "hash_match": result.summary_a.hash_hex == result.summary_b.hash_hex,
            "exact_match": result.exact_match,
            "boolean_metrics": result.metrics,
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

    # -------- 3D / diff tab actions --------

    def _get_freecad_exe(self) -> Path | None:
        exe = Path(sys.executable).resolve()
        root = exe.parent.parent  # bin -> root
        candidate = root / "bin" / "FreeCAD.exe"
        if candidate.is_file():
            return candidate
        return None

    def _open_in_freecad(self, path: str) -> None:
        freecad_exe = self._get_freecad_exe()
        if freecad_exe is None:
            QMessageBox.critical(self, "FreeCAD not found", "FreeCAD.exe not found next to python.exe.")
            return
        if not Path(path).is_file():
            QMessageBox.critical(self, "File not found", f"File does not exist:\n{path}")
            return
        try:
            subprocess.Popen([str(freecad_exe), path])
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to launch FreeCAD:\n{exc}")

    def _open_a_in_freecad(self) -> None:
        path = self.file_a_input.text().strip()
        if not path:
            QMessageBox.information(self, "No file", "Select File A first.")
            return
        self._open_in_freecad(path)

    def _open_b_in_freecad(self) -> None:
        path = self.file_b_input.text().strip()
        if not path:
            QMessageBox.information(self, "No file", "Select File B first.")
            return
        self._open_in_freecad(path)

    def _export_diffs(self) -> None:
        if self.latest_result is None:
            QMessageBox.information(self, "No comparison", "Run a comparison first.")
            return

        file_a = self.file_a_input.text().strip()
        file_b = self.file_b_input.text().strip()
        if not file_a or not file_b:
            QMessageBox.information(self, "Missing files", "Select both files first.")
            return

        # Ask where to put diff files (default: user's home / step_compare_diffs)
        default_dir = Path.home() / "step_compare_diffs"
        target_dir_str = QFileDialog.getExistingDirectory(
            self,
            "Select folder for difference STEP files",
            str(default_dir),
        )
        if not target_dir_str:
            return

        target_dir = Path(target_dir_str)
        comparator = StepComparator(float(self.tolerance_input.text() or "0.001"))
        try:
            files = comparator.export_differences(file_a, file_b, target_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Failed to export difference shapes:\n{exc}")
            return

        info_lines = [
            "Eksportirane STEP datoteke:\n",
            f"  A_clean:    {files['A_clean']}\n",
            f"  B_clean:    {files['B_clean']}\n",
            f"  A_minus_B:  {files['A_minus_B']}\n",
            f"  B_minus_A:  {files['B_minus_A']}\n\n",
            "Otvori ih u FreeCAD-u za vizualnu provjeru razlika.\n",
        ]
        self.diff_info_text.setPlainText("".join(info_lines))

        # Open folder in explorer for convenience
        try:
            os.startfile(str(target_dir))
        except Exception:
            pass


def main() -> int:
    app = QApplication(sys.argv)
    win = ComparatorWindow()
    win.show()

    # PySide6 has exec(), PySide2 has exec_()
    if hasattr(app, "exec"):
        return app.exec()
    else:
        return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
