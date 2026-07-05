"""
app.py
PyQt5 desktop GUI for BERT-based Twitter sentiment analysis.
Loads a dataset CSV, loads a saved BERT model/tokenizer, lets the user click
a tweet (or type one) and shows the predicted sentiment + confidence.

Run:
    python app.py
"""

import json
import os
import sys

import pandas as pd
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from transformers import BertForSequenceClassification, BertTokenizerFast


class SentimentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Twitter Sentiment Analysis - BERT PyQt GUI")
        self.resize(1050, 620)

        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = None
        self.current_selected_text = None

        self._build_ui()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        header = QLabel("Twitter Sentiment Analysis - BERT PyQt GUI")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        main_layout.addWidget(header)

        subheader = QLabel("Load dataset, load trained model, click a tweet, and view predicted sentiment.")
        subheader.setStyleSheet("color: gray;")
        main_layout.addWidget(subheader)

        # Top bar
        top_bar = QHBoxLayout()
        self.btn_load_dataset = QPushButton("Load Dataset CSV")
        self.btn_load_model = QPushButton("Load BERT Model")
        self.btn_predict_selected = QPushButton("Predict Selected")
        self.status_label = QLabel("Model Status: not loaded  |  Dataset: not loaded")

        for btn in (self.btn_load_dataset, self.btn_load_model, self.btn_predict_selected):
            btn.setMinimumHeight(34)

        top_bar.addWidget(self.btn_load_dataset)
        top_bar.addWidget(self.btn_load_model)
        top_bar.addWidget(self.btn_predict_selected)
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # Middle: table + prediction panel
        middle_layout = QHBoxLayout()

        table_box = QVBoxLayout()
        table_box.addWidget(self._section_title("Loaded Tweets"))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Tweet Text", "Actual Label"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.cellClicked.connect(self.on_row_clicked)
        table_box.addWidget(self.table)
        middle_layout.addLayout(table_box, 3)

        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel_layout = QVBoxLayout(panel)

        panel_layout.addWidget(self._section_title("Prediction Panel"))

        panel_layout.addWidget(self._section_title("Selected Sentence:"))
        self.selected_text_label = QLabel("(click a tweet from the table)")
        self.selected_text_label.setWordWrap(True)
        panel_layout.addWidget(self.selected_text_label)

        panel_layout.addWidget(self._section_title("Predicted Sentiment:"))
        self.prediction_label = QLabel("-")
        pred_font = QFont()
        pred_font.setPointSize(18)
        pred_font.setBold(True)
        self.prediction_label.setFont(pred_font)
        panel_layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("Confidence: -")
        panel_layout.addWidget(self.confidence_label)

        panel_layout.addSpacing(15)
        panel_layout.addWidget(self._section_title("Manual Input"))
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("Type a sentence to analyze...")
        panel_layout.addWidget(self.manual_input)

        self.btn_predict_manual = QPushButton("Predict")
        self.btn_predict_manual.setMinimumHeight(32)
        panel_layout.addWidget(self.btn_predict_manual)

        panel_layout.addStretch()
        middle_layout.addWidget(panel, 2)

        main_layout.addLayout(middle_layout)

        note = QLabel("Note: load the saved_bert_model folder before predicting.")
        note.setStyleSheet("color: gray; font-size: 11px;")
        main_layout.addWidget(note)

        # Connect signals
        self.btn_load_dataset.clicked.connect(self.load_dataset)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_predict_selected.clicked.connect(self.predict_selected_row)
        self.btn_predict_manual.clicked.connect(self.predict_manual)

    @staticmethod
    def _section_title(text):
        label = QLabel(text)
        font = QFont()
        font.setBold(True)
        label.setFont(font)
        return label

    def update_status(self):
        model_status = "loaded" if self.model is not None else "not loaded"
        dataset_status = "loaded" if self.df is not None else "not loaded"
        self.status_label.setText(f"Model Status: {model_status}  |  Dataset: {dataset_status}")

    # ---------------------------------------------------------- Dataset --
    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Twitter Dataset CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{e}")
            return

        text_col = self._guess_column(df, ["text", "tweet", "sentence"])
        label_col = self._guess_column(df, ["sentiment", "label", "class"])

        if text_col is None:
            QMessageBox.critical(self, "Error", "Could not find a text/tweet column in this CSV.")
            return

        cols = [text_col] + ([label_col] if label_col else [])
        df = df[cols].dropna(subset=[text_col])
        df.columns = ["text"] + (["label"] if label_col else [])
        self.df = df.reset_index(drop=True)

        self.table.setRowCount(0)
        self.table.setRowCount(len(self.df))
        for row_idx, row in self.df.iterrows():
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(row["text"])))
            label_val = str(row["label"]) if "label" in self.df.columns else ""
            self.table.setItem(row_idx, 1, QTableWidgetItem(label_val))

        self.update_status()

    @staticmethod
    def _guess_column(df, candidates):
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        return None

    # ------------------------------------------------------------ Model --
    def load_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Select saved_bert_model folder")
        if not folder:
            return
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(folder)
            self.model = BertForSequenceClassification.from_pretrained(folder)
            self.model.to(self.device)
            self.model.eval()

            label_map_path = os.path.join(folder, "label_map.json")
            if os.path.exists(label_map_path):
                with open(label_map_path) as f:
                    mapping = json.load(f)
                self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
            else:
                num_labels = self.model.config.num_labels
                self.id2label = {i: f"class_{i}" for i in range(num_labels)}
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
            return

        self.update_status()
        QMessageBox.information(self, "Model Loaded", f"Model loaded from:\n{folder}")

    # -------------------------------------------------------- Prediction --
    def _predict_text(self, text):
        if self.model is None or self.tokenizer is None:
            QMessageBox.warning(self, "No Model", "Please load a trained BERT model first.")
            return None, None

        inputs = self.tokenizer(
            text, truncation=True, padding=True, max_length=128, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item())

        label = self.id2label.get(pred_id, str(pred_id))
        return label, confidence

    def on_row_clicked(self, row, _col):
        item = self.table.item(row, 0)
        if not item:
            return
        self.current_selected_text = item.text()
        self.selected_text_label.setText(self.current_selected_text)
        # Auto-predict as soon as a tweet is clicked (per assignment spec)
        if self.model is not None:
            self._run_prediction(self.current_selected_text)

    def predict_selected_row(self):
        if not self.current_selected_text:
            QMessageBox.warning(self, "No Selection", "Please click a tweet from the table first.")
            return
        self._run_prediction(self.current_selected_text)

    def predict_manual(self):
        text = self.manual_input.text().strip()
        if not text:
            QMessageBox.warning(self, "Empty Input", "Please type a sentence.")
            return
        self.selected_text_label.setText(text)
        self._run_prediction(text)

    def _run_prediction(self, text):
        label, confidence = self._predict_text(text)
        if label is None:
            return
        self.prediction_label.setText(label.upper())
        self.confidence_label.setText(f"Confidence: {confidence * 100:.1f}%")

        color = {
            "positive": "#2e7d32",
            "negative": "#c62828",
            "neutral": "#616161",
        }.get(label.lower(), "#37474f")
        self.prediction_label.setStyleSheet(f"color: {color};")


def main():
    app = QApplication(sys.argv)
    window = SentimentApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
