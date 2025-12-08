from PyQt5.QtWidgets import QDialog, QVBoxLayout, QComboBox, QLabel

class StatsWindow(QDialog):
    def __init__(self, title, stats_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 350, 180)
        self.stats_dict = stats_dict  # {label: [stat1, stat2, stat3]}
        self.labels = [QLabel() for _ in range(3)]
        layout = QVBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems(list(stats_dict.keys()))
        self.combo.currentTextChanged.connect(self.update_stats)
        layout.addWidget(self.combo)
        for label in self.labels:
            layout.addWidget(label)
        self.setLayout(layout)
        self.update_stats(self.combo.currentText())

    def update_stats(self, selected_label):
        stats = self.stats_dict.get(selected_label, ["N/A", "N/A", "N/A"])
        # If the stats look like custom strings (for binary), display as-is
        if all(isinstance(s, str) and (":" in s or "%" in s) for s in stats):
            for i in range(3):
                self.labels[i].setText(stats[i])
        else:
            self.labels[0].setText(f"Estimated Mn: {stats[0]}")
            self.labels[1].setText(f"Estimated Mw: {stats[1]}")
            self.labels[2].setText(f"Estimated PDI: {stats[2]}")