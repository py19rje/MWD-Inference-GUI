from PyQt5.QtWidgets import QDialog, QVBoxLayout

class StatsWindow(QDialog):
    def __init__(self, title, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 300, 150)

        layout = QVBoxLayout()
        self.labels = labels  

        for label in labels:
            layout.addWidget(label)

        self.setLayout(layout)