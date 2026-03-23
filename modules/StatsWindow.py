#  MWD-Inference-GUI
#  --------------------------------------------------------------------------------------------------------

#  Authors:
#      Robert J. Elliott, py19rje@leeds.ac.uk
#      Daniel J. Read, d.j.read@leeds.ac.uk
#      Luisa Cutillo, l.cutillo@leeds.ac.uk
#      Johan Mattsson, k.j.l.mattsson@leeds.ac.uk

#  GitHub:
#      https://github.com/py19rje/MWD-Inference-GUI

#  --------------------------------------------------------------------------------------------------------

#  Copyright (2026): Robert J. Elliott, University of Leeds

#  This file is part of the software MWD-Inference-GUI

#  MWD-Inference-GUI is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  MWD-Inference-GUI is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this software.  If not, see <http://www.gnu.org/licenses/>.

#  This work forms part of the research programme of DPI, project \#861.

#  --------------------------------------------------------------------------------------------------------

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