from PyQt5.QtWidgets import QDialog, QLineEdit, QFormLayout, QLabel, QComboBox, QVBoxLayout, QDialogButtonBox, QMessageBox, QApplication
from modules.themes import themes

class MaxwellDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Maxwell Fitting Parameters')
        
        self.tau_H_input = QLineEdit(self)
        self.tau_L_input = QLineEdit(self)
        self.modes_per_decade_input = QLineEdit(self)
        self.tau_H_input.setEnabled(False)
        self.tau_L_input.setEnabled(False)
        self.modes_per_decade_input.setEnabled(False)

        form_layout = QFormLayout()
        
        tau_H_label = QLabel("&#964;<sub>H</sub>:")
        tau_L_label = QLabel("&#964;<sub>L</sub>:")
        modes_label = QLabel("Modes/Decade:")

        form_layout.addRow(tau_H_label, self.tau_H_input)
        form_layout.addRow(tau_L_label, self.tau_L_input)
        form_layout.addRow(modes_label, self.modes_per_decade_input)

        self.preset_dropdown = QComboBox(self)
        self.preset_dropdown.addItem("Univ Default")
        self.preset_dropdown.addItem("Enter Manually")
        self.app = QApplication.instance()
        if self.app.is_dark:
            self.preset_dropdown.setStyleSheet(themes['dropdown_dark'])
        else:
            self.preset_dropdown.setStyleSheet(themes['dropdown_light'])
        self.apply_preset()
        self.preset_dropdown.currentIndexChanged.connect(self.apply_preset)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel("Use pre-sets:"))
        main_layout.addWidget(self.preset_dropdown)
        main_layout.addLayout(form_layout)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        main_layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def apply_preset(self):
        presets = {
            "Univ Default": {"tau_H": "4", "tau_L": "-13", "modes": "8"}}

        preset_name = self.preset_dropdown.currentText()
        if preset_name == "Enter Manually":
            self.tau_H_input.setEnabled(True)
            self.tau_L_input.setEnabled(True)
            self.modes_per_decade_input.setEnabled(True) 
            self.tau_H_input.clear()
            self.tau_L_input.clear()
            self.modes_per_decade_input.clear()
            return
        if preset_name in presets:
            self.tau_H_input.setText(presets[preset_name]["tau_H"])
            self.tau_L_input.setText(presets[preset_name]["tau_L"])
            self.modes_per_decade_input.setText(presets[preset_name]["modes"])
        

    def get_inputs(self):
        try:            
            tau_H = int(self.tau_H_input.text())
            tau_L = int(self.tau_L_input.text())
            modes_per_decade = int(self.modes_per_decade_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid Maxwell mode input")
            return
        return tau_H, tau_L, modes_per_decade
