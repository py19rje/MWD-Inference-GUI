import json
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QComboBox, QLineEdit, QLabel, QFormLayout, QDialogButtonBox, QCheckBox, QMessageBox, QInputDialog, QPushButton)
from PyQt5.QtCore import Qt
import sys
from modules.themes import themes
from modules.params import load_parameters, params_file_path
from modules.ParamShiftDialog import ParamShiftDialog

try:
    loaded_params = load_parameters(params_file_path)
    Reference_params = loaded_params["Reference_params"]
    Arrhenius_params = loaded_params["Arrhenius_params"]
    WLF_params = loaded_params["WLF_params"]
except:
    print(f"Error loading {params_file_path}, please ensure it is formatted correctly.")
    sys.exit()

class UnivDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Universal Space Shift')
        self.loaded_params = loaded_params
        layout = QVBoxLayout(self)

        self.material_dropdown = QComboBox(self)
        self.material_dropdown.addItems(Reference_params.keys())
        self.material_dropdown.currentTextChanged.connect(self.search_parameters)
        
        self.app = QApplication.instance()
        if self.app.is_dark:
            self.material_dropdown.setStyleSheet(themes['dropdown_dark'])
        else:
            self.material_dropdown.setStyleSheet(themes['dropdown_light'])
        layout.addWidget(QLabel("Select Material:"))
        layout.addWidget(self.material_dropdown)
        
        self.add_new_material_button = QPushButton("Add New Material", self)
        self.add_new_material_button.clicked.connect(self.add_new_material)
        layout.addWidget(self.add_new_material_button)
        
        self.temperature_input = QLineEdit(self)
        self.temperature_input.setPlaceholderText("Enter rheology temperature in °C")
        self.temperature_input.textChanged.connect(self.search_parameters)
        layout.addWidget(QLabel("Temperature (°C):"))
        layout.addWidget(self.temperature_input)
        
        layout.addWidget(QLabel("Params will autofill if found in database"))
        
        self.manual_checkbox = QCheckBox("Enter parameters manually")
        self.manual_checkbox.stateChanged.connect(self.toggle_manual_entry)
        layout.addWidget(self.manual_checkbox)

        self.manual_input_layout = QFormLayout()

        self.me_input = QLineEdit(self)
        self.gn0_input = QLineEdit(self)
        self.tau_e_input = QLineEdit(self)

        self.me_label = QLabel(r"<b>M<sub>e</sub></b>:")
        self.gn0_label = QLabel(r"<b>G<sub>N</sub><sup>0</sup></b>:")
        self.tau_e_label = QLabel(r"<b>&#964;<sub>e</sub></b>:")

        for label in [self.me_label, self.gn0_label, self.tau_e_label]:
            label.setTextFormat(Qt.RichText)

        self.manual_input_layout.addRow(self.me_label, self.me_input)
        self.manual_input_layout.addRow(self.gn0_label, self.gn0_input)
        self.manual_input_layout.addRow(self.tau_e_label, self.tau_e_input)

        layout.addLayout(self.manual_input_layout)
        
        self.save_params_checkbox = QCheckBox("Save manual rheology parameters")
        layout.addWidget(self.save_params_checkbox)
        self.save_params_checkbox.setVisible(False)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(self.buttons)

        self.help_button = self.buttons.addButton("Help", QDialogButtonBox.ActionRole)
        self.help_button.clicked.connect(self.show_help)
        
        self.buttons.accepted.connect(self.handle_ok)
        self.buttons.rejected.connect(self.reject)
        
        self.toggle_manual_entry()
        
    def add_new_material(self):
        dialog = QDialog()
        dialog.setWindowTitle("Add New Material")   
        self.new_material_input = QLineEdit(self)
        self.ref_temperature_input = QLineEdit(self)
        self.new_me_input = QLineEdit(self)
        self.new_gn0_input = QLineEdit(self)
        self.new_tau_e_input = QLineEdit(self)
        
        self.new_material_label = QLabel("New Material Name:")
        self.ref_temperature_label = QLabel("Reference Temperature (°C): (Integer)")
        self.new_me_label = QLabel(r"<b>M<sub>e</sub></b>:")
        self.new_gn0_label = QLabel(r"<b>G<sub>N</sub><sup>0</sup></b>:")
        self.new_tau_e_label = QLabel(r"<b>&#964;<sub>e</sub></b>:")
        
        layout = QVBoxLayout()
        layout.addWidget(self.new_material_label)
        layout.addWidget(self.new_material_input)
        layout.addWidget(self.ref_temperature_label)
        layout.addWidget(self.ref_temperature_input)
        layout.addWidget(self.new_me_label)
        layout.addWidget(self.new_me_input)
        layout.addWidget(self.new_gn0_label)
        layout.addWidget(self.new_gn0_input)
        layout.addWidget(self.new_tau_e_label)
        layout.addWidget(self.new_tau_e_input)
        
        layout.addWidget(QLabel("Parameters entered here will be added to the materials database for future use."))
    
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            new_material = self.new_material_input.text()
            ref_temperature = self.ref_temperature_input.text()
            try: 
                ref_temperature = str(int(ref_temperature))
            except: 
                QMessageBox.warning(self, "Error", "Reference temperature must be an integer.")
                return
            if float(ref_temperature) < 0:
                QMessageBox.warning(self, "Error", "Reference temperature must be a positive integer.")
                return
            
            try:
                new_me = float(self.new_me_input.text())
                new_gn0 = float(self.new_gn0_input.text())
                new_tau_e = float(self.new_tau_e_input.text())
                
                if new_material and ref_temperature:
                    params = {
                        "M_e": new_me,
                        "G0": new_gn0,
                        "tau_e": new_tau_e,
                    }
                    
                    loaded_params["Reference_params"][new_material] = {ref_temperature: params}
                    self.loaded_params = loaded_params
                    self.save_parameters(loaded_params)
                    QMessageBox.information(self, "Success", f"Material '{new_material}' added successfully.")
                else:
                    QMessageBox.warning(self, "Error", "Please enter a valid material name and reference temperature.")
                    return
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid input. Please enter numerical values.")
                return
        else:
            return
        
        self.material_dropdown.clear()
        self.material_dropdown.addItems(Reference_params.keys())
        self.material_dropdown.setCurrentText(new_material)
        self.temperature_input.setText(str(ref_temperature))
        self.me_input.setText(f'{params["M_e"]}')
        self.gn0_input.setText(f'{params["G0"]:.2e}')
        self.tau_e_input.setText(f'{params["tau_e"]:.2e}')
        
        
    def toggle_manual_entry(self):
        read_only = not self.manual_checkbox.isChecked()
        for widget in [self.me_input, self.gn0_input, self.tau_e_input]:
            widget.setReadOnly(read_only)
        self.save_params_checkbox.setVisible(not read_only)
        
        self.adjustSize()

    def search_parameters(self):
        selected_material = self.material_dropdown.currentText()
        try:
            temperature = self.temperature_input.text()
        except:
            QMessageBox.warning(self, "Error", "Please enter the measurement temperature of the rheology measurement")
            
        if not temperature:
            return  

        try:
            temp_key = str(int(float(temperature)))  
            Reference_params = self.loaded_params["Reference_params"]
            params = Reference_params.get(selected_material, {}).get(temp_key)
            if params:
                self.me_input.setText(f'{params["M_e"]}')
                self.gn0_input.setText(f'{params["G0"]:.2e}')
                self.tau_e_input.setText(f'{params["tau_e"]:.2e}')
            else:
                self.me_input.clear()
                self.gn0_input.clear()
                self.tau_e_input.clear()

        except (ValueError, KeyError):
            print("clearing")
            self.me_input.clear()
            self.gn0_input.clear()
            self.tau_e_input.clear()

    def handle_ok(self):
        selected_material = self.material_dropdown.currentText()
        temperature = self.temperature_input.text()
        if not temperature:
            QMessageBox.warning(self, "Error", "Please enter the measurement temperature of the rheology measurement")
            return
        if not self.me_input.text() or not self.gn0_input.text() or not self.tau_e_input.text():

            dialog = ParamShiftDialog(self, selected_material, temperature)
            if dialog.exec_() == QDialog.Accepted:
                params = dialog.get_inputs()
                if params:
                    self.me_input.setText(str(params["M_e"]))
                    self.gn0_input.setText(str(params["G0"]))
                    self.tau_e_input.setText(str(params["tau_e"]))
                self.accept()  
        else:
            if self.save_params_checkbox.isChecked():
                try: 
                    manual_params = {
                        "M_e": float(self.me_input.text()),
                        "G0": float(self.gn0_input.text()),
                        "tau_e": float(self.tau_e_input.text()),
                    }
                    loaded_params["Reference_params"][selected_material][int(temperature)] = manual_params
                    self.save_parameters(loaded_params)
                except:
                    QMessageBox.warning(self, "Error", "Error saving parameters")
            self.accept()  

    def show_help(self):
        QMessageBox.information(self, "Help", 
                                "Enter a material and temperature to autofill parameters. "
                                "If no data is found, you can either enter parameters manually or they will be shifted from database parameters.")

    def get_inputs(self):
        try:
            params = {
                "M_e": float(self.me_input.text()),
                "G0": float(self.gn0_input.text()),
                "tau_e": float(self.tau_e_input.text()),
            }
            return params
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid input. Please enter numerical values.")
            return None
    
    def save_parameters(self, params, file_path=params_file_path):
        with open(file_path, "w") as file:
            json.dump(params, file, indent=4)
