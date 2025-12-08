import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QDialog, QComboBox, QLineEdit, QLabel, QFormLayout, QDialogButtonBox, QCheckBox, QMessageBox)
import sys
from modules.themes import themes
from modules.params import load_parameters, params_file_path

try:
    loaded_params = load_parameters(params_file_path)
    Reference_params = loaded_params["Reference_params"]
    Arrhenius_params = loaded_params["Arrhenius_params"]
    WLF_params = loaded_params["WLF_params"]
except:
    print(f"Error loading {params_file_path}, please ensure it is formatted correctly.")
    sys.exit()

class ParamShiftDialog(QDialog):
    def __init__(self, parent=None, material=None, temperature=None):
        super().__init__(parent)
        self.setWindowTitle('Parameter Shifting')
        
        self.user_confirmed = False
        self.material = material
        self.temperature = int(temperature)

        layout = QVBoxLayout(self)

        self.shift_dropdown = QComboBox(self)
        self.shift_dropdown.addItems(["WLF", "Arrhenius"])
        self.shift_dropdown.currentTextChanged.connect(self.update_shifted_params)
        
        self.app = QApplication.instance()
        if self.app.is_dark:
            self.shift_dropdown.setStyleSheet(themes['dropdown_dark'])
        else:
            self.shift_dropdown.setStyleSheet(themes['dropdown_light'])
        layout.addWidget(QLabel("Shift type:"))
        layout.addWidget(self.shift_dropdown)

        self.manual_shift_checkbox = QCheckBox("Enter shift parameters manually")
        self.manual_shift_checkbox.stateChanged.connect(self.toggle_shift_inputs)
        layout.addWidget(self.manual_shift_checkbox)

        self.shift_input_layout = QFormLayout()
        self.T_ref_dropdown = QComboBox(self)
        if self.app.is_dark:
            self.T_ref_dropdown.setStyleSheet(themes['dropdown_dark'])
        else:
            self.T_ref_dropdown.setStyleSheet(themes['dropdown_light'])
        self.B1_input = QLineEdit(self)
        self.B2_input = QLineEdit(self)
        self.log10alpha_input = QLineEdit(self)
        self.Ea_input = QLineEdit(self)

        self.T_ref_label = QLabel("Reference T (°C):")
        self.B1_label = QLabel("B1:")
        self.B2_label = QLabel("B2:")
        self.log10alpha_label = QLabel("log10(alpha):")
        self.Ea_label = QLabel("Activation Energy (J/mol):")

        self.shift_input_layout.addRow(self.T_ref_label, self.T_ref_dropdown)
        self.shift_input_layout.addRow(self.B1_label, self.B1_input)
        self.shift_input_layout.addRow(self.B2_label, self.B2_input)
        self.shift_input_layout.addRow(self.log10alpha_label, self.log10alpha_input)
        self.shift_input_layout.addRow(self.Ea_label, self.Ea_input)

        layout.addLayout(self.shift_input_layout)
        
        self.T_ref_dropdown.currentTextChanged.connect(self.validate_manual_inputs)
        self.B1_input.textChanged.connect(self.validate_manual_inputs)
        self.B2_input.textChanged.connect(self.validate_manual_inputs)
        self.log10alpha_input.textChanged.connect(self.validate_manual_inputs)
        self.Ea_input.textChanged.connect(self.validate_manual_inputs)

        self.shifted_param_layout = QFormLayout()

        self.shifted_Me_input = QLineEdit(self)
        self.shifted_Me_input.setReadOnly(True)
        self.shifted_G0_input = QLineEdit(self)
        self.shifted_G0_input.setReadOnly(True)
        self.shifted_tau_e_input = QLineEdit(self)
        self.shifted_tau_e_input.setReadOnly(True)

        self.shifted_param_layout.addRow(QLabel("Shifted M_e:"), self.shifted_Me_input)
        self.shifted_param_layout.addRow(QLabel("Shifted G0:"), self.shifted_G0_input)
        self.shifted_param_layout.addRow(QLabel("Shifted tau_e:"), self.shifted_tau_e_input)

        layout.addLayout(self.shifted_param_layout)

        self.save_shifted_checkbox = QCheckBox("Save shifted rheology parameters")
        layout.addWidget(self.save_shifted_checkbox)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(self.buttons)

        self.help_button = self.buttons.addButton("Help", QDialogButtonBox.ActionRole)
        self.help_button.clicked.connect(self.show_help)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.populate_T_ref_options()

        self.toggle_shift_inputs()
        self.update_shifted_params()
        
        self.adjustSize()
    
    def populate_T_ref_options(self):
        self.T_ref_dropdown.clear()
        if self.material in Reference_params:
            self.T_ref_dropdown.addItems(sorted(Reference_params[self.material].keys(), key=int))
        
    def accept(self):
        self.user_confirmed = True  
        self.update_shifted_params()
    
        if not self.get_shift_parameters():  
            return  
    
        super().accept()  
        
    def update_shifted_params(self):
        shift_type = self.shift_dropdown.currentText()
        try:
            if shift_type == "Arrhenius":
                params = self.ArrheniusShift(self.temperature)
            elif shift_type == "WLF":
                params = self.WLFShift(self.temperature)
            else:
                params = None
        except:
            params = None
    
        if params:
            self.shifted_Me_input.setText(f'{params["M_e"]}')
            self.shifted_G0_input.setText(f'{params["G0"]:.3g}')
            self.shifted_tau_e_input.setText(f'{params["tau_e"]:.2e}')
        else:
            self.clear_shifted_params()
    
            if self.user_confirmed and not self.manual_shift_checkbox.isChecked():
                QMessageBox.warning(self, "Error", f"No {shift_type} parameters found for {self.material}. Please enter values manually.")
        
    def clear_shifted_params(self):
        self.shifted_Me_input.clear()
        self.shifted_G0_input.clear()
        self.shifted_tau_e_input.clear()
        
    def toggle_shift_inputs(self):
        is_manual = self.manual_shift_checkbox.isChecked()
        shift_type = self.shift_dropdown.currentText()
        
        self.T_ref_dropdown.setVisible(is_manual)
        self.T_ref_label.setVisible(is_manual)
        self.B1_input.setVisible(is_manual and shift_type == "WLF")
        self.B1_label.setVisible(is_manual and shift_type == "WLF")
        self.B2_input.setVisible(is_manual and shift_type == "WLF")
        self.B2_label.setVisible(is_manual and shift_type == "WLF")
        self.log10alpha_input.setVisible(is_manual and shift_type == "WLF")
        self.log10alpha_label.setVisible(is_manual and shift_type == "WLF")
        self.Ea_input.setVisible(is_manual and shift_type == "Arrhenius")
        self.Ea_label.setVisible(is_manual and shift_type == "Arrhenius")
        
        self.layout().activate()
        self.adjustSize()
        QApplication.processEvents()
        
        if is_manual:
            self.clear_shifted_params()
        if not is_manual:
            self.update_shifted_params()
        
        
    
    def validate_manual_inputs(self):
        shift_type = self.shift_dropdown.currentText()
        try:
            T_ref = float(self.T_ref_dropdown.currentText())
    
            if shift_type == "WLF":
                B1 = float(self.B1_input.text())
                B2 = float(self.B2_input.text())
                log10alpha = float(self.log10alpha_input.text())
            elif shift_type == "Arrhenius":
                Ea = float(self.Ea_input.text())
            else:
                return  
    
            self.update_shifted_params()
    
        except ValueError:
            self.clear_shifted_params()
            pass
    

    def show_help(self):
        QMessageBox.information(self, "Help", 
                                "This shift is only for compatibility with NN models.\n"
                                "Please select the type of shift you would like to use for the material parameters.\n"
                                "The shifted parameters will be used to convert the rheology to the universal space.\n"
                                "If save checkboxes is ticked, shifted rheology parameters and/or shift parameters will be saved to the parameters database for future use")
    
    def get_shift_parameters(self):
        shift_type = self.shift_dropdown.currentText()
        
        try:
            if self.manual_shift_checkbox.isChecked():
                try:
                    T_ref = int(self.T_ref_dropdown.currentText())
                    if shift_type == "WLF":
                        B1 = float(self.B1_input.text())
                        B2 = float(self.B2_input.text())
                        log10alpha = float(self.log10alpha_input.text())
                        return {"B1": B1, "B2": B2, "T_ref": T_ref, "log10alpha": log10alpha}
                    elif shift_type == "Arrhenius":
                        Ea = float(self.Ea_input.text())
                        return {"Ea": Ea, "T_ref": T_ref}
                except ValueError:
                    QMessageBox.warning(self, "Error", "Invalid shift parameter input. Please enter valid numbers.")
                    return None
            else:
                if shift_type == "WLF":
                    if self.material not in WLF_params:
                        return None
                    B1 = WLF_params[self.material]["B1"]
                    B2 = WLF_params[self.material]["B2"]
                    log10alpha = WLF_params[self.material]["log10alpha"]
                    T_ref = WLF_params[self.material]['T_ref']
                    return {"B1": B1, "B2": B2, "T_ref": T_ref, "log10alpha": log10alpha}
                elif shift_type == "Arrhenius":
                    if self.material not in Arrhenius_params:
                        return None
                    Ea = Arrhenius_params[self.material]["Ea"]
                    T_ref = WLF_params[self.material]['T_ref']
                    return {"Ea": Ea, "T_ref": T_ref}
        except:
            QMessageBox.warning(self, "Error", "Invalid parameters")
        
    def ArrheniusShift(self, T):
        
        params = self.get_shift_parameters()
        Ea = params['Ea']
        T_ref = params['T_ref']
        
                
        R = 8.314  

        shift_factor = np.exp((Ea / R) * (1 / (T + 273.15) - 1 / (T_ref + 273.15)))

        self.shifted_params = {
            "M_e": Reference_params[self.material][str(int(T_ref))]["M_e"],
            "G0": Reference_params[self.material][str(int(T_ref))]["G0"],
            "tau_e": Reference_params[self.material][str(int(T_ref))]["tau_e"] * shift_factor,
        }
        
        if self.save_shifted_checkbox.isChecked():
            loaded_params["Reference_params"][self.material][int(self.temperature)] = self.shifted_params
            self.save_shifted_parameters(loaded_params)
            
        return self.shifted_params

    def WLFShift(self, T):
        
        params = self.get_shift_parameters()
        B1 = params['B1']
        B2 = params['B2']
        T_ref = params['T_ref']
        log10alpha = params['log10alpha']
        
        alpha = 10**log10alpha

        log10alpha_T = (-B1*(T-T_ref))/((B2+T_ref)*(B2+T))
        alpha_T = 10**log10alpha_T
        b_T = ((1+alpha*T)*(T_ref+273.15))/((1+alpha*T_ref)*(T+273.15))

        self.shifted_params = {
            "M_e": Reference_params[self.material][str(int(T_ref))]["M_e"],
            "G0": Reference_params[self.material][str(int(T_ref))]["G0"] * b_T,
            "tau_e": Reference_params[self.material][str(int(T_ref))]["tau_e"] * alpha_T,
        }
        
        if self.save_shifted_checkbox.isChecked():
            loaded_params["Reference_params"][f"{self.material}"][f"{int(self.temperature)}"] = self.shifted_params
            self.save_shifted_parameters(loaded_params)
        return self.shifted_params
    
    def save_shifted_parameters(self, params, file_path=params_file_path):
        with open(file_path, "w") as file:
            json.dump(params, file, indent=4)
    
    def get_inputs(self):
        shift_type = self.shift_dropdown.currentText()

        if shift_type == "Arrhenius":
            params = self.ArrheniusShift(self.temperature)
        elif shift_type == "WLF":
            params = self.WLFShift(self.temperature)
        else:
            QMessageBox.warning(self, "Error", "Invalid shift type selected.")
            return 

        return params   
