import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QDialog, QLabel, QLineEdit, QDialogButtonBox, QMessageBox, QSizePolicy, 
                             QCheckBox, QComboBox, QFormLayout)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QSize
import scipy.optimize as opt
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["QT_ENABLE_HIGHDPI_SCALING"]   = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"]             = "1"
import tensorflow as tf
import torch
from NN_models.PytorchPoly_model import PolyModel
from NN_models.PytorchBinary_model import BinaryModel
from NN_models.PytorchMono_model import MonoModel
tf.get_logger().setLevel('ERROR')
from scipy.integrate import simpson as simps
from decimal import Decimal
from modules.themes import themes
from modules.PlotCanvas import PlotCanvas
from modules.UnivDialog import UnivDialog
from modules.MaxwellDialog import MaxwellDialog
from modules.StatsWindow import StatsWindow
from modules.CheckableComboBox import CheckableComboBox
from modules.help_dialog import HelpDialog
import re

def flory_schulz(m, Mn):
    return m/(Mn**2) * np.exp(-m/Mn)

def lognormal(x, mean, sigma):
    return (1/np.sqrt(2*math.pi*sigma**2)) * np.exp(-(np.log(x) - mean)**2 / (2 * sigma**2))
  
def sum_of_lognormals(x, *weights):
    result = np.zeros(len(x))
    for i in range(num_params):
        result = result + (weights[0][i] * lognormal(x, means[i], sigma_poly))
    return result

def sum_of_lognormals_Z(x, *weights):
    if isinstance(weights[0], (list, np.ndarray)):  
        weights = weights[0]
    result = np.zeros(len(x))

    for i in range(num_params2):
        result = result + (weights[i] * lognormal(x, means_Z[i], sigma_poly))
    return result


M_e_PE = 820

num_params = 28
num_params2 = 34

means = np.linspace(np.log(0.1*M_e_PE), np.log(10000*M_e_PE), num_params) 
known_means = np.linspace(np.log(10*M_e_PE), np.log(1000*M_e_PE), 7) 
means_ratio = np.exp(means[1])/np.exp(means[0])
known_means_ratio = np.exp(known_means[1])/np.exp(known_means[0])
sigma_poly = 0.55 * (means_ratio/known_means_ratio)

means_Z = np.linspace(np.log(8e-3), np.log(8e+3), num_params2)

m = np.logspace(2,7,num=300,base=10)
x = np.log(m)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MWD Inference')
        self.setStyleSheet(themes['light_window'])
        screen_width, screen_height = self.screen().size().width(), self.screen().size().height()
        self.default_size = (int(screen_width*0.95),int(screen_height*0.85))
        self.resize(*self.default_size)
        
        self.rheo_data_loaded = False
        self.model_loaded = False
        self.prediction_made = False
        self.modes_fitted = False
        self.univ_space = False
        self.GPC_loaded = False
        self.app = QApplication.instance()
        self.app.is_dark = False

        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        self.button_layout = QVBoxLayout()
        
        self.button_layout.setSpacing(int(4/1080*screen_height))
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.load_rheo_button = QPushButton('Load Rheology', self)
        self.load_rheo_button.setIcon(QIcon('graphics/rheo.png'))
        
        self.univ_rheo_button = QPushButton('Universal Space', self)
        self.univ_rheo_button.setIcon(QIcon('graphics/univ.png'))
                
        self.Maxwell_button = QPushButton('Fit Maxwell Modes', self)
        self.Maxwell_button.setIcon(QIcon('graphics/spring_dash2.png'))
        
        self.Classify_button = QPushButton('Classify MWD', self)
        self.Classify_button.setIcon(QIcon('graphics/Classify.png'))
        
        self.load_GPC_button = QPushButton('Load GPC Data', self)
        self.load_GPC_button.setIcon(QIcon('graphics/MWD.png'))
        
        self.select_model_button = QPushButton('Select NN Model', self)
        self.select_model_button.setIcon(QIcon('graphics/NN.png'))
        
        self.make_prediction_button = QPushButton('Make MWD Prediction', self)
        self.make_prediction_button.setIcon(QIcon('graphics/target.png'))
        
        self.save_prediction_button = QPushButton('Save MWD Prediction', self)
        self.save_prediction_button.setIcon(QIcon('graphics/save.png'))
        
        self.tails_correct_button = QPushButton('MWD Tails Correction', self)
        self.tails_correct_button.setIcon(QIcon('graphics/clean.png'))
        
        self.undo_tails_correct_button = QPushButton('Undo Tails Correction', self)
        self.undo_tails_correct_button.setIcon(QIcon('graphics/undo_clean.png'))
        
        self.Func_buttons = [self.load_rheo_button,self.univ_rheo_button,self.Maxwell_button,self.Classify_button,self.load_GPC_button,
                   self.select_model_button,self.make_prediction_button,self.save_prediction_button, self.tails_correct_button,self.undo_tails_correct_button]
        
        
        
        button_size = (185, int(screen_height*0.05))  
        
        icon_size = QSize(int(screen_height*0.04), int(screen_height*0.04) ) 
        
        
        for button in self.Func_buttons:
            button.setFixedSize(*button_size)
            button.setIconSize(icon_size)
            self.button_layout.addWidget(button)
            button.setStyleSheet(themes['Func_buttons_light'])
            
        self.button_layout.addStretch(3)
        self.tails_correct_button.setVisible(False)
        self.undo_tails_correct_button.setVisible(False)
        
        self.clear_save_button_layout = QHBoxLayout()

        self.clear_rheo_button = QPushButton('Clear Rheology Figure', self)
        self.change_frequency_button = QPushButton('Adjust Frequency Range', self)
        # self.save_rheo_fig_button = QPushButton('Save Rheology Figure', self)
        self.clear_MWD_button = QPushButton('Clear MWD Figure', self)
        # self.save_MWD_fig_button = QPushButton('Save MWD Figure', self)
        self.save_fig_button = QPushButton('Save Figure', self)
        
        self.dark_mode_checkbox = QCheckBox("Dark Mode")
        self.dark_mode_checkbox.stateChanged.connect(self.switch_theme)
        # self.dark_mode_checkbox.setStyleSheet("""
        #     QCheckBox::indicator {
        #         border: 1px solid darkgrey;
        #         border-radius: 2px;}
        # """)
        
        
        self.clear_buttons = [self.clear_rheo_button,self.clear_MWD_button]
        # self.save_buttons = [self.save_rheo_fig_button,self.save_MWD_fig_button]
        clear_button_size = (150, int(screen_height*0.05))
        
        for button in self.clear_buttons:
            button.setStyleSheet(themes['Clear_button_light'])
        self.change_frequency_button.setStyleSheet(themes['Save_button_light'])
        self.save_fig_button.setStyleSheet(themes['Save_button_light'])   
        
        self.clear_save_button_layout.addStretch(1)
        self.dynamic_plot_dropdown = CheckableComboBox()
        self.clear_save_button_layout.addWidget(self.dynamic_plot_dropdown)
        self.clear_save_button_layout.addStretch(1)
        self.checkbox_dict_ax1 = {}
        self.checkbox_dict_ax2 = {}
        for button in [self.clear_rheo_button,self.change_frequency_button,self.clear_MWD_button,self.save_fig_button]:   
            button.setFixedSize(*clear_button_size)
            self.clear_save_button_layout.addWidget(button)
            self.clear_save_button_layout.addStretch(1)     
        self.clear_save_button_layout.addWidget(self.dark_mode_checkbox)
        self.clear_save_button_layout.addStretch(1)
        
        self.labels_and_help_layout = QHBoxLayout()
        
        self.model_label = QLabel("No NN model selected", self)
        self.rheo_label = QLabel("No rheology data loaded", self)
        self.GPC_label = QLabel("No GPC data loaded", self)
        
        self.help_button = QPushButton("Help", self)
        self.help_button.setFixedSize(*(50,50))
        self.help_button.setStyleSheet(themes['help_button_light'])
        
        labels_layout = QVBoxLayout()
        
        label_size = (500, 20)  
        self.model_label.setFixedSize(*label_size)
        self.rheo_label.setFixedSize(*label_size)
        self.GPC_label.setFixedSize(*label_size)
        
        for label in [self.model_label,self.rheo_label,self.GPC_label]:
            labels_layout.addWidget(label)
            
        self.labels_and_help_layout.addLayout(labels_layout)
        self.labels_and_help_layout.addStretch(1)
        self.labels_and_help_layout.addWidget(self.help_button)
        
        
        self.buttons_and_canvas_layout = QHBoxLayout()
        self.clear_and_canvas_layout = QVBoxLayout()
        
        
        self.canvas = PlotCanvas(self)  
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(int(screen_width * 0.5), 
                           int(screen_height * 0.35))
        
        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()
        
        self.canvas.plot_added.connect(self.add_checkbox)
        self.canvas.plot_removed.connect(self.remove_checkbox)
        
        self.clear_and_canvas_layout.addLayout(self.clear_save_button_layout)
        self.clear_and_canvas_layout.addWidget(self.canvas, 3)

        self.class_layout = QHBoxLayout()
        
        self.display_label_layout = QVBoxLayout()
        self.Class_display_label = QLabel("Predicted MWD Class: N/A", self)
        font = QFont()
        font.setPointSize(12)
        self.Class_display_label.setFont(font)
        self.display_label_layout.addWidget(self.Class_display_label)
                
        self.dropdown_layout = QHBoxLayout()
        self.dropdown_label = QLabel("Select MWD Class for Prediction:", self)
        self.dropdown_label.setFont(font)
        self.class_to_use = 0
        self.class_to_use_dropdown = QComboBox(self)
        self.class_to_use_dropdown.addItems(["Polydisperse", "Monodisperse", "Bidisperse"])  
        self.class_to_use_dropdown.setFixedWidth(150)
        self.class_to_use_dropdown.setStyleSheet(themes['dropdown_light'])
        
        self.PDI_change_button = QPushButton("Change Prediction PDI")
        self.PDI_change_button.clicked.connect(self.PDI_change)
        self.PDI_change_button.setFixedSize(*(200,int(screen_height*0.03)))
        self.PDI_change_button.setStyleSheet(themes['PDI_change_button_light'])
        self.PDI_change_button.setVisible(False)
        
        self.dropdown_layout.addStretch(1)
        self.dropdown_layout.addWidget(self.dropdown_label)
        self.dropdown_layout.addWidget(self.class_to_use_dropdown)
        self.dropdown_layout.addWidget(self.PDI_change_button)
        
        self.value_map = {"Polydisperse": 0, "Monodisperse": 1, "Bidisperse": 2}
        
        self.class_to_use_dropdown.currentTextChanged.connect(lambda text: self.on_dropdown_change(text))
          
        self.class_layout.addStretch(1)
        self.class_layout.addLayout(self.display_label_layout)
        self.class_layout.addSpacing(50)
        self.class_layout.addLayout(self.dropdown_layout)
        self.class_layout.addStretch(1)
        
        self.Est_Mn_label = QLabel("Estimated Mn: N/A")
        self.Est_Mw_label = QLabel("Estimated Mw: N/A")
        self.Est_PDI_label = QLabel("Estimated PDI: N/A")

        self.GPC_Mn_label = QLabel("GPC Mn: N/A")
        self.GPC_Mw_label = QLabel("GPC Mw: N/A")
        self.GPC_PDI_label = QLabel("GPC PDI: N/A")

        self.predicted_stats_button = QPushButton("Predicted MWD Stats")
        self.predicted_stats_button.clicked.connect(self.show_predicted_stats)

        self.gpc_stats_button = QPushButton("GPC Stats")
        self.gpc_stats_button.clicked.connect(self.show_gpc_stats)
        
        self.values_layout = QHBoxLayout()
        
        self.values_button_size = (300,int(screen_height*0.05))
        self.stats_buttons = [self.predicted_stats_button,self.gpc_stats_button]
        for button in self.stats_buttons:
            button.setStyleSheet(themes['stats_button_light'])
            self.values_layout.addWidget(button)
            button.setFixedSize(*self.values_button_size)

        self.predicted_stats_window = StatsWindow("Predicted MWD Stats", [
            self.Est_Mn_label, self.Est_Mw_label, self.Est_PDI_label], parent=self)

        self.gpc_stats_window = StatsWindow("GPC Stats", [
            self.GPC_Mn_label, self.GPC_Mw_label, self.GPC_PDI_label], parent=self)
        
        self.clear_and_canvas_layout.addLayout(self.class_layout)
        self.clear_and_canvas_layout.addSpacing(50)
        self.clear_and_canvas_layout.addLayout(self.values_layout)
        self.clear_and_canvas_layout.addSpacing(50)
        
        self.buttons_and_canvas_layout.addLayout(self.button_layout)
        self.buttons_and_canvas_layout.addLayout(self.clear_and_canvas_layout)
                
        self.main_layout.addLayout(self.buttons_and_canvas_layout)   
        self.main_layout.addLayout(self.labels_and_help_layout)  

        self.load_rheo_button.clicked.connect(self.load_rheo_file)
        self.univ_rheo_button.clicked.connect(self.univ_norm)
        self.Maxwell_button.clicked.connect(self.Maxwell_Fitting)
        self.Classify_button.clicked.connect(self.ClassifyMWD)
        self.load_GPC_button.clicked.connect(self.load_GPC)
        self.select_model_button.clicked.connect(self.select_model)
        self.make_prediction_button.clicked.connect(self.make_prediction)
        self.save_prediction_button.clicked.connect(self.save_prediction)
        self.tails_correct_button.clicked.connect(self.clean_up_pred)
        self.undo_tails_correct_button.clicked.connect(self.revert_clean_pred)
        self.clear_rheo_button.clicked.connect(self.clear_rheo_plot)
        self.clear_MWD_button.clicked.connect(self.clear_MWD_plot)
        # self.save_rheo_fig_button.clicked.connect(self.save_rheo_fig)
        # self.save_MWD_fig_button.clicked.connect(self.save_MWD_fig)
        self.save_fig_button.clicked.connect(self.save_figure)
        self.change_frequency_button.clicked.connect(self.change_frequency_range)
        self.help_button.clicked.connect(self.show_help)
        
    def show_help(self):
        if hasattr(self, 'help_dialog') and self.help_dialog is not None:
            self.help_dialog.close()
        self.help_dialog = HelpDialog(self)
        self.help_dialog.show()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.canvas.draw()
        self.canvas.update()
        if hasattr(self, "canvas") and self.canvas is not None:  
            self.canvas.fig.tight_layout()
            self.canvas.draw_idle()
        self.update()
        QApplication.processEvents()

    
    def add_checkbox(self, label, axis):
        checkbox_dict = self.checkbox_dict_ax1 if axis == "ax1" else self.checkbox_dict_ax2

        if label in checkbox_dict:
            return  

        checkbox = QCheckBox(label)
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(lambda state, lbl=label, ax=axis: self.toggle_plot(lbl, ax, state))
        self.dynamic_plot_dropdown.add_checkbox(
            label, lambda lbl, state, ax=axis: self.toggle_plot(lbl, ax, state)
        )
        checkbox_dict[label] = checkbox

    def remove_checkbox(self, label, axis):
        checkbox_dict = self.checkbox_dict_ax1 if axis == "ax1" else self.checkbox_dict_ax2

        if label in checkbox_dict:
            self.dynamic_plot_dropdown.remove_checkbox(label)
            del checkbox_dict[label]
        

    def toggle_plot(self, label, axis, state):
        visible = state == 2  # 2 means Checked, 0 means Unchecked
        self.canvas.toggle_visibility(label, axis, visible)
        if axis == "ax1":
            self.canvas.autoscale_plot1()
        elif axis == "ax2":
            self.canvas.autoscale_plot2()
        QApplication.processEvents()
        
                
    def switch_theme(self):
        if not self.app.is_dark:
            self.setStyleSheet(themes['dark_window'])
            for button in self.Func_buttons:
                button.setStyleSheet(themes['Func_buttons_dark'])
            for button in self.clear_buttons:
                button.setStyleSheet(themes['Clear_button_dark'])
            for button in [self.save_fig_button,self.change_frequency_button]:
                button.setStyleSheet(themes['Save_button_dark'])
            for button in self.stats_buttons:
                button.setStyleSheet(themes['stats_button_dark'])
            self.PDI_change_button.setStyleSheet(themes['PDI_change_button_dark'])
            self.help_button.setStyleSheet(themes['help_button_dark'])
            if hasattr(self, 'help_dialog') and self.help_dialog is not None:
                self.help_dialog.zoom_in_button.setStyleSheet(themes['Zoom_button_dark'])
                self.help_dialog.zoom_out_button.setStyleSheet(themes['Zoom_button_dark'])
            self.class_to_use_dropdown.setStyleSheet(themes['dropdown_dark'])
            self.canvas.fig.patch.set_facecolor((145/255, 145/255, 155/255))
            self.canvas.draw()
        else:
            self.setStyleSheet(themes['light_window'])
            for button in self.Func_buttons:
                button.setStyleSheet(themes['Func_buttons_light'])
            for button in self.clear_buttons:
                button.setStyleSheet(themes['Clear_button_light'])
            for button in [self.save_fig_button,self.change_frequency_button]:
                button.setStyleSheet(themes['Save_button_light'])
            for button in self.stats_buttons:
                button.setStyleSheet(themes['stats_button_light'])
            self.PDI_change_button.setStyleSheet(themes['PDI_change_button_light'])
            self.help_button.setStyleSheet(themes['help_button_light'])
            if hasattr(self, 'help_dialog') and self.help_dialog is not None:
                self.help_dialog.zoom_in_button.setStyleSheet(themes['Zoom_button_light'])
                self.help_dialog.zoom_out_button.setStyleSheet(themes['Zoom_button_light'])
            self.class_to_use_dropdown.setStyleSheet(themes['dropdown_light'])
            self.canvas.fig.patch.set_facecolor((220/255, 220/255, 220/255))
            self.canvas.draw()
        self.app.is_dark = not self.app.is_dark
        
    def show_predicted_stats(self):
        if hasattr(self, 'predicted_stats_window') and self.predicted_stats_window is not None:
            self.predicted_stats_window.close()
        self.predicted_stats_window = StatsWindow("Predicted MWD Stats", [
            self.Est_Mn_label, self.Est_Mw_label, self.Est_PDI_label], parent=self)
        self.predicted_stats_window.show()

    def show_gpc_stats(self):
        if hasattr(self, 'gpc_stats_window') and self.gpc_stats_window is not None:
            self.gpc_stats_window.close()
        self.gpc_stats_window = StatsWindow("GPC Stats", [
            self.GPC_Mn_label, self.GPC_Mw_label, self.GPC_PDI_label], parent=self)
        self.gpc_stats_window.show()
        
    def update_stats(self, pred_MWD, x):
        self.Est_Mn = 1 / np.trapz(pred_MWD * np.exp(-x), x=x)
        self.Est_Mw = np.trapz(pred_MWD * np.exp(x), x=x)
        self.Est_PDI = self.Est_Mw / self.Est_Mn

        def format_e(n):
            a = '%E' % n
            return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]

        self.Est_Mn_label.setText(f'Estimated Mn: {format_e(Decimal(self.Est_Mn))} (g/mol)')
        self.Est_Mw_label.setText(f'Estimated Mw: {format_e(Decimal(self.Est_Mw))} (g/mol)')
        self.Est_PDI_label.setText(f'Estimated PDI: {self.Est_PDI:.2f}')

        self.predicted_stats_window.update()  
        self.gpc_stats_window.update()  
    
    def set_dropdown_value(self, value):
        for text, index in self.value_map.items():
            if index == value:
                self.class_to_use_dropdown.setCurrentText(text)
                self.class_to_use = value
                break
        
    def on_dropdown_change(self, selected_text):
        
        self.class_to_use = self.value_map.get(selected_text, -1)
        
        
    def load_rheo_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Rheology File", "data", "All Files (*);;Text Files (*.txt)", options=options)
        if not file_path:
            return
        file_name = os.path.basename(file_path)
        
        self.rheo_label.setText(f"Loaded Rheology: {file_name}")
        
        unsorted_Gp_data_Exp = []
        unsorted_Gpp_data_Exp = []
        unsorted_w_values_Exp = []
        
        negative_w_indices = []
        negative_Gp_indices = []
        negative_Gpp_indices = []
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
            
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue  
            
            values = re.split(r'[,\s]+', line)
    
            if len(values) < 3:
                continue
    
            try:
                value1 = float(values[0])
                value2 = float(values[1])
                value3 = float(values[2])
            except ValueError:
                if i == 0:  
                    continue
                QMessageBox.warning(self, "Error", f"Invalid data in row {i+1}: Non-numeric values detected.\nPlease ensure file is in the correct format.")
                return
            if value1 <= 0:
                negative_w_indices.append(i+1)
            if value2 <= 0:
                negative_Gp_indices.append(i+1)
            if value3 <= 0:
                negative_Gpp_indices.append(i+1)
            
            unsorted_w_values_Exp.append(value1)
            unsorted_Gp_data_Exp.append(value2)
            unsorted_Gpp_data_Exp.append(value3)
            
        warning_msgs = []
        if negative_w_indices:
            warning_msgs.append(f"Negative frequency (ω) values found at line(s): {', '.join(map(str, negative_w_indices))}")
        if negative_Gp_indices:
            warning_msgs.append(f"Negative G' values found at line(s): {', '.join(map(str, negative_Gp_indices))}")
        if negative_Gpp_indices:
            warning_msgs.append(f"Negative G'' values found at line(s): {', '.join(map(str, negative_Gpp_indices))}")
        if warning_msgs:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("\n".join(warning_msgs) + "\n\nDo you want to ignore these rows and continue loading the rest of the data?\n\nChoose 'Ignore' to skip the problematic rows, or 'Cancel' to fix the file yourself and try again.")
            ignore_button = msg_box.addButton("Ignore", QMessageBox.AcceptRole)
            cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
            msg_box.exec_()
            if msg_box.clickedButton() == cancel_button:
                return


        filtered_w = []
        filtered_Gp = []
        filtered_Gpp = []
        for i in range(len(unsorted_w_values_Exp)):
            if unsorted_w_values_Exp[i] > 0 and unsorted_Gp_data_Exp[i] > 0 and unsorted_Gpp_data_Exp[i] > 0:
                filtered_w.append(unsorted_w_values_Exp[i])
                filtered_Gp.append(unsorted_Gp_data_Exp[i])
                filtered_Gpp.append(unsorted_Gpp_data_Exp[i])
                
        unsorted_w_values_Exp = np.array(filtered_w)
        unsorted_Gp_data_Exp = np.array(filtered_Gp)
        unsorted_Gpp_data_Exp = np.array(filtered_Gpp)
        
        if len(unsorted_w_values_Exp) == 0:
            QMessageBox.warning(self, "Error", "No valid data found in the file.\nPlease ensure file is a rheology file in the correct format.")
            return
        
        sorted_indices = np.argsort(unsorted_w_values_Exp)
        self.w_values_Exp = unsorted_w_values_Exp[sorted_indices]                 
        self.Gp_data_Exp = unsorted_Gp_data_Exp[sorted_indices] 
        self.Gpp_data_Exp = unsorted_Gpp_data_Exp[sorted_indices] 
        
        self.original_w_values_Exp = self.w_values_Exp[:]
        self.original_Gp_data_Exp = self.Gp_data_Exp[:]
        self.original_Gpp_data_Exp = self.Gpp_data_Exp[:]
        
        self.canvas.clear_plot1()
        self.canvas.change_axes_plot1(r'$\omega$ (rad/s)', r"G', G'' (Pa)")
        self.canvas.plot_scatter_on_axes1(self.w_values_Exp, self.Gp_data_Exp, facecolors='none', edgecolors='black', linewidth = 1, label = "G' Input", size = 20)
        self.canvas.plot_scatter_on_axes1(self.w_values_Exp, self.Gpp_data_Exp, facecolors='none', edgecolors='red', linewidth = 1, label = "G'' Input", size = 20)
        
        
        self.canvas.autoscale_plot1()
        self.rheo_data_loaded = True
            
    
    def univ_norm(self):
        if not self.rheo_data_loaded:
            QMessageBox.warning(self, "Error", "Please load the rheology data first before attempting to perform shift.")
            return 
        
        
        dialog = UnivDialog(self)
        
        dialog_exec = dialog.exec_()
        
        if dialog_exec == QDialog.Rejected:
            return
        
        elif dialog_exec == QDialog.Accepted:
            inputs  = dialog.get_inputs()
            
            if inputs is None:
                QMessageBox.warning(self, "Error", "Invalid input detected, try again.")
                return
                 
            self.M_e = inputs["M_e"]
            self.G0 = inputs["G0"]
            self.tau_e = inputs["tau_e"]

        
        
        self.z = m / self.M_e
        
        self.canvas.clear_plot1()
        
        self.w_values_Exp = self.w_values_Exp * self.tau_e
        self.Gp_data_Exp = self.Gp_data_Exp / self.G0
        self.Gpp_data_Exp = self.Gpp_data_Exp / self.G0
        

        
        
        self.canvas.clear_plot1()
        
        self.canvas.change_axes_plot1(r'$\omega$ * $\tau_e$', r"$G' / G_N^0$, $G'' / G_N^0$")
        
        self.canvas.plot_scatter_on_axes1(self.w_values_Exp, self.Gp_data_Exp, facecolors='none', edgecolors='black', linewidth = 1, label = "G' Input Univ", size = 20)
        self.canvas.plot_scatter_on_axes1(self.w_values_Exp, self.Gpp_data_Exp, facecolors='none', edgecolors='red', linewidth = 1, label = "G'' Input Univ", size = 20)
            
            
        self.canvas.autoscale_plot1() 
        
        self.univ_space = True

        
           
    def Maxwell_Fitting(self):
        
        if not self.rheo_data_loaded:
            QMessageBox.warning(self, "Error", "Please load the rheology data first before fitting Maxwell modes.")
            return        
        if not self.univ_space:
            QMessageBox.warning(self, "Error", "Please convert rheology to universal space before fitting Maxwell modes.")
            return
        
        def G_prime_fit(w_alpha, g_values):
            sum_over_i = 0.0
            w_squared = w_alpha*w_alpha
            numerator = g_values * tau_values * tau_values
            denom = 1 + (w_squared * tau_values * tau_values)
            to_sum = numerator/denom
            sum_over_i = np.sum(to_sum)
            G_prime_for_alpha = w_squared * sum_over_i
            return G_prime_for_alpha

        def G_dub_prime_fit(w_alpha, g_values):
            sum_over_j = 0.0
            w_squared = w_alpha*w_alpha
            numerator = g_values * tau_values
            denom = 1 + (w_squared * tau_values * tau_values)
            to_sum = numerator/denom
            sum_over_j = np.sum(to_sum)
            G_dub_prime_for_alpha = w_alpha * sum_over_j
            return G_dub_prime_for_alpha

        def G_concat_fit(w_values, *log_g_values):
            log_g_second_diffs = np.array([])
            for i in range(len(log_g_values)-2):
                g_i_second_diff = log_g_values[i+2] + log_g_values[i] - 2*log_g_values[i+1]
                log_g_second_diffs = np.append(log_g_second_diffs, g_i_second_diff)
            g_values = np.exp(log_g_values)
            G_prime_fit_values = np.array([])
            G_dub_prime_fit_values = np.array([])
            for alpha in w_values:
                G_prime_max = G_prime_fit(alpha, g_values)
                G_prime_fit_values = np.append(G_prime_fit_values, G_prime_max)
                G_dub_prime_max = G_dub_prime_fit(alpha, g_values)
                G_dub_prime_fit_values = np.append(G_dub_prime_fit_values, G_dub_prime_max)
            G_concat = np.concatenate((G_prime_fit_values, G_dub_prime_fit_values), axis=0) 
            G_log_concat_diff = (np.log(G_concat) - G_concat_inp_LN) 
            G_concat_with_diffs = np.concatenate((G_log_concat_diff, lam*log_g_second_diffs), axis = 0)
            return G_concat_with_diffs
        
        dialog = MaxwellDialog(self)
        
        dialog_exec = dialog.exec_()
        
        if dialog_exec == QDialog.Rejected:
            return
        
        elif dialog_exec == QDialog.Accepted:
            try:
                tau_H, tau_L, modes_per_decade = dialog.get_inputs()
            except:
                return
        
        self.canvas.clear_plot1()
        self.canvas.text_plot1("Fitting Spectrum...", 'green', 20)
        QApplication.processEvents()
        
        w_values_Exp = self.w_values_Exp
        self.nat_log_w_values_Exp = np.log(self.w_values_Exp)
        self.log10_w_values_Exp = np.log10(self.w_values_Exp)
        self.G_concat_inp_nat_Exp = np.concatenate((self.Gp_data_Exp, self.Gpp_data_Exp))
        
        longtau = -tau_L
        smalltau = -tau_H
        numoftau = 1 + (longtau-smalltau) * modes_per_decade
        tau_values = np.logspace(longtau,smalltau,numoftau,base=10)
        
        
        G_len_zeros = np.zeros(2*np.shape(self.w_values_Exp)[0] + len(tau_values) - 2)
        
        N = len(self.w_values_Exp)
        
        p_w = N / (max(np.log10(w_values_Exp))-min(np.log10(w_values_Exp)))
        p_w_base = 95 / (14)
        lam = 1 * (p_w/p_w_base)**0.5
        
        numomega = np.shape(self.w_values_Exp)[0]
        
        if self.univ_space == True:
            bounds = ([-100] * numoftau, [50]*numoftau)
        else:
            bounds = ([-40] * numoftau, [50]*numoftau)
        
        G_concat_inp_LN_Exp = np.log(self.G_concat_inp_nat_Exp)
        G_concat_inp_LN = G_concat_inp_LN_Exp

        if numoftau < numomega:
            G_split = np.array_split(G_concat_inp_LN_Exp[0:int(numomega)], numoftau)
            G_dub_split = np.array_split(G_concat_inp_LN_Exp[int(numomega):], numoftau)
            log_prime_means = np.array([np.mean(subarray) for subarray in G_split])
            log_dub_means = np.array([np.mean(subarray) for subarray in G_dub_split])
            initial_guess = (log_prime_means + log_dub_means) / 2  - 2
        else:
            desired_size = numoftau        
            interpolated_indices = np.linspace(0, self.G_concat_inp_nat_Exp.shape[0] / 2 - 1, desired_size)
            interpolated_prime = np.interp(interpolated_indices, np.arange(self.G_concat_inp_nat_Exp.shape[0]/2), G_concat_inp_LN_Exp[0:int(numomega)])
            interpolated_dub = np.interp(interpolated_indices, np.arange(self.G_concat_inp_nat_Exp.shape[0]/2), G_concat_inp_LN_Exp[int(numomega):])
            initial_guess = (interpolated_prime + interpolated_dub) / 2 - 2

        optimiseresult, pcov = opt.curve_fit(G_concat_fit, self.w_values_Exp, G_len_zeros, p0=initial_guess, bounds=bounds, method='trf')
        self.optimiseresult = optimiseresult    

        pred_G_prime = np.array([])
        pred_G_dub_prime = np.array([])

        for alpha in w_values_Exp:
            predicted_Gprime = G_prime_fit(alpha, np.exp(optimiseresult))
            pred_G_prime = np.append(pred_G_prime, predicted_Gprime)
            predicted_Gdubprime = G_dub_prime_fit(alpha, np.exp(optimiseresult))
            pred_G_dub_prime = np.append(pred_G_dub_prime, predicted_Gdubprime)            
        
        self.canvas.clear_plot1()
        if self.univ_space == False:
            self.canvas.plot_scatter_on_axes1(self.w_values_Exp, self.Gp_data_Exp, facecolors='none', edgecolors='black', linewidth = 1, label = "G' Input", size = 20)
            self.canvas.plot_scatter_on_axes1(self.w_values_Exp, self.Gpp_data_Exp, facecolors='none', edgecolors='red', linewidth = 1, label = "G'' Input", size = 20)
        else:
            self.canvas.plot_scatter_on_axes1(self.w_values_Exp, self.Gp_data_Exp, facecolors='none', edgecolors='black', linewidth = 1, label = "G' Input Univ", size = 20)
            self.canvas.plot_scatter_on_axes1(self.w_values_Exp, self.Gpp_data_Exp, facecolors='none', edgecolors='red', linewidth = 1, label = "G'' Input Univ", size = 20)
        
        self.canvas.plot_line_on_axes1(self.w_values_Exp, pred_G_prime, linetype='-', color='black', label = "G' Maxwell Fit", linewidth=1)
        self.canvas.plot_line_on_axes1(self.w_values_Exp, pred_G_dub_prime, linetype='-', color='red', label = "G'' Maxwell Fit", linewidth=1)
        
        mode_omega = 1/tau_values
        
        self.canvas.plot_scatter_on_axes1(mode_omega, np.exp(optimiseresult), facecolors='green', edgecolors='green', linewidth=1, label='Maxwell Modes', size=20)
        
        self.canvas.autoscale_plot1()
        
        self.modes_fitted = True
    
    def select_model(self):
        directory = "NN_models"
        if not os.path.exists(directory):
            QMessageBox.warning(self, "Error", f"The directory '{directory}' does not exist. Please ensure the folder containing NN models is present.")
            return

        class_to_use_map = {
            0: "Polydisperse",
            1: "Monodisperse",
            2: "Bidisperse"
        }
        class_filter = class_to_use_map.get(self.class_to_use, "")
        files = [f for f in os.listdir(directory) if class_filter in str(f) and f.endswith(".pth")]

        if not files:
            QMessageBox.warning(self, "Error", f"No models found for class '{class_filter}' in '{directory}'.")
            return
        
        display_names = [os.path.splitext(f)[0] for f in files]

        dialog = QDialog(self)
        dialog.setWindowTitle("Select NN Model")
        layout = QVBoxLayout(dialog)
        
        font = QFont()
        font.setPointSize(16)
        dialog.setFont(font)
        
        if self.app.is_dark:
            dialog.setStyleSheet(themes['dark_window'])
        else:
            dialog.setStyleSheet(themes['light_window'])
        
        
        label_1 = QLabel(f"You have selected the class '{class_filter}'.\nIf this is incorrect, close this dialog and change the option in the dropdown.", dialog)
        layout.addWidget(label_1)
        label_2 = QLabel(f"Select a model for MWD inference:", dialog)
        layout.addWidget(label_2)

        dropdown = QComboBox(dialog)
        dropdown.addItems(display_names)
        layout.addWidget(dropdown)
        
        if self.app.is_dark:
            dropdown.setStyleSheet(themes['dropdown_dark'])
        else:
            dropdown.setStyleSheet(themes['dropdown_light'])
        
        label_3 = QLabel("NNs are not deterministic, so you may want to try multiple models and compare.", dialog)
        layout.addWidget(label_3)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            selected_display_name = dropdown.currentText()
            selected_file = f"{selected_display_name}.pth"
            file_path = os.path.join(directory, selected_file)

            try:
                if self.class_to_use == 0:
                    self.model = PolyModel()
                elif self.class_to_use == 1:    
                    self.model = MonoModel()
                elif self.class_to_use == 2:
                    self.model = BinaryModel()  
                
                
                self.model.load_state_dict(torch.load(file_path, weights_only=True, map_location=torch.device('cpu')))
                self.model.eval()
                self.model_label.setText(f"Loaded model: {selected_file}")
                self.model_loaded = True
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load model: {e}")
            
    def ClassifyMWD(self):
        if not self.rheo_data_loaded:
            QMessageBox.warning(self, "Error", "Please load the rheology data first before attempting to classify.")
            return
        if not self.univ_space:
            QMessageBox.warning(self, "Error", "Please convert rheology to universal space before attempting to classify.")
            return
        if not self.modes_fitted:
            QMessageBox.warning(self, "Error", "Please fit relaxation spectrum before attempting to classify.")
            return 
        
        directory = "NN_models"
        if not os.path.exists(directory):
            QMessageBox.warning(self, "Error", f"The directory '{directory}' does not exist. Please ensure the folder containing NN models is present.")
            return

        
        file_names = [f for f in os.listdir(directory) if "Classifier" in str(f) and f.endswith(".keras")]
        if not file_names:
            QMessageBox.warning(self, "Error", f"No classification models found in '{directory}'.")
            return
        pred_class_nums = np.zeros(3)
        self.classes = ['Polydisperse','Monodisperse','Bidisperse']
        X_val = np.log10(np.exp(self.optimiseresult))
        X_val = X_val.reshape(1, -1)
        
        for file in file_names:
            
            path = os.path.join(directory, file)
            self.model = tf.keras.models.load_model(path)
            
            prediction = self.model.predict(X_val, verbose=0)
            
            indiv_predicted_class = np.argmax(prediction)
            
            pred_class_nums[indiv_predicted_class] += 1
        
        predicted_class = np.argmax(pred_class_nums)
        self.predicted_label = self.classes[predicted_class]
        
        self.Class_display_label.setText(f'Predicted MWD Class: {self.predicted_label} {int(max(pred_class_nums))}/{len(file_names)} model(s)')
        
        self.set_dropdown_value(predicted_class)
        
        
    
    def make_prediction(self):
        self.cleaned_pred = False
        if not self.model_loaded:
            QMessageBox.warning(self, "Error", "Please load a NN model before attempting to make a prediction.")
            return  
        if not self.modes_fitted:
            QMessageBox.warning(self, "Error", "Please fit relaxation spectrum before attempting to make a prediction.")
            return  
        if self.prediction_made == True:
            self.canvas.remove_single_plot("Predicted")
        
        X_val = np.log10(np.exp(np.stack((self.optimiseresult, self.optimiseresult))))
        X_val = np.concatenate((X_val, X_val), axis=1)
        X_val = np.reshape(X_val, (2, 1, 2, -1))  
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        try:
            with torch.no_grad():
                prediction = torch.abs(self.model(X_val_tensor))
            self.prediction = prediction.numpy()
            self.prediction[self.prediction < 0.0005] = 0
            
            if self.class_to_use == 0:
                if not self.univ_space:
                    QMessageBox.warning(self, "Error", "PLease convert data to universal space to make a prediction.")
                    return 
                
                if self.univ_space == True:
                    pred_MWD = sum_of_lognormals_Z(self.z, self.prediction[0])
                    pred_MWD = pred_MWD / np.trapz(pred_MWD, x = np.log(self.z))
                    self.pred_MWD = pred_MWD
                    self.canvas.plot_line_on_axes2(m, pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
                    for k in range(num_params2):
                        individual_pred_curve = self.prediction[0, k] * lognormal(self.z, means_Z[k], sigma_poly)
                        self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means)), linewidth=1)
                                
                    self.Est_Mn = 1/np.trapz(pred_MWD*np.exp(-x), x=x)
                    self.Est_Mw = np.trapz(pred_MWD*np.exp(x), x=x)
                    self.Est_PDI = self.Est_Mw/self.Est_Mn
                    
                    def format_e(n):
                        a = '%E' % n
                        return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
                    self.Est_Mn_label.setText(f'Estimated Mn: {format_e(Decimal(self.Est_Mn))} (g/mol)')
                    self.Est_Mw_label.setText(f'Estimated Mw: {format_e(Decimal(self.Est_Mw))} (g/mol)')
                    self.Est_PDI_label.setText(f'Estimated PDI: {self.Est_PDI:.2f}')  
                    
                    self.predicted_stats_window.update() 
                    
                    self.prediction_made = True
                    
                    self.tails_correct_button.setVisible(True)
                    
            elif self.class_to_use == 1:
                if not self.univ_space:
                    QMessageBox.warning(self, "Error", "PLease convert data to universal space to make a monodisperse prediction.")
                    return 
                PDI = 1.03
                sigma_mono = np.sqrt(np.log(PDI))
                Z_pred = 10**float(self.prediction[0])
                mean_Z_pred = np.log(Z_pred)-(sigma_mono**2)/2
                
                pred_MWD = lognormal(self.z, mean_Z_pred, sigma_mono)
                pred_MWD = pred_MWD / simps(pred_MWD, x = np.log(self.z))
                self.pred_MWD = pred_MWD
                    
                self.canvas.plot_line_on_axes2(m, pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
                
                self.Est_Mw = Z_pred * self.M_e
                self.Est_Mn = self.Est_Mw / PDI
                self.Est_PDI = PDI
                def format_e(n):
                    a = '{:.3E}'.format(n)
                    return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
                self.Est_Mn_label.setText(f'Estimated Mn: {format_e(Decimal(self.Est_Mn))} (g/mol)')
                self.Est_Mw_label.setText(f'Estimated Mw: {format_e(Decimal(self.Est_Mw))} (g/mol)')
                self.Est_PDI_label.setText(f'Assumed PDI: {self.Est_PDI:.2f}')
                
                self.predicted_stats_window.update()  
                
                self.prediction_made = True
                self.PDI_change_button.setVisible(True)
                
            elif self.class_to_use == 2:
                if not self.univ_space:
                    QMessageBox.warning(self, "Error", "PLease convert data to universal space to make a binary prediction.")
                    return
                PDI = 1.03
                sigma_bin = np.sqrt(np.log(PDI))
                
                ZS_pred = 10**float(self.prediction[0][0])
                ZL_pred = 10**float(self.prediction[0][1])
                
                self.phiL_pred = float(self.prediction[0][2])
                self.phiS_pred = 1 - self.phiL_pred 
                
                mean_ZS_pred = np.log(ZS_pred)-(sigma_bin**2)/2
                mean_ZL_pred = np.log(ZL_pred)-(sigma_bin**2)/2
                
                pred_MWD = self.phiL_pred * lognormal(self.z, mean_ZL_pred, sigma_bin) + self.phiS_pred * lognormal(self.z, mean_ZS_pred, sigma_bin)
                pred_MWD = pred_MWD / np.trapz(pred_MWD, x = np.log(self.z))
                self.pred_MWD = pred_MWD
                    
                self.canvas.plot_line_on_axes2(m, pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
                
                self.Est_MwS = ZS_pred * self.M_e
                self.Est_MnS = self.Est_MwS / PDI
                
                self.Est_MwL = ZL_pred * self.M_e
                self.Est_MnL = self.Est_MwL / PDI
                self.Est_phiL = self.phiL_pred
                self.Est_phiS = self.phiS_pred
                
                self.Est_PDIL = PDI
                self.Est_PDIS = PDI
                
                def format_e(n):
                    a = '{:.3E}'.format(n)
                    return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
                self.Est_Mn_label.setText(f'Component 1 (Mw): {self.phiS_pred*100:.2f}%   {format_e(Decimal(self.Est_MwS))} (g/mol)')
                self.Est_Mw_label.setText(f'Component 2 (Mw): {self.phiL_pred*100:.2f}%   {format_e(Decimal(self.Est_MwL))} (g/mol)')
                self.Est_PDI_label.setText(f'PDI 1: {self.Est_PDIS:.2f}, PDI 2: {self.Est_PDIL:.2f}')
                
                self.predicted_stats_window.update()
                
                self.prediction_made = True
                self.PDI_change_button.setVisible(True)
            self.canvas.autoscale_plot2()
                
        except:
            QMessageBox.warning(self, "Error", "There was an error. Please make sure the NN loaded is correct for the type of MWD you have selected using the dropdown menu.")
            return
        
    def load_GPC(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Load GPC")
        layout = QVBoxLayout(dialog)
        
        message_label = QLabel("Load GPC from file\n\nAlternatively, generate an example GPC curve from MWD statistics.\n", dialog)


        file_button = QPushButton("From File", dialog)
        function_button = QPushButton("From Statistics", dialog)

        file_button.setStyleSheet(themes['Func_buttons_dark'] if self.app.is_dark else themes['Func_buttons_light'])
        function_button.setStyleSheet(themes['Func_buttons_dark'] if self.app.is_dark else themes['Func_buttons_light'])

        file_button.setFixedSize(200, 40)
        function_button.setFixedSize(200, 40)

        layout.addWidget(message_label)

        button_col = QVBoxLayout()
        button_col.addWidget(file_button)
        button_col.addWidget(function_button)

        center_row = QHBoxLayout()
        center_row.addStretch(1)
        center_row.addLayout(button_col)
        center_row.addStretch(1)

        layout.addLayout(center_row)

        file_button.clicked.connect(lambda: (dialog.accept(), self.load_GPC_file()))
        function_button.clicked.connect(lambda: (dialog.accept(), self.load_GPC_func()))

        dialog.setLayout(layout)
        dialog.exec_()
        
    
    def load_GPC_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load GPC file - .dat format", "data", "All Files (*);;Text Files (*.txt)", options=options)
        if file_path:
            
            if self.GPC_loaded == False and self.prediction_made == False:
                self.clear_MWD_plot()
            if self.GPC_loaded == True:
                self.canvas.remove_single_plot("GPC Data")
                self.canvas.autoscale_plot2()
                self.GPC_loaded = False
            if self.prediction_made == True:
                self.canvas.remove_single_plot("Predicted")
            
            file_name = os.path.basename(file_path)
            try:
                if self.class_to_use == 0:
                    self.GPC_label.setText(f"Loaded GPC: {file_name}")
                    datafile = np.loadtxt(file_path, delimiter='\t', skiprows=1)
                    data = datafile[datafile[:, 0].argsort()]
                    if data[0,0] < 5:
                        try:
                            m_data = 10**data[:, 0]
                        except:
                            self.clear_MWD_plot()
                            self.canvas.autoscale_plot2()
                            QMessageBox.warning(self, "Error", "Please ensure you have selected the correct MWD class for the GPC data you are loading, \n and that the file is of the correct format.")
                            return
                    else:
                        m_data = data[:,0]
                    x_data = np.log(m_data)
                    y_data = data[:, 1]
                    y_data = y_data / np.trapz(y_data, x=x_data)
                    
                    self.m_data = m_data
                    self.y_data_GPC = y_data
                    
                    self.canvas.plot_line_on_axes2(m_data, y_data, color='blue', linetype='-', label="GPC Data", linewidth=3)
                                        
                    self.GPC_Mn = 1/np.trapz(y_data*np.exp(-x_data), x=x_data)
                    self.GPC_Mw = np.trapz(y_data*np.exp(x_data), x=x_data)
                    self.GPC_PDI = self.GPC_Mw/self.GPC_Mn
        
                    def format_e(n):
                        a = '%E' % n
                        return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
                    self.GPC_Mn_label.setText(f'GPC Mn: {format_e(Decimal(self.GPC_Mn))} (g/mol)')
                    self.GPC_Mw_label.setText(f'GPC Mw: {format_e(Decimal(self.GPC_Mw))} (g/mol)')
                    self.GPC_PDI_label.setText(f'GPC PDI: {self.GPC_PDI:.2f}')
                    
                    self.gpc_stats_window.update()
                    
                elif self.class_to_use == 1:
                    self.GPC_label.setText(f"Loaded GPC: {file_name}")
                    data = np.loadtxt(file_path, delimiter='\t')
                    Mw_data = data[0]
                    PDI_data = data[1]
                    sigma_data = np.sqrt(np.log(PDI_data))
                    mean_data = np.log(Mw_data)-(sigma_data**2)/2
                    y_data = lognormal(m, mean_data, sigma_data)
                    y_data = y_data / np.trapz(y_data, x = np.log(m))
                    
                    self.y_data_GPC = y_data
                    
                    self.canvas.plot_line_on_axes2(m, y_data, color='blue', linetype='-', label="GPC Data", linewidth=3)
                    
                    self.GPC_Mw = Mw_data
                    self.GPC_Mn = Mw_data / PDI_data
                    self.GPC_PDI = PDI_data
                    def format_e(n):
                        a = '{:.3E}'.format(n)
                        return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
                    self.GPC_Mn_label.setText(f'GPC Mn: {format_e(Decimal(self.GPC_Mn))} (g/mol)')
                    self.GPC_Mw_label.setText(f'GPC Mw: {format_e(Decimal(self.GPC_Mw))} (g/mol)')
                    self.GPC_PDI_label.setText(f'Assumed PDI: {self.GPC_PDI:.2f}')
                    
                    self.gpc_stats_window.update()
                    
                elif self.class_to_use == 2:
                    self.GPC_label.setText(f"Loaded GPC: {file_name}")
                    data = np.loadtxt(file_path, delimiter='\t')
                    
                    phi_values_data = data[:,0]
                    Mw_values_data = data[:,1]
                    PDI_values_data = data[:,2]
    
                    phiL_data,phiS_data = phi_values_data[0],phi_values_data [1]
                    MwL_data,MwS_data = Mw_values_data[0],Mw_values_data[1]
                    PDIL_data,PDIS_data = PDI_values_data[0],PDI_values_data[1]
                    
                    sigmaL_data = np.sqrt(np.log(PDIL_data))
                    sigmaS_data = np.sqrt(np.log(PDIS_data))
    
                    mean_L_data = np.log(MwL_data)-(sigmaL_data**2)/2
                    mean_S_data = np.log(MwS_data)-(sigmaS_data**2)/2
    
                    y_data = phiL_data * lognormal(m, mean_L_data, sigmaL_data) + phiS_data * lognormal(m, mean_S_data, sigmaS_data)
                    y_data = y_data / np.trapz(y_data, x = np.log(m))
                    
                    self.y_data_GPC = y_data
                    
                    self.canvas.plot_line_on_axes2(m, y_data, color='blue', linetype='-', label="GPC Data", linewidth=3)
                    
                    self.GPC_MwS = MwS_data
                    self.GPC_MwL = MwL_data
                    
                    self.GPC_PDIS = PDIS_data
                    self.GPC_PDIL = PDIL_data
                    
                    
                    def format_e(n):
                        a = '{:.3E}'.format(n)
                        return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
                    
                    self.GPC_Mn_label.setText(f'GPC Component 1 (Mw): {phiS_data*100:.2f}% {format_e(Decimal(self.GPC_MwS))} (g/mol)')
                    self.GPC_Mw_label.setText(f'GPC Component 2 (Mw): {phiL_data*100:.2f}% {format_e(Decimal(self.GPC_MwL))} (g/mol)')
                    self.GPC_PDI_label.setText(f'PDI 1: {self.GPC_PDIS:.2f}, PDI 2: {self.GPC_PDIL:.2f}')
                    
                    self.gpc_stats_window.update()
                
                if self.prediction_made:
                    if not self.cleaned_pred:
                        self.canvas.plot_line_on_axes2(m, self.pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
                        if self.class_to_use == 0:
                            for k in range(num_params2):
                                individual_pred_curve = self.prediction[0, k] * lognormal(self.z, means_Z[k], sigma_poly)
                                self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means)), linewidth=1)
                    else:
                        self.canvas.plot_line_on_axes2(m, self.clean_pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
                        if self.class_to_use == 0:
                            for k in range(num_params2):
                                individual_pred_curve = self.prediction_to_clean[k] * lognormal(self.z, means_Z[k], sigma_poly)
                                self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means)), linewidth=1)
                self.canvas.autoscale_plot2()
                self.GPC_loaded = True 
  
                
            except:
                self.clear_MWD_plot()
                self.canvas.autoscale_plot2()
                QMessageBox.warning(self, "Error", "Please ensure you have selected the correct MWD class for the GPC data you are loading.")
                return
            
        
    def load_GPC_func(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate GPC from MWD Statistics")
        layout = QVBoxLayout(dialog)
        message_label = QLabel("Generate GPC curve from MWD statistics:", dialog)
        layout.addWidget(message_label)
        
        lognormal_button = QPushButton("Log-Normal", dialog)   
        lognormal_button.setStyleSheet(themes['Func_buttons_dark'] if self.app.is_dark else themes['Func_buttons_light'])
        lognormal_button.setFixedSize(200, 40)
        lognormal_button.clicked.connect(lambda: (dialog.accept(), self.generate_GPC_lognormal()))
        layout.addWidget(lognormal_button)
        
        flory_button = QPushButton("Flory", dialog)
        flory_button.setStyleSheet(themes['Func_buttons_dark'] if self.app.is_dark else themes['Func_buttons_light'])
        flory_button.setFixedSize(200, 40)
        flory_button.clicked.connect(lambda: (dialog.accept(), self.generate_GPC_flory()))
        layout.addWidget(flory_button)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        dialog.exec_()
    
    def generate_GPC_lognormal(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Log-Normal GPC Parameters")
        layout = QFormLayout(dialog)
        message_label = QLabel("Enter the parameters for the Log-Normal GPC curve:", dialog)
        layout.addWidget(message_label)
        
        Mw_label = QLabel("Enter the Mw (g/mol):", dialog)
        self.Mw_input = QLineEdit(dialog)
        self.Mw_input.setPlaceholderText("e.g. 100000")
        
        PDI_label = QLabel("Enter the PDI:", dialog)
        self.PDI_input = QLineEdit(dialog)
        self.PDI_input.setPlaceholderText("e.g. 1.8")
        
        layout.addRow(Mw_label, self.Mw_input)
        layout.addRow(PDI_label, self.PDI_input)
        
        dialog.setLayout(layout)
        dialog.setStyleSheet(themes['dark_window'] if self.app.is_dark else themes['light_window'])
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)       
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            Mw = self.Mw_input.text()
            PDI = self.PDI_input.text()
            if not Mw or not PDI:
                QMessageBox.warning(self, "Error", "Please enter both Mw and PDI values.")
                return
            try:
                Mw = float(Mw)
                PDI = float(PDI)
                sigma = np.sqrt(np.log(PDI))
                mean = np.log(Mw) - (sigma**2) / 2
            except ValueError:
                QMessageBox.warning(self, "Error", "Please enter valid numerical values for Mw and PDI.")
                return
            y_data = lognormal(m, mean, sigma)
            y_data = y_data / np.trapz(y_data, x=np.log(m))
            self.y_data_GPC = y_data
            
            if self.GPC_loaded == False and self.prediction_made == False:
                self.clear_MWD_plot()
            if self.GPC_loaded == True:
                self.canvas.remove_single_plot("GPC Data")
                self.canvas.autoscale_plot2()
                self.GPC_loaded = False
            if self.prediction_made == True:
                self.canvas.remove_single_plot("Predicted")
                
            self.canvas.plot_line_on_axes2(m, y_data, color='blue', linetype='-', label="GPC Data", linewidth=3)
                                        
            
            self.GPC_Mw = Mw
            self.GPC_PDI = PDI
            self.GPC_Mn = Mw / PDI

            def format_e(n):
                a = '%E' % n
                return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
            self.GPC_Mn_label.setText(f'GPC Mn: {format_e(Decimal(self.GPC_Mn))} (g/mol)')
            self.GPC_Mw_label.setText(f'GPC Mw: {format_e(Decimal(self.GPC_Mw))} (g/mol)')
            self.GPC_PDI_label.setText(f'GPC PDI: {self.GPC_PDI:.2f}')
            
            if self.prediction_made:
                    if not self.cleaned_pred:
                        self.canvas.plot_line_on_axes2(m, self.pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
                        if self.class_to_use == 0:
                            for k in range(num_params2):
                                individual_pred_curve = self.prediction[0, k] * lognormal(self.z, means_Z[k], sigma_poly)
                                self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means)), linewidth=1)
                    else:
                        self.canvas.plot_line_on_axes2(m, self.clean_pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
                        if self.class_to_use == 0:
                            for k in range(num_params2):
                                individual_pred_curve = self.prediction_to_clean[k] * lognormal(self.z, means_Z[k], sigma_poly)
                                self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means)), linewidth=1)
            self.canvas.autoscale_plot2()
            self.GPC_loaded = True 
            
    def generate_GPC_flory(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Flory GPC Parameters")
        layout = QFormLayout(dialog)
        message_label = QLabel("Enter the parameters for the Flory GPC curve:", dialog)
        layout.addWidget(message_label)
        
        Mw_label = QLabel("Enter the Mw (g/mol):", dialog)
        self.Mw_input = QLineEdit(dialog)
        self.Mw_input.setPlaceholderText("e.g. 100000")
                
        layout.addRow(Mw_label, self.Mw_input)
        
        dialog.setLayout(layout)
        dialog.setStyleSheet(themes['dark_window'] if self.app.is_dark else themes['light_window'])
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)       
        layout.addWidget(buttons)

        
        if dialog.exec_() == QDialog.Accepted:
            Mw = self.Mw_input.text()
            if not Mw:
                QMessageBox.warning(self, "Error", "Please enter the Mw value.")
                return
            try:
                Mw = float(Mw)
                Mn = Mw / 2
            except ValueError:
                QMessageBox.warning(self, "Error", "Please enter valid numerical value for Mw.")
                return
            y_data = flory_schulz(m, Mn)
            y_data = y_data / np.trapz(y_data, x=np.log(m))
            self.y_data_GPC = y_data
            
            if self.GPC_loaded == False and self.prediction_made == False:
                self.clear_MWD_plot()
            if self.GPC_loaded == True:
                self.canvas.remove_single_plot("GPC Data")
                self.canvas.autoscale_plot2()
                self.GPC_loaded = False
            if self.prediction_made == True:
                self.canvas.remove_single_plot("Predicted")
                
            self.canvas.plot_line_on_axes2(m, y_data, color='blue', linetype='-', label="GPC Data", linewidth=3)
                                        
            self.GPC_Mn = Mn
            self.GPC_Mw = Mw
            self.GPC_PDI = 2

            def format_e(n):
                a = '%E' % n
                return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
            self.GPC_Mn_label.setText(f'GPC Mn: {format_e(Decimal(self.GPC_Mn))} (g/mol)')
            self.GPC_Mw_label.setText(f'GPC Mw: {format_e(Decimal(self.GPC_Mw))} (g/mol)')
            self.GPC_PDI_label.setText(f'GPC PDI: {self.GPC_PDI:.2f}')
            
            if self.prediction_made:
                    if not self.cleaned_pred:
                        self.canvas.plot_line_on_axes2(m, self.pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
                        if self.class_to_use == 0:
                            for k in range(num_params2):
                                individual_pred_curve = self.prediction[0, k] * lognormal(self.z, means_Z[k], sigma_poly)
                                self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means)), linewidth=1)
                    else:
                        self.canvas.plot_line_on_axes2(m, self.clean_pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
                        if self.class_to_use == 0:
                            for k in range(num_params2):
                                individual_pred_curve = self.prediction_to_clean[k] * lognormal(self.z, means_Z[k], sigma_poly)
                                self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means)), linewidth=1)
            self.canvas.autoscale_plot2()
            self.GPC_loaded = True             
                                       
            
    def clear_rheo_plot(self):
        self.rheo_label.setText("No rheology data loaded")
        self.Class_display_label.setText("Predicted MWD Class: N/A")
        self.rheo_data_loaded = False
        self.modes_fitted = False
        self.canvas.clear_plot1()
        self.canvas.change_axes_plot1(r'$\omega$ (rad/s)', r"G', G'' (Pa)")
        QApplication.processEvents()
        
            
    def clear_MWD_plot(self):
        self.GPC_Mn_label.setText('GPC Mn: N/A')
        self.GPC_Mw_label.setText('GPC Mw: N/A')
        self.GPC_PDI_label.setText('GPC PDI: N/A')
        
        self.Est_Mn_label.setText('Estimated Mn: N/A')
        self.Est_Mw_label.setText('Estimated Mw: N/A')
        self.Est_PDI_label.setText('Estimated PDI: N/A')  
        
        self.predicted_stats_window.update()
        self.gpc_stats_window.update()
        self.GPC_label.setText("No GPC data loaded")
        self.prediction_made = False
        self.cleaned_pred = False
        self.GPC_loaded = False
        self.tails_correct_button.setVisible(False)
        self.undo_tails_correct_button.setVisible(False)
        self.PDI_change_button.setVisible(False)
        self.canvas.clear_plot2()
        QApplication.processEvents()
    
    def save_prediction(self):
        if not self.prediction_made:
            QMessageBox.warning(self, "Error", "Please make a MWD prediction before attempting to save.")
            return 
        if self.cleaned_pred:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Select Prediction Type")
            msg_box.setText("Do you want to save the cleaned prediction or the original prediction?")
            cleaned_button = msg_box.addButton("Cleaned Prediction", QMessageBox.YesRole)
            original_button = msg_box.addButton("Original Prediction", QMessageBox.NoRole)
            msg_box.exec_()
    
            if msg_box.clickedButton() == cleaned_button:
                y_tosave = self.clean_pred_MWD
            elif msg_box.clickedButton() == original_button:
                y_tosave = self.pred_MWD  
        else:
            y_tosave = self.pred_MWD  
            
        y_tosave = np.where(y_tosave < 1e-10, 0, y_tosave)
            
    
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Prediction", "", "Text Files (*.txt);;All Files (*)", options=options)
    
        if file_path:
            try:
                try:
                    if self.class_to_use == 0 or self.class_to_use == 1:
                        if self.cleaned_pred and msg_box.clickedButton() == cleaned_button:
                            with open(file_path, 'w') as file:
                                file.write(f"Mn = {self.cleaned_Est_Mn:.2f} (g/mol),")
                                file.write(f"Mw = {self.cleaned_Est_Mw:.2f} (g/mol),")
                                file.write(f"PDI = {self.cleaned_Est_PDI:.2f},\n")
                                np.savetxt(file, np.column_stack((m, y_tosave)), delimiter='\t', fmt='%.6e')
                        else:
                            with open(file_path, 'w') as file:
                                file.write(f"Mn = {self.Est_Mn:.2f} (g/mol),")
                                file.write(f"Mw = {self.Est_Mw:.2f} (g/mol),")
                                file.write(f"PDI = {self.Est_PDI:.2f},\n")
                                np.savetxt(file, np.column_stack((m, y_tosave)), delimiter='\t', fmt='%.6e')
                    elif self.class_to_use == 2:
                        with open(file_path, 'w') as file:
                            file.write(f"Component 1 (Mw) = {self.Est_MwS:.2f} (g/mol),")
                            file.write(f" Volume Fraction 1 = {self.Est_phiS:.2f},")
                            file.write(f" PDI 1 = {self.Est_PDIS:.2f},\n")
                            file.write(f"Component 2 (Mw) = {self.Est_MwL:.2f} (g/mol),")
                            file.write(f" Volume Fraction 2 = {self.Est_phiL:.2f},")
                            file.write(f" PDI 2 = {self.Est_PDIL:.2f},\n")
                            np.savetxt(file, np.column_stack((m, y_tosave)), delimiter='\t', fmt='%.6e')
                except:
                    with open(file_path, 'w') as file:
                        np.savetxt(file, np.column_stack((m, y_tosave)), delimiter='\t', fmt='%.6e')
            except Exception:
                QMessageBox.warning(self, "Error", "Error encountered while saving file.")
    
        
    def clean_up_pred(self):
        dialog = QDialog()
        dialog.setWindowTitle("Clean MWD")
        
        if self.app.is_dark:
            dialog.setStyleSheet(themes['dark_window'])
        else:
            dialog.setStyleSheet(themes['light_window'])

        message_label = QLabel("Choose cleaning method:", dialog)
        
        method_dropdown = QComboBox(self)
        method_dropdown.addItem("Min/Max M Range")
        method_dropdown.addItem("Min/Max M Range with low tail redistribution")
        method_dropdown.addItem("Threshold")       
        
        if self.app.is_dark:
            method_dropdown.setStyleSheet(themes['dropdown_dark'])
        else:
            method_dropdown.setStyleSheet(themes['dropdown_light'])

        threshold_input = QLineEdit(dialog)
        threshold_input.setPlaceholderText("Enter threshold value")  

        threshold_label = QLabel("Threshold:", dialog)
        
        threshold_label.hide()
        threshold_input.hide()
                
        min_x_input = QLineEdit(dialog)
        min_x_input.setPlaceholderText("Enter min M value")
        min_x_label = QLabel("Min M:", dialog)
    
        max_x_input = QLineEdit(dialog)
        max_x_input.setPlaceholderText("Enter max M value")
        max_x_label = QLabel("Max M:", dialog)
        
        dump_below_label = QLabel("Redistribute below limit:", dialog)
        dump_below_input = QLineEdit(dialog)
        dump_below_input.setPlaceholderText("Enter value to redistribute below")
        
        dump_below_input.hide()
        dump_below_label.hide()
    
        def toggle_method():
            index = method_dropdown.currentText()
            if index == "Min/Max M Range":
                threshold_label.hide()
                threshold_input.hide()
                min_x_label.show()  
                min_x_input.show()
                max_x_label.show()
                max_x_input.show()
                dump_below_label.hide()
                dump_below_input.hide()
            elif index == "Min/Max M Range with low tail redistribution":
                threshold_label.hide()
                threshold_input.hide()
                min_x_label.show()
                min_x_input.show()
                max_x_label.show()
                max_x_input.show()
                dump_below_label.show()
                dump_below_input.show()
            if index == "Threshold":
                threshold_label.show()
                threshold_input.show()
                min_x_label.hide()
                min_x_input.hide()
                max_x_label.hide()
                max_x_input.hide()
                dump_below_label.hide()
                dump_below_input.hide()
            
            dialog.adjustSize()
            QApplication.processEvents()
                
        method_dropdown.currentIndexChanged.connect(toggle_method) 
    
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        def show_clean_help(dialog):
            QMessageBox.information(dialog, "Help", 
                                "Select one of the cleaning methods from the dropdown menu.\n"
                                "Min/Max M Range is as described, where values below the min and above the max will be cut smoothly.\n"
                                "Min/Max M Range dynamic will redistribute the removed values from the lower tail proportionally within the range specified.\n"
                                "Threshold will remove values below the specified threshold.")
        
        help_button = buttons.addButton("Help", QDialogButtonBox.ActionRole)
        help_button.clicked.connect(lambda: show_clean_help(dialog))


        layout = QVBoxLayout()
        layout.addWidget(message_label)
        layout.addWidget(method_dropdown)
        layout.addWidget(min_x_label)
        layout.addWidget(min_x_input)
        layout.addWidget(dump_below_label)
        layout.addWidget(dump_below_input)
        layout.addWidget(max_x_label)
        layout.addWidget(max_x_input)
        layout.addWidget(threshold_label)
        layout.addWidget(threshold_input)

        layout.addWidget(buttons)
        dialog.setLayout(layout)

        
        if dialog.exec_() == QDialog.Accepted:
            min_x, max_x = None, None
            if method_dropdown.currentText() == "Min/Max M Range":
                try:
                    min_x = float(min_x_input.text())
                except:
                    min_x = None
                try:
                    max_x = float(max_x_input.text())
                except:
                    max_x = None
                if min_x is None and max_x is None:
                    QMessageBox.warning(None, "Invalid Input", "Please enter valid min/max M values.")
                    return
            elif method_dropdown.currentText() == "Min/Max M Range with low tail redistribution":
                try:
                    min_x = float(min_x_input.text())
                except:
                    min_x = None
                try:
                    max_x = float(max_x_input.text())
                except:
                    max_x = None
                if min_x is None:
                    QMessageBox.warning(None, "Invalid Input", "Please enter valid min/max M values.")
                    return
                if min_x is None and max_x is None:
                    QMessageBox.warning(None, "Invalid Input", "Please enter valid min/max M values.")
                    return
                try:
                    dump_below = float(dump_below_input.text())
                except:
                    QMessageBox.warning(None, "Invalid Input", "Please enter a valid value to redistribute below.")
                    return
                if dump_below <= min_x:
                    QMessageBox.warning(None, "Invalid Input", "Redistribution value must be greater than the min M value.")
                    return
            threshold = float(threshold_input.text()) if method_dropdown.currentText() == "Threshold" else 0
            
        else:
            return
        
        sum_cleaned = 0
        
        
        
        self.prediction_to_clean = self.prediction[0].copy()
        
        if min_x is not None:
            for n in range(len(self.prediction_to_clean)):
                if np.exp(means_Z[n]+ sigma_poly**2/2)*self.M_e < min_x:
                    sum_cleaned += self.prediction_to_clean[n]
                    self.prediction_to_clean[n] = 0
        if max_x is not None:
            for n in range(len(self.prediction_to_clean)):            
                if np.exp(means_Z[n]+ sigma_poly**2/2)*self.M_e > max_x:
                    self.prediction_to_clean[n] = 0
        
        if method_dropdown.currentText() == "Min/Max M Range with low tail redistribution":
            dump_split = 0
            values_where_dump = np.zeros(len(self.prediction_to_clean))
            for n in range(len(self.prediction_to_clean)):
                if np.exp(means_Z[n]+ sigma_poly**2/2)*self.M_e > min_x:
                    if np.exp(means_Z[n]+ sigma_poly**2/2)*self.M_e < dump_below:
                        dump_split += 1
                        values_where_dump[n] = self.prediction_to_clean[n]
            
            fractions_to_dump = values_where_dump / sum(values_where_dump)

            to_dump = fractions_to_dump * sum_cleaned
            self.prediction_to_clean += to_dump
        
        if threshold < max(self.pred_MWD) and threshold != 0:
            self.prediction_to_clean[1.5*self.prediction_to_clean < threshold] = 0
        
        if self.univ_space == False:
            self.clean_pred_MWD = sum_of_lognormals(m, self.prediction_to_clean)
        
        if self.univ_space == True:
            self.clean_pred_MWD = sum_of_lognormals_Z(self.z, self.prediction_to_clean)
        
        
        if self.univ_space == False:
            self.clean_pred_MWD = self.clean_pred_MWD / np.trapz(self.clean_pred_MWD, x = x)

        if self.univ_space == True:
            self.clean_pred_MWD = self.clean_pred_MWD / np.trapz(self.clean_pred_MWD, x = np.log(self.z))

        self.canvas.remove_single_plot("Predicted")
            
        self.canvas.plot_line_on_axes2(m, self.clean_pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
        
        if self.univ_space == False:
            for k in range(num_params):
                individual_pred_curve = self.prediction_to_clean[k] * lognormal(m, means[k], sigma_poly)
                self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means)), linewidth=1)
        if self.univ_space == True:
            for k in range(num_params2):
                individual_pred_curve = self.prediction_to_clean[k] * lognormal(self.z, means_Z[k], sigma_poly)
                self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means_Z)), linewidth=1)

        
        self.canvas.autoscale_plot2()
        
        self.cleaned_Est_Mn = 1/np.trapz(self.clean_pred_MWD*np.exp(-x), x=x)
        self.cleaned_Est_Mw = np.trapz(self.clean_pred_MWD*np.exp(x), x=x)
        self.cleaned_Est_PDI = self.cleaned_Est_Mw/self.cleaned_Est_Mn
        
        def format_e(n):
            a = '%E' % n
            return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
        self.Est_Mn_label.setText(f'Estimated Mn: {format_e(Decimal(self.cleaned_Est_Mn))} (g/mol)')
        self.Est_Mw_label.setText(f'Estimated Mw: {format_e(Decimal(self.cleaned_Est_Mw))} (g/mol)')
        self.Est_PDI_label.setText(f'Estimated PDI: {self.cleaned_Est_PDI:.2f}') 
        
        self.predicted_stats_window.update()
        
        self.undo_tails_correct_button.setVisible(True)
        self.cleaned_pred = True
        
    def revert_clean_pred(self):
        if not self.cleaned_pred:
            QMessageBox.warning(self, "Error", "Cannot undo clean until the prediction has been cleaned.")
            return
        self.canvas.remove_single_plot("Predicted")
                
        self.canvas.plot_line_on_axes2(m, self.pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
        
        if self.class_to_use == 0:
            if self.univ_space == False:
                for k in range(num_params):
                    individual_pred_curve = self.prediction[0, k] * lognormal(m, means[k], sigma_poly)
                    self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means)), linewidth=1)
            if self.univ_space == True:
                for k in range(num_params2):
                    individual_pred_curve = self.prediction[0, k] * lognormal(self.z, means_Z[k], sigma_poly)
                    self.canvas.plot_line_on_axes2(m, individual_pred_curve, label=None, linetype='--', color=plt.cm.viridis(k / len(means)), linewidth=1)
        
        self.canvas.autoscale_plot2()
        
        def format_e(n):
            a = '%E' % n
            return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
        self.Est_Mn_label.setText(f'Estimated Mn: {format_e(Decimal(self.Est_Mn))} (g/mol)')
        self.Est_Mw_label.setText(f'Estimated Mw: {format_e(Decimal(self.Est_Mw))} (g/mol)')
        self.Est_PDI_label.setText(f'Estimated PDI: {self.Est_PDI:.2f}')  
        
        self.predicted_stats_window.update()
        
        self.cleaned_pred = False
        self.undo_tails_correct_button.setVisible(False)
        
    def PDI_change(self): 
        dialog = QDialog()
        dialog.setWindowTitle("Set PDI")

        message_label = QLabel("Set PDI for predicted monodisperse/binary peaks", dialog) 

        if self.class_to_use == 1:
            
            PDI_input = QLineEdit(dialog)
            PDI_input.setPlaceholderText("Default = 1.03")  
    
            input_label = QLabel("PDI:", dialog)
    
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject) 
    
            layout = QVBoxLayout()
            layout.addWidget(message_label)
            layout.addWidget(input_label)
            layout.addWidget(PDI_input)    
            layout.addWidget(buttons)
            dialog.setLayout(layout)
            
            if dialog.exec_() == QDialog.Accepted:
                try:
                    PDI = float(PDI_input.text()) 
                except:
                    PDI = 1.03
            else:
                return
            
            self.canvas.clear_plot2()
            
            sigma_mono = np.sqrt(np.log(PDI))
            Z_pred = 10**float(self.prediction[0])
            mean_Z_pred = np.log(Z_pred)-(sigma_mono**2)/2
            
            pred_MWD = lognormal(self.z, mean_Z_pred, sigma_mono)
            pred_MWD = pred_MWD / simps(pred_MWD, x = np.log(self.z))
            self.pred_MWD = pred_MWD
            
            if self.GPC_loaded:
                self.canvas.plot_line_on_axes2(m, self.y_data_GPC, color='blue', linetype='-', label="GPC Data", linewidth=3)
                
            self.canvas.plot_line_on_axes2(m, pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
            
            self.Est_Mw = Z_pred * self.M_e
            self.Est_Mn = self.Est_Mw / PDI
            self.Est_PDI = PDI
            def format_e(n):
                a = '{:.3E}'.format(n)
                return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
            self.Est_Mn_label.setText(f'Estimated Mn: {format_e(Decimal(self.Est_Mn))} (g/mol)')
            self.Est_Mw_label.setText(f'Estimated Mw: {format_e(Decimal(self.Est_Mw))} (g/mol)')
            self.Est_PDI_label.setText(f'Assumed PDI: {self.Est_PDI:.2f}')
            
            self.predicted_stats_window.update() 
        
        if self.class_to_use == 2:
            
            PDI1_input = QLineEdit(dialog)
            PDI1_input.setPlaceholderText("Default = 1.03")  
            
            PDI2_input = QLineEdit(dialog)
            PDI2_input.setPlaceholderText("Default = 1.03")
    
            input1_label = QLabel("PDI Short Component:", dialog)
            input2_label = QLabel("PDI Long Component:", dialog)
    
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
    
            layout = QVBoxLayout()
            layout.addWidget(message_label)
            layout.addWidget(input1_label)
            layout.addWidget(PDI1_input)  
            layout.addWidget(input2_label)
            layout.addWidget(PDI2_input)  
            layout.addWidget(buttons)
            dialog.setLayout(layout)
    
            if dialog.exec_() == QDialog.Accepted:
                try:
                    PDIS = float(PDI1_input.text())  
                except:
                    PDIS = 1.03
                try:
                    PDIL = float(PDI2_input.text())  
                except:
                    PDIL = 1.03
            else:
                PDIS = 1.03
                PDIL = 1.03
            
            self.canvas.clear_plot2()
            
            sigmaS = np.sqrt(np.log(PDIS))
            sigmaL = np.sqrt(np.log(PDIL))
            
            ZS_pred = 10**float(self.prediction[0][0])
            ZL_pred = 10**float(self.prediction[0][1])
            
            self.phiL_pred = float(self.prediction[0][2])
            self.phiS_pred = 1 - self.phiL_pred 
            
            mean_ZS_pred = np.log(ZS_pred)-(sigmaS**2)/2
            mean_ZL_pred = np.log(ZL_pred)-(sigmaL**2)/2
            
            pred_MWD = self.phiL_pred * lognormal(self.z, mean_ZL_pred, sigmaL) + self.phiS_pred * lognormal(self.z, mean_ZS_pred, sigmaS)
            pred_MWD = pred_MWD / np.trapz(pred_MWD, x = np.log(self.z))
            self.pred_MWD = pred_MWD
                
            if self.GPC_loaded:
                self.canvas.plot_line_on_axes2(m, self.y_data_GPC, color='blue', linetype='-', label="GPC Data", linewidth=3)           
            
            self.canvas.plot_line_on_axes2(m, pred_MWD, linetype='--', color='orange', label="Predicted", linewidth=5)
            
            self.Est_MwS = ZS_pred * self.M_e
            
            self.Est_MwL = ZL_pred * self.M_e
            
            self.Est_PDIL = PDIL
            self.Est_PDIS = PDIS
            self.Est_phiL = self.phiL_pred
            self.Est_phiS = self.phiS_pred
            
            def format_e(n):
                a = '{:.3E}'.format(n)
                return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
            self.Est_Mn_label.setText(f'Component 1 (Mw): {self.phiS_pred*100:.2f}%   {format_e(Decimal(self.Est_MwS))} (g/mol)')
            self.Est_Mw_label.setText(f'Component 2 (Mw): {self.phiL_pred*100:.2f}%   {format_e(Decimal(self.Est_MwL))} (g/mol)')
            self.Est_PDI_label.setText(f'PDI 1: {self.Est_PDIS:.2f}, PDI 2: {self.Est_PDIL:.2f}')
            
            self.predicted_stats_window.update()
            
    def save_figure(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Figure")
        layout = QVBoxLayout(dialog)

        rheo_button = QPushButton("Save Rheology Figure", dialog)
        mwd_button = QPushButton("Save MWD Figure", dialog)

        rheo_button.setStyleSheet(themes['Save_button_dark'] if self.app.is_dark else themes['Save_button_light'])
        mwd_button.setStyleSheet(themes['Save_button_dark'] if self.app.is_dark else themes['Save_button_light'])

        rheo_button.setFixedSize(200, 40)
        mwd_button.setFixedSize(200, 40)

        layout.addWidget(rheo_button)
        layout.addWidget(mwd_button)

        rheo_button.clicked.connect(lambda: (dialog.accept(), self.save_rheo_fig()))
        mwd_button.clicked.connect(lambda: (dialog.accept(), self.save_MWD_fig()))

        dialog.setLayout(layout)
        dialog.exec_()
            
    def save_rheo_fig(self):
        if not self.rheo_data_loaded:
            QMessageBox.warning(self, "Error", "Cannot save empty figure.")
            return
        self.canvas.save_ax_figure(self.canvas.ax1)
    
    def save_MWD_fig(self):
        if not self.prediction_made and not self.GPC_loaded:
            QMessageBox.warning(self, "Error", "Cannot save empty figure.")
            return
        self.canvas.save_ax_figure(self.canvas.ax2)
        
    def change_frequency_range(self):
        if not self.rheo_data_loaded:
            QMessageBox.warning(self, "Error", "Cannot change frequency range without loaded rheology data.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Change Frequency Range")
        layout = QVBoxLayout(dialog)
        message_label = QLabel("Enter new frequency range (in rad/s).\n\nYou will then need to fit Maxwell modes to the new data.\n\nYou can leave one of the inputs blank to have no effect on the respective limit.\n\nTo undo a previous frequency change, leave both inputs blank.\n", dialog)
        min_freq_input = QLineEdit(dialog)
        min_freq_input.setPlaceholderText("Min Frequency (rad/s)")
        max_freq_input = QLineEdit(dialog)
        max_freq_input.setPlaceholderText("Max Frequency (rad/s)")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(message_label)
        layout.addWidget(min_freq_input)
        layout.addWidget(max_freq_input)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        if dialog.exec_() == QDialog.Accepted:
            try:
                min_freq_text = min_freq_input.text().strip()
                max_freq_text = max_freq_input.text().strip()
                if self.univ_space:
                    min_freq = float(min_freq_text) if min_freq_text else np.min(self.original_w_values_Exp) * self.tau_e
                    max_freq = float(max_freq_text) if max_freq_text else np.max(self.original_w_values_Exp) * self.tau_e
                else:
                    min_freq = float(min_freq_text) if min_freq_text else np.min(self.original_w_values_Exp)
                    max_freq = float(max_freq_text) if max_freq_text else np.max(self.original_w_values_Exp)
                if min_freq >= max_freq:
                    raise ValueError("Min frequency must be less than max frequency.")
            except ValueError as e:
                QMessageBox.warning(self, "Error", f"Invalid input: {e}")
                return
            
            self.clear_rheo_plot()
            if self.univ_space:
                self.w_values_Exp = self.original_w_values_Exp[(self.original_w_values_Exp >= min_freq/self.tau_e) & (self.original_w_values_Exp <= max_freq/self.tau_e)] * self.tau_e
                self.Gp_data_Exp = self.original_Gp_data_Exp[(self.original_w_values_Exp >= min_freq/self.tau_e) & (self.original_w_values_Exp <= max_freq/self.tau_e)] / self.G0
                self.Gpp_data_Exp = self.original_Gpp_data_Exp[(self.original_w_values_Exp >= min_freq/self.tau_e) & (self.original_w_values_Exp <= max_freq/self.tau_e)] / self.G0
            else:
                self.w_values_Exp = self.original_w_values_Exp[(self.original_w_values_Exp >= min_freq) & (self.original_w_values_Exp <= max_freq)]
                self.Gp_data_Exp = self.original_Gp_data_Exp[(self.original_w_values_Exp >= min_freq) & (self.original_w_values_Exp <= max_freq)]
                self.Gpp_data_Exp = self.original_Gpp_data_Exp[(self.original_w_values_Exp >= min_freq) & (self.original_w_values_Exp <= max_freq)]     
            self.canvas.clear_plot1()
            if self.univ_space:
                self.canvas.change_axes_plot1(r'$\omega$ * $\tau_e$', r"$G' / G_N^0$, $G'' / G_N^0$")
            else:
                self.canvas.change_axes_plot1(r'$\omega$ (rad/s)', r"$G'$, $G''$")
        
            self.canvas.plot_scatter_on_axes1(self.w_values_Exp, self.Gp_data_Exp, facecolors='none', edgecolors='black', linewidth = 1, label = "G' Input Univ", size = 20)
            self.canvas.plot_scatter_on_axes1(self.w_values_Exp, self.Gpp_data_Exp, facecolors='none', edgecolors='red', linewidth = 1, label = "G'' Input Univ", size = 20)
            self.canvas.autoscale_plot1()
            self.modes_fitted = False
            self.rheo_data_loaded = True
        else:
            return
            
            
    def closeEvent(self, event):
        for window in [self.predicted_stats_window, self.gpc_stats_window]:
            if window is not None:
                window.close()  
        event.accept()