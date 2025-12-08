import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox, QPushButton, QHBoxLayout, QApplication)
from PyQt5.QtGui import QFont
from markdown import markdown
from modules.themes import themes

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.app = QApplication.instance()
        
        self.setWindowTitle('Workflow Help')
        screen_width, screen_height = self.screen().size().width(), self.screen().size().height()
        self.setMinimumWidth(int(0.5 * screen_width))
        self.setMinimumHeight(int(0.5 * screen_height))

        layout = QVBoxLayout(self)

        self.help_text = QTextEdit(self)
        self.help_text.setReadOnly(True) 

        readme_path = "README.md"  
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.readlines()

            processed_content = self.preprocess_readme(readme_content)

            html_content = markdown(processed_content)

            font = QFont()
            prop_font = int(0.012 * screen_height) if int(0.012 * screen_height) >= 12 else 12
            font.setPointSize(prop_font)
            font.setFamily("Arial")
            self.help_text.setFont(font)
            self.help_text.setHtml(html_content)
        else:
            self.help_text_edit.setPlainText("README.md file not found.")

        layout.addWidget(self.help_text)
        
        self.zoom_out_button = QPushButton(self)
        self.zoom_out_button.setText("-")  
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_in_button = QPushButton(self)
        self.zoom_in_button.setText("+")
        self.zoom_in_button.clicked.connect(self.zoom_in)  
        
        monospace_plus_font = QFont("Consolas")
        monospace_plus_font.setPointSize(18)
        monospace_minus_font = QFont("Consolas")  
        monospace_minus_font.setPointSize(26)  

        for button in [self.zoom_out_button, self.zoom_in_button]:
            button.setFixedHeight(35)
            button.setFixedWidth(100)
            if self.app.is_dark:
                button.setStyleSheet(themes['Zoom_button_dark'])
            else:
                button.setStyleSheet(themes['Zoom_button_light'])
        
        self.zoom_out_button.setFont(monospace_minus_font)
        self.zoom_in_button.setFont(monospace_plus_font)

        ZoomLayout = QHBoxLayout()
        ZoomLayout.addStretch(1)
        ZoomLayout.addWidget(self.zoom_out_button)
        ZoomLayout.addSpacing(30)
        ZoomLayout.addWidget(self.zoom_in_button)
        ZoomLayout.addStretch(1)
        layout.addLayout(ZoomLayout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok, self)
        self.button_box.accepted.connect(self.accept)
        layout.addWidget(self.button_box)
    
    def zoom_in(self):
        current_font = self.help_text.font()
        current_size = current_font.pointSize()
        new_size = current_size + 1
        current_font.setPointSize(new_size)
        self.help_text.setFont(current_font)
    
    def zoom_out(self):
        current_font = self.help_text.font()
        current_size = current_font.pointSize()
        new_size = max(current_size - 1, 8)  
        current_font.setPointSize(new_size)
        self.help_text.setFont(current_font)

    def preprocess_readme(self, lines):
        processed_lines = []

        processed_lines.append("## Workflow Quick-Guide\n\n")
        processed_lines.append("This is a quick guide to help load rheology and make a prediction.\nThe steps are as follows:\n\n")
        processed_lines.append("1. (Required) Load the rheology data using the first of the function buttons on the left of the screen. You can adjust the frequency range of the rheology, and cut the highest/lowest frequency data points using the relevant button on the top row. This may help with prediction if e.g. the lowest-frequency values may not represent the sample's response, but instead equipment limitations.\n\n")
        processed_lines.append("2. (Required) Convert the rheology to Universal space using the 'Universal Space' function button. This will open a dialog box where you are prompted to enter the material type and temperature of measurement.\n\n")
        processed_lines.append("3. (Required) Fit a relaxation spectrum to the rheology data using the 'Fit Maxwell Modes' function button. This will open a dialog box where you will select the relaxation spectrum timescale range. Unless you are using your own custom NN models, leave the default values and press OK.\n\n")
        processed_lines.append("4. (Optional) Classify the MWD using the 'Classify MWD' function button. This will automatically use all classifier models available to determine whether the polymer is polydisperse, monodisperse, or a binary blend of monodisperse components from the melt rheology.\n\n")
        processed_lines.append("5. (Optional) Load GPC data to compare with the prediction made. This can be done before or after the prediction is made. You may have a GPC MWD file, which can be loaded from a text file. If you only have some MWD statistics (e.g. Mw and PDI), you can load a sample distribution using some pre-defined functions e.g. log-Gaussian.\n\n")
        processed_lines.append("6. (Required) Select a NN model to make prediction. Make sure the correct class of MWD is selected in the dropdown 'Select MWD Class for Prediction'. Then press 'Select NN Model', and a dialog will appear with a dropdown of the available models. Select one and press 'ok'. This will load the model choice into the software and make it ready for prediction. Note that different models will produce slightly different results as NN training is not deterministic. Try making multiple predictions to see how similar they are!\n\n")
        processed_lines.append("7. (Required) Make a prediction using the 'Make Prediction' function button. This will display the prediction on the 'MWD' plot.\n\n")
        processed_lines.append("8. (Optional) Perform a tails correction using the 'MWD Tails Correction' function button, which will appear after the prediction is made. This will open a dialog box where you can select the tails correction method to use and the relevant parameters. This dialog has a separate help icon if this is unclear. This can be undone using the button that will appear after the correction is made.\n\n")
        processed_lines.append("9. (Optional) Save the prediction using the 'Save Prediction' function button. This will open a file save dialog where you can select the location and name of the file to save. You can also choose whether you would like to save the tails-corrected or original prediction.\n\n")
        processed_lines.append("10. (Optional) Save figures. You can save either the rheology or MWD figure by pressing the save button at the top of the screen, selecting the figure you would like to save, and choosing the file name and save destination.\n\n")
        processed_lines.append("11. (Optional) Dynamic plots. At any point, you can change the visibility of the items on the two plots with the 'Dynamic Plots' dropdown. This can be used to view certain data more easily.\n\n")
        processed_lines.append("\n\nFor a more detailed guide to the software, please refer to the more complete user manual below.\n\n")
        
        include = False
        for line in lines:
            if "## User Manual" in line:  
                include = True

            if include:
                processed_lines.append(line)        

        return "".join(processed_lines)
