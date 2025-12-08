from PyQt5.QtWidgets import QApplication, QComboBox, QListWidget, QListWidgetItem, QCheckBox, QMessageBox

class CheckableComboBox(QComboBox):
    def __init__(self, title="Dynamic Plots", parent=None):
        super().__init__(parent)
        self.title = title
        self.setEditable(True)  
        self.lineEdit().setReadOnly(True) 
        self.setEditText(title) 
        self.setFixedWidth(150)
        self.screen_width, self.screen_height = self.screen().size().width(), self.screen().size().height()
        self.checkbox_size = int(self.screen_height * 0.015)  

        self.list_widget = QListWidget()
        self.setModel(self.list_widget.model())
        self.setView(self.list_widget)
        self.app = QApplication.instance()
        self.checkbox_dict = {}  

    def add_checkbox(self, label, callback):
        if label in self.checkbox_dict:
            return  

        item = QListWidgetItem(self.list_widget)
        checkbox = QCheckBox(label)
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(lambda state: callback(label, state))
        self.set_styles()
            
        self.list_widget.setItemWidget(item, checkbox)

        self.checkbox_dict[label] = checkbox  

    def remove_checkbox(self, label):
        if label in self.checkbox_dict:
            checkbox = self.checkbox_dict.pop(label)
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if self.list_widget.itemWidget(item) == checkbox:
                    self.list_widget.takeItem(i)
                    break
            checkbox.deleteLater()
            
    def set_styles(self):
        self.app = QApplication.instance()
        if self.app.is_dark:
            style = f"""
                QCheckBox::indicator:unchecked {{
                    background-color: white;
                    border: 1px solid gray;
                    border-radius: {self.checkbox_size // 4}px;  /* Rounded corners for the checkbox */
                    width: {self.checkbox_size}px;
                    height: {self.checkbox_size}px;
                }}
                QCheckBox::indicator:checked {{
                    background-color: green;
                    border: 1px solid gray;
                    border-radius: {self.checkbox_size // 4}px;  
                    width: {self.checkbox_size}px;
                    height: {self.checkbox_size}px;
                }}
                QCheckBox {{
                    padding-left: 4px;
                    color: white;  /* Label color for dark mode */
                }}
            """
        else:
            style = f"""
                QCheckBox::indicator:unchecked {{
                    background-color: white;
                    border: 1px solid gray;
                    border-radius: {self.checkbox_size // 4}px;  
                    width: {self.checkbox_size}px;
                    height: {self.checkbox_size}px;
                }}
                QCheckBox::indicator:checked {{
                    background-color: green;
                    border: 1px solid gray;
                    border-radius: {self.checkbox_size // 4}px;  
                    width: {self.checkbox_size}px;
                    height: {self.checkbox_size}px;
                }}
                QCheckBox {{
                    padding-left: 4px;
                    color: black;  /* Label color for light mode */
                }}
            """
            
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            checkbox = self.list_widget.itemWidget(item)
            if checkbox:  
                checkbox.setStyleSheet(style)
            
    def showPopup(self):
        if not self.checkbox_dict:  
            QMessageBox.warning(self, "Warning", "No plots have been added yet!\nYou can control plots here after they have been added to either of the figures.")
        else:
            self.set_styles()
            super().showPopup()
    
    def paintEvent(self, event):
        self.lineEdit().setText(self.title)
        super().paintEvent(event)