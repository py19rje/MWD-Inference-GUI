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

themes = {
    "Func_buttons_light": """
        QPushButton {
            background-color: rgb(217,217,217);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: left;           /* Align text to the left */
            padding-left: 10px;
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(150, 150, 230);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "Func_buttons_dark": """
        QPushButton {
            background-color: rgb(90,90,90);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: left;           /* Align text to the left */
            padding-left: 10px;
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(150, 150, 230);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "Clear_button_light": """
        QPushButton {
            background-color: rgb(217,217,217);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(165, 94, 94);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "Clear_button_dark": """
        QPushButton {
            background-color: rgb(90,90,90);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(165, 94, 94);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "Save_button_light": """
        QPushButton {
            background-color: rgb(217,217,217);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(94, 165, 94);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "Save_button_dark": """
        QPushButton {
            background-color: rgb(90,90,90);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(94, 165, 94);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "PDI_change_button_light": """
        QPushButton {
            background-color: rgb(217,217,217);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(94, 94, 165);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "PDI_change_button_dark": """
        QPushButton {
            background-color: rgb(90,90,90);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(94, 94, 165);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "stats_button_light": """
        QPushButton {
            background-color: rgb(217,217,217);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(94, 165, 94);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "stats_button_dark": """
        QPushButton {
            background-color: rgb(90,90,90);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(94, 165, 94);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "help_button_light": """
        QPushButton {
            background-color: rgb(217,217,217);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(250, 198, 27);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "help_button_dark": """
        QPushButton {
            background-color: rgb(90,90,90);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: rgb(250, 198, 27);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: #e0e0e0;  /* Darker gray when clicked */
        }
    """,
    "light_window": """
    QWidget {
        background-color: rgb(250,250,250);
        color: black;
        }
    """,
    "dark_window": """
    QWidget {
        background-color: rgb(30,38,52); 
        color: white;
        }
    """,
    "dropdown_light": """
    QComboBox QAbstractItemView {
        background: white; /* Background of dropdown */
        color: black; /* Text color */
        selection-background-color: #d2e7f7; /* Highlight color */
        selection-color: #5d5d5d; /* Text color when selected */
    }    
    QComboBox QAbstractItemView::item:hover {
        background: #d2e7f7; /* Hover background */
        color: #5d5d5d; /* Hover text color */
    }
    """,
    "dropdown_dark": """
    QComboBox QAbstractItemView {
        background: rgb(90,90,90); /* Background of dropdown */
        color: black; /* Text color */
        selection-background-color: #1e69f4; /* Highlight color */
        selection-color: white; /* Text color when selected */
    }    
    QComboBox QAbstractItemView::item:hover {
        background: #1e69f4; /* Hover background */
        color: white; /* Hover text color */
    }
    """,
    "Zoom_button_light": """
        QPushButton {
            background-color: rgb(217,217,217);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            vertical-align: middle; /* Center vertically */
        }
        
        QPushButton:hover {
            background-color: rgb(150, 150, 230);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: rgb(170, 170, 250);  /* Darker gray when clicked */
        }
    """,
    "Zoom_button_dark": """
        QPushButton {
            background-color: rgb(90,90,90);    /* White background */
            border-radius: 10px;        /* Curved corners */
            text-align: center;           /* Align text to the left */
            vertical-align: middle; /* Center vertically */
        }
        
        QPushButton:hover {
            background-color: rgb(150, 150, 230);  /* Slight gray on hover */
        }
        
        QPushButton:pressed {
            background-color: rgb(170, 170, 250);  /* Darker gray when clicked */
        }
    """
}