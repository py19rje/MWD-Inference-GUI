::  MWD-Inference-GUI
::  --------------------------------------------------------------------------------------------------------

::  Authors:
::      Robert J. Elliott, py19rje@leeds.ac.uk
::      Daniel J. Read, d.j.read@leeds.ac.uk
::      Luisa Cutillo, l.cutillo@leeds.ac.uk
::      Johan Mattsson, k.j.l.mattsson@leeds.ac.uk

::  GitHub:
::      https://github.com/py19rje/MWD-Inference-GUI

::  --------------------------------------------------------------------------------------------------------

::  Copyright (2026): Robert J. Elliott, University of Leeds

::  This file is part of the software MWD-Inference-GUI

::  MWD-Inference-GUI is free software: you can redistribute it and/or modify
::  it under the terms of the GNU General Public License as published by
::  the Free Software Foundation, either version 3 of the License, or
::  (at your option) any later version.

::  MWD-Inference-GUI is distributed in the hope that it will be useful,
::  but WITHOUT ANY WARRANTY; without even the implied warranty of
::  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
::  GNU General Public License for more details.

::  You should have received a copy of the GNU General Public License
::  along with this software.  If not, see <http://www.gnu.org/licenses/>.

::  This work forms part of the research programme of DPI, project \#861.


::  --------------------------------------------------------------------------------------------------------

@echo off
REM Print a message
echo Launching...
echo This may take a few minutes

REM Activate Conda Environment
call C:\Users\py19rje\AppData\Local\anaconda3\Scripts\activate.bat C:\Users\py19rje\AppData\Local\anaconda3\envs\MWD_Inference_ENV

REM Run Python script
cd C:\Users\py19rje\Documents\GUI_test\
start "" pythonw Inference_GUI.py
REM python Inference_GUI.py

REM Keep window open if there is an error
IF %ERRORLEVEL% NEQ 0 PAUSE

REM timeout /t 4 /nobreak >nul

exit

