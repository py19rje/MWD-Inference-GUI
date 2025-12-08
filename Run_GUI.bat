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

