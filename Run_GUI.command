#!/bin/bash
# Print a message
echo "Launching..."
echo "This may take a few minutes"

# Activate Conda Environment
source /opt/anaconda3/bin/activate /opt/anaconda3/envs/MWD_Inference_ENV

# Run Python script
cd /Users/simon/Documents/RobTesting2/GUI_test/
python Inference_GUI.py &

if [ $? -ne 0 ]; then
    echo "An error occurred. Press any key to exit."
    read -n 1
else
    echo "Script executed successfully. Closing terminal."
    osascript -e 'tell application "Terminal" to close (every window whose name contains "Run_GUI.command")' & exit 0
fi