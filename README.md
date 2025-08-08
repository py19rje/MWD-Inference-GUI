# GUI for MWD Inference

## Overview

This repository contains code for a Python project that includes a graphical user interface (GUI) for MWD inference tasks. Follow the instructions below to set up your environment and run the `Inference_GUI.py` file.

## Prerequisites

1. **[Anaconda](https://www.anaconda.com/products/distribution):** Ensure that Anaconda is installed. If not, download and install it from the Anaconda website. For mac users, after installation you should be able to use conda commands in the "terminal". For Windows users, your Anaconda installation will install an "Anaconda Prompt". This should be used for all the commands detailed below.

## Installation Instructions

### 1. Install Git (if needed)

If Git is not installed, you can install it using conda or pip:

```bash
conda install git
```
```bash
pip install git
```

### 2. Clone the Repository

You can either download the repository as a ZIP file from GitHub or clone it using Git. To clone the repository, follow these steps:

1. Open Anaconda Prompt (Windows) or terminal window (Linux or MacOS).
2. Navigate to desired parent directory of repository, replacing /parent/directory/ with the location you want e.g. Documents:
```bash
cd /parent/directory/
```  
3. Run the following command to clone the repository:
```bash
git clone https://github.com/py19rje/Inference_GUI_test.git
```

At this stage, you may be asked to enter a username and password if the repository is private. If this is the case, please follow advice you should have received on how to do this (it is not as simple as just entering your GitHub username and password). This will likely involve downloading git credential manager.

### 3. Navigate to the cloned directory 
Run the following code in the Anaconda Prompt or Terminal to make the working directory correct (ensure you the directory name matches the new directory):
```bash
cd GUI_test
```

### 4. Create and activate a conda environment:
Running the code below will create a new conda environment with a specified version of python and the required packages. This may take some time. The name can be changed freely by editing the first line of the .yml file, but ensure that it is likewise changed for the activation step.
```bash
conda env create -f GUI_env.yml
```
Run the line below to activate the conda environment you have just created. This is using the default name of the environment.
```bash
conda activate MWD_Inference_ENV
```

### 5. Run the user interface
#### a). From the anaconda prompt
If there have been no issues with the previous steps, you should be ready to launch the user interface. To do so, run the following command, making sure that you are still in the directory of the repository, and that the python environment is activated:
```bash
python Inference_GUI.py
```
#### b). From runfile (requires setup)
For future use, to avoid having to open the terminal and activate the conda environment every time you want to run the GUI, there is a way of loading the GUI automatically without the command line by running an executable file. 

Windows:

For windows users, the run file is named Run_Gui.bat and can be found in the main directory. Open this file in a text editor e.g. Notepad. 

Find the line "call C:\path\to\anaconda\...". This line contains two filepaths. This needs to be edited so that the first filepath is the path to your local copy of the anaconda Scripts/activate.bat file. This will typically be found in a hidden folder called AppData. To see hidden folders on windows make sure view->show->hidden items is checked. The second filepath should be changed to the loaction of the conda environement you have made (MWD_Inference_ENV). This will also be found in the anaconda directoies in AppData. If you cannot find the location of your conda environment, enter this line into the anaconda prompt:

```bash
conda info --envs
```

This should return all your conda environments and their locations on your machine. Copy the path to the MWD_Inference_ENV environment to replace the second filepath on the line.

Once you have updated these lines in the .bat file, you can simply run it by double clicking etc.. This should activate the conda environment for you and load the GUI, and hence not require manual environment activation for every use. It is also possible to create a shortcut to this .bat file on your homescreen and set an icon (recommend the .ico file /graphics/NN.ico). 

macOS:

For macOS users, the run file is named Run_GUI.command and is in the main directory. Open this file using a text editor e.g. TextEdit. 

Find the line "source /opt/anaconda3/...". This line contains two filepaths. The first should be to the activate file in your local copy of anaconda. Try to find this file by typing into your terminal the following commands:

```bash
cd /opt/
```
```bash
ls
```

This will list all the contents of the /opt/ folder. If anaconda3 is one of the results, then the default line in the provided .command file should be sufficient. The second filepath in this line should be the path to the conda environment you have just made. This may also be correct, but you can check by running:

```bash
conda info -envs
```

Make sure the filepath in the .command file matches the path returned for the environment you have made.

Next, find the line "cd /Users/...". Replace the filepath with the path to the directory you have cloned from Github that contains the file Inference_GUI.py and the .command file you are editing. The next step is to make the file executable. Do this by changing directory into the main directory and running the following command:

```bash
chmod +x Run_GUI.command
```

After this is done, the file is complete. You can now run the GUI by double-clicking on the .command file or by entering the following line in your terminal:

```bash
source Run_GUI.command
```

Note: if you prefer to use the terminal and therefore will run the above line, you do not need to be in the correct conda environment to run the GUI this way.

You can add the .command file as a shortcut on the homescreen by copying and pasting with Cmd + c, Cmd + v onto your homescreen or in another location. You can then change the image of the shortcut by first finding an image e.g. graphics/NN.icns, then opening the image, selecting it and then copying it. Then go to File->Get Info after selecting the shortcut for the .command file. Select the image on the left of the name Run_GUI.command at the top of the window, and paste the image in your clipboard. You can also hide the .command extension by selecting the checkbox under "Name & Extension" in the Get Info window.

### Additional Notes
Ensure that all commands are run in the Anaconda Prompt or Term                                                                                                                                                                                                                                                                                                                                                                                                                                                   inal.

## User Manual

This GUI is for the task of inferring the molecular weight distribution (MWD) from linear melt rheology using neural networks (NNs). The NNs are supplied along with the code here. 

Therefore, most of the workflow is centered on loading rheology and making it ready for input into the models. The following will detail how to do this.

### 1. Loading Rheology Data

Rheology data files can be loaded using the corresponding button on the function column to the left of thr UI. Please ensure the datafiles are in the correct format. This format should comprise of three columns: frequency in rad/s, G' in Pa, G'' in Pa. Column headers are fine but will be skipped, so please ensure the columns are in this order. Columns should be tab delimited. 

If rheology data is loaded correctly, it should be shown in the rheology plot panel, and the name of the file should be displayed at the bottom of your screen.

### 2. Universal space

One of the primary innovations of this methodology is the use of the "universal space of rheology". This allows one neural network to infer the MWD of many different polymer melts, simplifying system requirements. 

This means that for most polymers, after loading the rheology data, you will need to convert it into the universal space. This requires knowledge of three material parameters: the plateau modulus, the entanglement time, and the entanglement molecular weight. 

When the 'Universal Space' button is pressed after loading rheology, you will be prompted to enter the material and measurement temperature of the rheology data. If the material and temperature correspond to entries in the materials database, this will autofill the three parameters (press ok to apply). Alternatively, you can ignore these fields and check the box for "Enter parameters manually", and type in values for the parameters that will be used to apply the shift when ok is pressed.

If the parameters are not found in the database and you do not have parameters to enter manually, when ok is pressed, another dialog box will be opened prompting you to select a method of parameter shifting. This is used to shift parameters from values in the material database to the temperature you have entered using one of two methods: the Williams–Landel–Ferry (WLF) equation, or Arrhenius. Based on the selected shift method, parameters will automatically be shifted for you. Press ok when satisfied, and the shift will be applied.

You can save the parameters that are either manually entered or shifted from reference values to the materials database by ticking the relavent boxes in either dialog, and next time this material and temperature are entered, the parameters will be autofilled. If parameters are entered manually, even though the material and temperature are not used in the shift, make sure to select the correct values to ensure saving with the correct location in the database.

After this process, your rheology data should be on a different axis corresponding the to the universal space. You are ready to fit a relaxation spectrum.

### 3. Relaxation spectrum

A relaxation spectrum is fitted to all rheology data before being passed to the NN to make sure all data is formatted correctly and smooth out any noise that may be present. To do this, press the button marked "Fit Maxwell Modes". This will used the discrete multimode Maxwell model to represent the rheology as a number of relaxation modes with known relaxation timescales.

To get to this point you must have converted the data to the universal rheology space, so use the "Univ Default" pre-set for the parameters which should be auto-filled, and press ok. (You can use your own parameters if you have a different NN model). The plot should display text while the fit is completed, and then a number of green points should be added to the plot, representing the Maxwell modes. The relaxation spectrum has been fit to the data. If the dashed lines labelled "Maxwell Fit" in the legend to not agree with your data, something may have gone wrong in the process.

If the relaxation spectrum is fit correctly, you are now ready to pass the data to the NNs.

### 4. MWD Classification

If you know that the rheology data you have corresponds to a polydisperse polymer melt, you can skip this step. If you know that it is either monodisperse or bidisperse, please select the correct option from the "Select MWD Class for prediction" dropdown menu and continue to the next step.

However, if you are unsure about the class of MWD of your sample, you can use the classification function of this software to asses this. To initiated this, please press the "Classify MWD" button on the left of the screen. This will automatically use all of the avaialable classification models in the "NN_models" directory to classify the polymer's MWD into one of the three MWD classes mentioned.

The result will be presented below the plots panel, and the fraction of the number of models that agree with this prediction will also be shown. If all models predicted the same class, this is a good sign the prediction is accurate. The corresponding selection should have automatically been made in the "Select MWD Class for prediction" dropdown menu.

### 5. Make prediction

At this point, you should have loaded rheology data, converted to universal space, fit a relaxation spectrum, and know what class of MWD to use. The next step is to make a MWD prediction. Make sure the correct class is selected in the "Select MWD Class for Prediction" dropdown, below the MWD plot.

First, press the "Select NN Model" button. This will open a dialog, with a dropdown menu displaying all of the models for the MWD class you have selected. Multiple options are given as NNs will give slightly different results even if trained identically. Select one of the models from the dropdown, and confirm by pressing "ok". The name of this model should be displayed at the bottom of your screen. Next, press "Make MWD Prediction". This may take a moment, but after the prediction is made, it will be displayed on the right plot of the plots panel.

You can now view the predicted statistics with the "Predicted MWD Stats" button, save the prediction as a text file, save the figure, or clean up the prediction before any of these options.

To clean the prediction, press the button that has appeared on the left of your screen marked "Clean Prediction". You will be prompted to enter a threshold, or alternatively a min/max x range. These options will remove non-zero values of the prediction below the threshold or outside the x range respectively. This is because occasionally the NN predicts low volume-fraction components where they are unlikely to be, which will affect statistics such as Mn, but not noticeably affect the rheology. The MWD will be renormalised after this is done, and can be undone afterwards with a corresponding button that will appear. 

### 6. Compare with GPC

If you want to test the prediction against experimental GPC results, you can load the GPC file using the button on the left of the window. The file format must be made compatible with the software to be loaded correctly. For polydisperse samples, this should be M (g/mol) or log10M in the left column, and dW/dlogM in the right column. Normalisation will be done on loading so do not worry about this. This will then be plotted on the MWD plot.

For monodisperse and bidisperse samples, the file format is different. Each component (1 or 2 respectively) should be on a single row. The columns go as: volume fraction   Mw  PDI. The file should be tab delimited. This will plot the components as if they are narrow log-normal distributions, although this assumption will not be valid for slightly higher dispersities (e.g. >1.1).

Stats are also calculated for the loaded GPC and can be seen by pressing the "GPC Stats" button. 

### Additional Notes

For further questions or support, please contact py19rje@leeds.ac.uk.
