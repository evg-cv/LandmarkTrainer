## LipDetector

## Overview

This project is to detect the lips for the estimation of person talking. The deep learning model is used to detect the lips
and the OpenCV & dlib frameworks are used for tracking the face and lips.

## Structure

- src

    The source code for estimation and detection

- utils

    * model directory
    * source code to manage the folders and files of this project
    
- app

    The main execution file
    
- requirements

    All the dependencies for this project
    
- settings

    Image path to estimate the face, models path
    
## Installation

- Environment

    Ubuntu 18.04, Windows 10, Python 3.6
    
- Dependency Installation

    Please go ahead to this directory and run the following command in the terminal.
    ```
    pip3 install -r requirements.txt
    ```

## Execution

- Please create the "model" folder in the utils folder and copy the 2 models of face & lip detection.

- Please run the following command in the terminal

    ```
    python3 app.py
    ```
