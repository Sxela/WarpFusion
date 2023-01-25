# WarpFusion
WarpFusion

# Local installation guide for Windows

## Run once

1. Download and install [git](https://github.com/git-for-windows/git/releases/download/v2.39.1.windows.1/Git-2.39.1-64-bit.exe)
2. Download and install [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
* You can skip these two steps and get a batch file [here](https://github.com/Sxela/WarpFusion/releases/download/v0.1.0/install_git_conda.bat). Download it and run (doesn't matter which folder you run it from)
It will download and install Git and Miniconda for you, you'll just need to accept all the incoming menus with default settings.

## Run once per notebook version (recommended)

1. Create a folder for WarpFusion. It's recommended to have a general folder for WarpFusion and subfolders for each version. Like ```C:\code\WarpFusion\v5.27.5\```
2. Download [prepare_env_relative.bat](https://github.com/Sxela/WarpFusion/releases/download/v0.1.0/prepare_env_relative.bat) and save it into your WarpFolder, ```C:\code\WarpFusion\v5.27.5\``` in this example.
3. Run prepare_env_relative.bat. It will create a virtual python environment called "env" inside our folder and install dependencies, required to run the notebook and jupyter server for local colab.
4. Download [run_relative.bat](https://github.com/Sxela/WarpFusion/releases/download/v0.1.0/run_relative.bat) and save it into your WarpFolder, ```C:\code\WarpFusion\v5.27.5\``` in this example.

## Run to launch
1. Execute run_relative.bat. It will activate the environment and start jupyter server. 
2. After the server has launched, go to https://colab.research.google.com
3. Click File -> Upload Notebook and upload the *.ipynb file
4. Click on the dropdown menu near "Connect" or "Reconnect" button on the topright part of the interface.
5. Select "connect to a local runtime" and paste the URL that will be generated below, which looks like "http://localhost:8888/?token=somenumbers" 
6. Click "Connect" and CTRL+F9 to run all cells. 
