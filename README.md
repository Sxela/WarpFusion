# WarpFusion
WarpFusion

Guides made by users:

- 05.05.2023, v0.10 [Video to AI Animation Tutorial For Beginners: Stable WarpFusion + Controlnet | MDMZ](https://youtu.be/HkM-7wxtkGA)
- 11.05.2023, v0.11 [How to use Stable Warp Fusion](https://www.youtube.com/watch?v=FxRTEILPCQQ)

- 13.05.2023, v0.8  [Warp Fusion Local Install Guide (v0.8.6) with Diffusion Demonstration](https://www.youtube.com/watch?v=wqXy_r_9qw8)

- 14.05.2023, v0.12 [Warp Fusion Alpha Masking Tutorial | Covers Both Auto-Masking and Custom Masking](https://www.youtube.com/watch?v=VMF7L0czyIg)

- 23.05.2023, v0.12 [STABLE WARPFUSION TUTORIAL - Colab Pro & Local Install](https://www.youtube.com/watch?v=m8xaPnaooyg)

- 15.06.2023, v0.13 [AI Animation out of Your Video: Stable Warpfusion Guide (Google Colab & Local Intallation)](https://www.youtube.com/watch?v=-B7WtxAAXLg)

- 17.06.2023, v0.14 [Stable Warpfusion Tutorial: Turn Your Video to an AI Animation](https://www.youtube.com/watch?v=tUHCtQaBWCw)

- 21.06.2023, v0.14 [Avoiding Common Problems with Stable Warpfusion](https://www.youtube.com/watch?v=GH420ol2sCw)

- 21.06.2023, v0.15 [Warp Fusion: Step by Step Tutorial](https://www.youtube.com/watch?v=0AT8esyY0Fw)

- 04.07.2023, v0.15 [Intense AI Video Maker (Stable WarpFusion Tutorial)](https://www.youtube.com/watch?v=mVze7REhjCI&ab_channel=MattWolfe)

- 15.08.2023, v0.17 [BEST Laptop for AI ( SDXL & Stable Warpfusion ) ft. RTX 4090 - Make AI Art FREE and FAST!](https://www.youtube.com/watch?v=SM0Mxmhfj7A)

# Local installation guide for Windows (venv)

## Run once per notebook version (recommended)

1. Create a folder for WarpFusion. It's recommended to have a general folder for WarpFusion and subfolders for each version. Like ```C:\code\WarpFusion\0.16.11\``` for version 0.16.11
2. Download [install.bat](https://github.com/Sxela/WarpFusion/releases/download/v0.1.0/install.bat) and save it into your WarpFolder, ```C:\code\WarpFusion\0.16.11\``` in this example.
3. Run install.bat. It will download and install python, git, and create a virtual python environment called "env" inside our folder and install dependencies, required to run the notebook and jupyter server for local colab. When git install window appears, use the default settings. The installation will contiinue after you install git. 
4. Download [run.bat](https://github.com/Sxela/WarpFusion/releases/download/v0.1.0/run.bat) and save it into your WarpFolder, ```C:\code\WarpFusion\0.16.11\``` in this example.

## Run to launch
1. Execute run.bat. It will activate the environment and start jupyter server. 
2. After the server has launched, go to https://colab.research.google.com
3. Click File -> Upload Notebook and upload the *.ipynb file
4. Click on the dropdown menu near "Connect" or "Reconnect" button on the topright part of the interface.
5. Select "connect to a local runtime" and paste the URL that will be generated below, which looks like "http://localhost:8888/?token=somenumbers" 
6. Click "Connect" and CTRL+F9 to run all cells. 

# Local installation guide for Windows (anaconda)

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

# Docker install
## Run once to install (and once per notebook version)
1. Create a folder for warp, for example ```d:\warp```
2. Download Dockerfile and docker-compose.yml to ```d:\warp```
3. Edit docker-compose.yml so that volumes point to your model, init_images, images_out folders that are outside of the warp folder. For example, ```d:\models\:/content/models``` will expose d:\models as /content/models to the notebook
4. Download and install docker from here - https://docs.docker.com/get-docker/ 
5. Run ```docker-compose up --build``` inside the warp folder. 
6. Go to https://colab.research.google.com
3. Click File -> Upload Notebook and upload the *.ipynb file
4. Click on the dropdown menu near "Connect" or "Reconnect" button on the topright part of the interface.
5. Select "connect to a local runtime" and paste the token that was generated in your docker container, but leave the url as localhost. Should look like "http://localhost:8888/?token=somenumbers" 
6. Click "Connect" and CTRL+F9 to run all cells. 

## Run to launch 
1. Run ```docker-compose up ``` inside the warp folder. 
2. Go to https://colab.research.google.com
3. File -> open notebook -> open your previouslty uploaded notebook
4. Click on the dropdown menu near "Connect" or "Reconnect" button on the topright part of the interface.
5. Select "connect to a local runtime" and paste the token that was generated in your docker container, but leave the url as localhost. Should look like "http://localhost:8888/?token=somenumbers" 
6. Click "Connect" and CTRL+F9 to run all cells. 
