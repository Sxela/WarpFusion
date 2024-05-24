
@echo off

setlocal

set "python_url=https://www.python.org/ftp/python/3.10.0/python-3.10.0-embed-amd64.zip"
set "pip_url=https://bootstrap.pypa.io/get-pip.py"
set "python_zip=%~dp0python.zip"
set "python_dir=%~dp0python"
set "scripts_dir=%~dp0python\Scripts"
set "lib_dir=%~dp0python\Lib\site-packages"
set "pip_py=%~dp0get-pip.py"
set "venv_dir=%~dp0env"

REM include and libs folders download for locally buiding packages like insightface 
set "libs_zip=%~dp0libs.zip"
set "libs_dir=%~dp0python\libs"
set "include_zip=%~dp0include.zip"
set "include_dir=%~dp0python\include"
set "libs_url=https://github.com/Sxela/WarpFusion/releases/download/v0.1.0/libs.zip"
set "include_url=https://github.com/Sxela/WarpFusion/releases/download/v0.1.0/include.zip"

REM Set the filename of the Git installer and the download URL
set "GIT_INSTALLER=Git-2.33.0-64-bit.exe"
set "GIT_DOWNLOAD_URL=https://github.com/git-for-windows/git/releases/download/v2.33.0.windows.2/Git-2.33.0.2-64-bit.exe"

set "ffmpeg_zip=%~dp0ffmpeg-6.0-full_build.zip"
set "ffmpeg_url=https://github.com/GyanD/codexffmpeg/releases/download/6.0/ffmpeg-6.0-full_build.zip"
set "ffmpeg_dir=%~dp0ffmpeg-6.0-full_build"

REM Check if Git is already installed
git --version > nul 2>&1
if %errorlevel% equ 0 (
    echo Skipping git as it`s installed.
) else (
    REM Download Git installer if not already downloaded
    if not exist "%GIT_INSTALLER%" (
        echo Downloading Git installer...
        powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%GIT_DOWNLOAD_URL%', '%GIT_INSTALLER%')"
        echo Git installer downloaded.
    )

    if exist "%GIT_INSTALLER%" (
    REM Install Git using the installer
    echo Installing Git...
    "%GIT_INSTALLER%"
    echo Git has been installed. )
)

if not exist "%ffmpeg_zip%" (
    echo Downloading Ffmpeg...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%ffmpeg_url%', '%ffmpeg_zip%')"
)

if not exist "%ffmpeg_dir%" (
    echo Extracting Ffmpeg
    powershell -Command "Expand-Archive '%ffmpeg_zip%' -DestinationPath '%ffmpeg_dir%'"
    copy "%ffmpeg_dir%\ffmpeg-6.0-full_build\bin\ffmpeg.exe" "%~dp0%\"
)

if not exist "%python_zip%" (
    echo Downloading Python 3.10...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%python_url%', '%python_zip%')"
)

if not exist "%python_dir%" (
    echo Extracting Python 3.10...
    powershell -Command "Expand-Archive '%python_zip%' -DestinationPath '%python_dir%'"
)

if not exist "%libs_zip%" (
    echo Downloading Python libs...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%libs_url%', '%libs_zip%')"
)

if not exist "%libs_dir%" (
    echo Extracting Python libs...
    powershell -Command "Expand-Archive '%libs_zip%' -DestinationPath '%python_dir%'"
)

if not exist "%include_zip%" (
    echo Downloading Python include...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%include_url%', '%include_zip%')"
)

if not exist "%include_dir%" (
    echo Extracting Python include...
    powershell -Command "Expand-Archive '%include_zip%' -DestinationPath '%python_dir%'"
)

REM Set environment variable for embedded Python
set "PATH=%python_dir%;%scripts_dir%;%lib_dir%;%PATH%"

if not exist "%python_dir%\Lib\site-packages\pip" (
echo Installing pip...
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%pip_url%', '%pip_py%')"
python "%pip_py%" )

( 
echo python310.zip
echo Lib\site-packages
echo .
) > "%python_dir%\python310._pth"

if not exist "%python_dir%\Lib\site-packages\virtualenv" (
echo Installing virtualenv...
call pip install virtualenv )

if not exist "%venv_dir%" (
echo Creating virtual environment with Python 3.10...
call "%python_dir%\python" -m virtualenv --python="%python_dir%\python.exe" env )

powershell -Command "Expand-Archive '%include_zip%' -DestinationPath '%venv_dir%'"
powershell -Command "Expand-Archive '%libs_zip%' -DestinationPath '%venv_dir%'"

echo Activating virtual environment 
call "%venv_dir%\Scripts\activate"

call python -m pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118 xformers==0.0.21 

python -c "import torch; print('Checking if cuda is available:', torch.cuda.is_available(), '\n,Checking xformers install:'); from xformers import ops"

call python -m pip install requests mediapipe piexif safetensors==0.3.2 lark Pillow==9.0.1 wget webdataset open_clip_torch opencv-python==4.5.5.64 pandas matplotlib fvcore ipywidgets==7.7.1 transformers==4.31.0 omegaconf einops "pytorch_lightning>1.4.1,<=1.7.7" scikit-image opencv-python ai-tools cognitive-face zprint kornia==0.5.0 lpips keras datetime timm==0.6.7 prettytable basicsr fairscale realesrgan torchmetrics==0.11.4
call python -m pip install onnxruntime-gpu gdown
call python -m pip install diffusers==0.11.1

call git clone https://github.com/Sxela/sxela-stablediffusion "%~dp0stablediffusion" --depth=1 && python -m pip install -e "%~dp0stablediffusion"
call git clone https://github.com/Sxela/ControlNet-v1-1-nightly "%~dp0ControlNet" --depth=1
call python -m pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers -e git+https://github.com/openai/CLIP.git@main#egg=clip
call git clone https://github.com/Sxela/WarpFusion "%~dp0WarpFusion" --depth=1
call git clone https://github.com/crowsonkb/guided-diffusion "%~dp0guided-diffusion" --depth=1 && pip install -e "%~dp0guided-diffusion"
call git clone https://github.com/Sxela/k-diffusion "%~dp0k-diffusion" --depth=1 && pip install -e "%~dp0k-diffusion"
call git clone https://github.com/assafshocher/ResizeRight.git "%~dp0ResizeRight" --depth=1
call git clone https://github.com/salesforce/BLIP "%~dp0BLIP" --depth=1
call git clone https://github.com/pengbo-learn/python-color-transfer "%~dp0python-color-transfer" --depth=1
call git clone https://github.com/Sxela/generative-models "%~dp0generative-models" --depth=1
call git clone https://github.com/Sxela/ComfyUI "%~dp0ComfyUI" --depth=1
call git clone https://github.com/ArtVentureX/comfyui-animatediff "%~dp0comfyui-animatediff" 
call cd "%~dp0comfyui-animatediff"
call git checkout 9d32153349aa15c6867a61f65b3e4bec74aa403a
call cd "%~dp0"
call git clone https://github.com/guoyww/animatediff "%~dp0animatediff" --depth=1
call git clone https://github.com/Sxela/Depth-Anything-light "%~dp0Depth-Anything" --depth=1
call cd "%~dp0Depth-Anything"
call python -m pip install -r requirements.txt
call cd "%~dp0"
call git clone https://github.com/pengbo-learn/python-color-transfer "%~dp0python-color-transfer" --depth=1
call git clone https://github.com/Sxela/flow_tools "%~dp0flow_tools" --depth=1

REM Installing MSVC build tools 
call winget install -e --id Microsoft.VCRedist.2015+.x64 --force
call winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended" --force
call python -m pip install insightface
call python -m pip install Pillow==9.0.1

call python -m pip install notebook
call python -m pip install entrypoints==0.4 ipython==8.10.0 jupyter_client==7.4.9 jupyter_core==5.2.0 packaging==22.0 tzdata==2022.7 ipykernel --force-reinstall
call python -m ipykernel install --user

REM Setting var to skip install inside the notebook
set IS_DOCKER=1
set IS_LOCAL_INSTALL=1

echo Launching jupyter server.
echo -----
echo After the server has launched, go to https://colab.research.google.com
echo Click File -> Upload Notebook and upload the *.ipynb file
echo Click on the dropdown menu near "Connect" or "Reconnect" button on the topright part of the interface.
echo Select "connect to a local runtime" and paste the URL that will be generated below.
echo which looks like "http://localhost:8888/?token=somenumbers" 
echo Click "Connect" and CTRL+F9 to run all cells. 
echo ------
call jupyter notebook ./ --NotebookApp.allow_origin='https://colab.research.google.com' --no-browser --port=8888 --NotebookApp.port_retries=0
echo|set/p="Press <ENTER> to continue.."&runas/u: "">NUL

echo Deactivating virtual environment...
deactivate

REM This script downloads the embeddable Python 3.10 zip file from the official website, extracts it to a directory named `python`, creates a virtual environment using the `venv` module, installs `pip` and `virtualenv`, creates another virtual environment using `virtualenv`, and deactivates the virtual environment.

REM Source: Conversation with Bing, 19/07/2023
REM (1) python - Activate virtualenv and run .py script from .bat - Stack Overflow. https://stackoverflow.com/questions/47425520/activate-virtualenv-and-run-py-script-from-bat.
REM (2) VirtualEnv and python-embed - Stack Overflow. https://stackoverflow.com/questions/47754357/virtualenv-and-python-embed.
REM (3) python - Downloading sqlite3 in virtualenv - Stack Overflow. https://stackoverflow.com/questions/45704177/downloading-sqlite3-in-virtualenv.
