@echo off

setlocal

set "python_dir=%~dp0python"
set "scripts_dir=%~dp0python\Scripts"
set "lib_dir=%~dp0python\Lib\site-packages"
set "pip_py=%~dp0get-pip.py"
set "venv_dir=%~dp0env"

REM Check if Git is already installed
git --version > nul 2>&1
if %errorlevel% equ 0 (
    echo Skipping git as it`s installed.
) else ( 
    echo Git not installed, please run install.bat
    echo Exiting.
    exit /b -1 )

if not exist %venv_dir%\Scripts\activate.bat ( 
    echo Virtual env not installed, please run install.bat
    echo Exiting.
    exit /b -1
)

REM Setting var to skip install inside the notebook
set IS_DOCKER=1 
set IS_LOCAL_INSTALL=1

echo Activating virtual environment 
call %venv_dir%\Scripts\activate"

python -c "import torch; from xformers import ops; assert torch.cuda.is_available(), 'Cuda not available, plese run install.bat'"
if %errorlevel% equ 1 (exit /b -1)

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