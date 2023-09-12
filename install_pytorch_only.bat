
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

REM Set the filename of the Git installer and the download URL
set "GIT_INSTALLER=Git-2.33.0-64-bit.exe"
set "GIT_DOWNLOAD_URL=https://github.com/git-for-windows/git/releases/download/v2.33.0.windows.2/Git-2.33.0.2-64-bit.exe"



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

if not exist "%python_zip%" (
    echo Downloading Python 3.10...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%python_url%', '%python_zip%')"
)

if not exist "%python_dir%" (
    echo Extracting Python 3.10...
    powershell -Command "Expand-Archive '%python_zip%' -DestinationPath '%python_dir%'"
)

REM Set environment variable for embedded Python
set "PATH=%python_dir%;%scripts_dir%;%lib_dir%;%PATH%"

if not exist "%python_dir%\Lib\site-packages\pip" (
echo Installing pip...
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%pip_url%', '%pip_py%')"
python %pip_py% )

( 
echo python310.zip
echo Lib\site-packages
echo .
) > %python_dir%\python310._pth

if not exist "%python_dir%\Lib\site-packages\virtualenv" (
echo Installing virtualenv...
call pip install virtualenv )

if not exist "%venv_dir%" (
echo Creating virtual environment with Python 3.10...
call %python_dir%\python -m virtualenv --python="%python_dir%\python.exe" env )

echo Activating virtual environment 
call %venv_dir%\Scripts\activate"

@REM call pip show torch
call python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 xformers


python -c "import torch; print('Checking if cuda is available:', torch.cuda.is_available(), '\n,Checking xformers install:'); from xformers import ops"

if exist "%python_dir%/requirements.txt" (
	call python -m pip install -r requirements.txt
)
