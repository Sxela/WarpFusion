#!/bin/bash

# Prompt for Warp version
read -p "Warp version: " warp_version

# Create the directory structure with the specified version
warp_directory="WarpFusion${warp_version}"
mkdir -p "${warp_directory}"

# Change to the Warp directory
cd "${warp_directory}"

# Update and install required packages
echo "Updating and installing packages..."

source /etc/lsb-release
if [ -f "/etc/arch-release" ]; then
  sudo pacman -Syu && sudo pacman -S --needed git python python-pip python-opencv \
    imagemagick ffmpeg jupyter-notebook python-virtualenv
elif [ -f "/etc/debian-version" ]; then
  sudo apt update && sudo apt install -y git python3.10 python3-pip python3-opencv \
	imagemagick ffmpeg jupyter-notebook python3.10-venv > /dev/null
else
  echo -e "This program only officially supports Arch and Debian based distros, you require these dependencies \n[python3-venv, python3-pip, python3-opencv, imagemagick, ffmpeg jupyter-notebook, ] \nIf you would like to continue with the install and install the dependencies yourself type Y \notherwise press anykey to exit the installer. (NOTE the installation may fail if dependencies aren't found)"
  read continue
  if [[ $continue != "y" ]] && [[ $continue != "Y" ]]; then
    echo "exiting installer"
    exit
  fi
fi



# Function to display a progress bar
function progress_bar {
    local duration="$1"
    local size="$2"
    local i
    for ((i = 0; i < size; i++)); do
        echo -ne "#"
        sleep "$duration"
    done
    echo -ne "\n"
}

# Create a Python virtual environment if it doesn't exist and activate it
if [ ! -d "warpenv" ]; then
    echo "Creating and activating Python virtual environment..."
    python3 -m venv warpenv
    source warpenv/bin/activate
else
    echo "Python virtual environment 'warpenv' already exists. Skipping environment creation and activation."
fi

# Check if required Python packages are already installed
if ! pip list | grep -q "torch\|torchvision\|torchaudio"; then
    echo "Installing Python packages..."
    pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
    pip uninstall torchtext -y 
    pip install xformers==0.0.19 
    pip install requests mediapipe piexif safetensors lark Pillow==9.0.0 wget webdataset open_clip_torch opencv-python==4.5.5.64 pandas matplotlib fvcore ipywidgets==7.7.1 transformers==4.19.2 omegaconf einops "pytorch_lightning>1.4.1,<=1.7.7" scikit-image opencv-python ai-tools cognitive-face zprint kornia==0.5.0 lpips keras datetime timm==0.6.7 prettytable basicsr fairscale realesrgan torchmetrics==0.11.4
fi

# Function to clone or refresh a Git repository
function clone_or_refresh_repo {
    local repo_url="$1"
    local repo_name=$(basename "$repo_url" .git)
    local install_cmd="$2"

    if [ ! -d "$repo_name" ]; then
        echo "Cloning $repo_name..."
        git clone "$repo_url" "content/$repo_name"
    else
        echo "Updating $repo_name..."
        cd "$repo_name"
        git pull
        cd ..
    fi

    if [ -n "$install_cmd" ]; then
        eval "$install_cmd"
    fi
}

# Clone and install repositories with conditions
clone_or_refresh_repo "https://github.com/Sxela/sxela-stablediffusion.git" "pip install -e content/sxela-stablediffusion"
clone_or_refresh_repo "https://github.com/Sxela/Segment-and-Track-Anything-CLI.git" "pip install -e content/Segment-and-Track-Anything-CLI/sam"
clone_or_refresh_repo "https://github.com/Sxela/ControlNet-v1-1-nightly.git"
clone_or_refresh_repo "https://github.com/CompVis/taming-transformers.git" "pip install -e content/taming-transformers"
clone_or_refresh_repo "https://github.com/openai/CLIP.git" "pip install -e content/CLIP"
clone_or_refresh_repo "https://github.com/IDEA-Research/GroundingDINO.git"
clone_or_refresh_repo "https://github.com/crowsonkb/guided-diffusion.git" "pip install -e content/guided-diffusion"
clone_or_refresh_repo "https://github.com/Sxela/k-diffusion.git" "pip install -e content/k-diffusion"
clone_or_refresh_repo "https://github.com/assafshocher/ResizeRight.git"
clone_or_refresh_repo "https://github.com/salesforce/BLIP.git"
clone_or_refresh_repo "https://github.com/pengbo-learn/python-color-transfer.git"
clone_or_refresh_repo "https://github.com/Stability-AI/generative-models.git"
clone_or_refresh_repo "https://github.com/comfyanonymous/ComfyUI.git"

# Install Jupyter kernel and extensions
echo "Installing Jupyter kernel and extensions..."
pip install entrypoints==0.4 ipython==8.10.0 jupyter_client==7.4.9 jupyter_core==5.2.0 packaging==22.0 tzdata==2022.7 ipykernel --force-reinstall
python -m ipykernel install --user
pip install --upgrade jupyter_http_over_ws>=0.0.7
jupyter serverextension enable --py jupyter_http_over_ws

# Create symbolic link for libnvrtc
echo "Creating symbolic link for libnvrtc..."
ln -s warpenv/lib/python3.10/site-packages/torch/lib/libnvrtc-672ee683.so.11.2 warpenv/lib/python3.10/dist-packages/torch/lib/libnvrtc.so

# Start Jupyter Notebook
echo "Starting Jupyter Notebook..."
jupyter notebook content/ --allow-root --NotebookApp.open_browser=False --no-browser --NotebookApp.allow_remote_access=True --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0 --ip=0.0.0.0

