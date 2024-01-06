#!/bin/bash

# Define current $PYTHON environment installation
PYTHON="python3.10" # Replace with your desired $PYTHON version

# Prompt for Warp version and store it in a variable
read -p "Warp version: " warp_version

# Create the directory structure with the specified version
warp_directory=$(pwd)/"WarpFusion-${warp_version}"
mkdir -p "${warp_directory}"

# Change to the Warp directory
cd "${warp_directory}"

# Update and install required packages
echo "Updating and installing packages..."
source /etc/lsb-release
if [ -f "/etc/arch-release" ]; then
  sudo pacman -Syu && sudo pacman -S --needed git $PYTHON python3-pip python3-opencv \
    imagemagick ffmpeg jupyter-notebook $PYTHON-virtualenv
elif [ -f "/etc/debian_version" ]; then	# Removed typo in the original script: "debian-version" -> "debian_version"
  sudo apt update && sudo apt install -y git $PYTHON python3-pip python3-opencv \
	imagemagick ffmpeg jupyter-notebook $PYTHON-venv libjpeg-dev $PYTHON-dev 2>&1 # Added libjpeg-dev and $PYTHON variable
else
  echo -e "This program is officially supported only on Arch and Debian based distributions. The script will install the following dependencies for you: \n[$PYTHON-venv, python3-pip, python3-opencv, imagemagick, ffmpeg, jupyter-notebook]\nIf you'd like to proceed with the installation, enter 'Y'. \nPress any other key to exit the installer."
  read continue
  if [[ $continue != "y" ]] && [[ $continue != "Y" ]]; then
    echo "exiting installer"
    exit
  fi
fi

# Get the current directory and display it
current_dir=$(pwd)
echo "Current directory: ${current_dir}"

# Function to handle errors
handle_error() {
    if [ $? -ne 0 ]; then
        echo "An error occurred. Exiting."
        exit 1
    fi
}

# Check for existence of the virtual environment directory
if [ ! -d "warpenv" ]; then
    echo "Creating $PYTHON virtual environment..."
    $PYTHON -m venv warpenv
    handle_error
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source warpenv/bin/activate
handle_error

echo "Environment is set up and activated."

# Upgrade pip and setuptools
echo "Upgrading pip..."
pip install --upgrade pip
pip install --upgrade setuptools wheel
handle_error

# Create a $PYTHON virtual environment if it doesn't exist and activate it
echo "Current directory: $(pwd)"
if [ ! -d "warpenv" ]; then
        echo "Creating and activating $PYTHON virtual environment..."
        $PYTHON -m venv warpenv
else
        source warpenv/bin/activate
        echo "Activating existing environment"
fi
# Error handling
if [ $? -ne 0 ]; then
        echo "An error ocurred. Exiting."
        exit 1
fi

# clean pip cache
echo "Cleaning pip cache..."
pip cache purge
handle_error

# Check if required $PYTHON packages are already installed
if ! pip list | grep -q "torch\|torchvision\|torchaudio"; then
    echo "Installing $PYTHON packages..."
    pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
    pip uninstall torchtext -y
    pip install xformers==0.0.19
    pip install onnxruntime onnxruntime-gpu gdown
    pip install diffusers==0.11.1
    pip install requests mediapipe piexif safetensors==0.3.2 lark pillow==9.0.0 wget webdataset open_clip_torch opencv-python==4.5.5.64 pandas matplotlib fvcore ipywidgets==7.7.1 transformers==4.31.0 omegaconf einops "pytorch_lightning>1.4.1,<=1.7.7" scikit-image opencv-python ai-tools cognitive-face zprint kornia==0.5.0 lpips keras datetime timm==0.6.7 prettytable basicsr fairscale realesrgan torchmetrics==0.11.4   
fi

# Function to clone or refresh a Git repository
function clone_or_refresh_repo {
    local repo_url="$1"
    local repo_name=$(basename "$repo_url" .git)
    local install_cmd="$2"
    local repo_dir="$3"

    if [ -z "$repo_dir" ]; then
	local repo_dir="$repo_name"
    fi

    if [ ! -d "$repo_name" ]; then
        echo "Cloning $repo_name..."
	git clone "$repo_url" "content/$repo_dir"	
    else
        echo "Updating $repo_name..."
        cd "$repo_dir"
        git pull
        cd ..
    fi

    if [ -n "$install_cmd" ]; then
        eval "$install_cmd"
    fi
}

source warpenv/bin/activate
# Clone and install repositories with conditions
clone_or_refresh_repo "https://github.com/Sxela/sxela-stablediffusion.git" "pip install -e content/stablediffusion" "stablediffusion"
clone_or_refresh_repo "https://github.com/Sxela/Segment-and-Track-Anything-CLI.git" "pip install -e content/Segment-and-Track-Anything-CLI/sam"
clone_or_refresh_repo "https://github.com/Sxela/ControlNet-v1-1-nightly.git" "" "ControlNet"
clone_or_refresh_repo "https://github.com/CompVis/taming-transformers.git" "pip install -e content/taming-transformers"
clone_or_refresh_repo "https://github.com/openai/CLIP.git" "pip install -e content/CLIP"
clone_or_refresh_repo "https://github.com/IDEA-Research/GroundingDINO.git"
clone_or_refresh_repo "https://github.com/crowsonkb/guided-diffusion.git" "pip install -e content/guided-diffusion"
clone_or_refresh_repo "https://github.com/Sxela/k-diffusion.git" "pip install -e content/k-diffusion"
clone_or_refresh_repo "https://github.com/assafshocher/ResizeRight.git"
clone_or_refresh_repo "https://github.com/salesforce/BLIP.git"
clone_or_refresh_repo "https://github.com/pengbo-learn/python-color-transfer.git"
clone_or_refresh_repo "https://github.com/Sxela/generative-models.git"
clone_or_refresh_repo "https://github.com/Sxela/ComfyUI"
clone_or_refresh_repo "https://github.com/guoyww/AnimateDiff.git" "" "animatediff"
echo "###############################################################################################################################################################################################################################################################"
clone_or_refresh_repo "https://github.com/ArtVentureX/comfyui-animatediff"
pwd
cd content
cd comfyui-animatediff
git checkout 9d32153349aa15c6867a61f65b3e4bec74aa403a
cd "${warp_directory}"
echo "###############################################################################################################################################################################################################################################################"


# Set JUPYTER_CONFIG_DIR to specify the configuration directory
export JUPYTER_CONFIG_DIR=$(pwd)/.jupyter

# Install Jupyter kernel and extensions
source warpenv/bin/activate # Removed "activate" after "source" to fix the error: "source: not found"
echo "Installing Jupyter kernel and extensions..."
pip install entrypoints==0.4 ipython==8.10.0 jupyter_client==7.4.9 jupyter_core==5.2.0 packaging==22.0 tzdata==2022.7 traitlets==5.9.0 ipykernel --force-reinstall diffusers==0.11.1 nbclassic gdown

# Install Jupyter kernel in the custom directory
echo "Installing Jupyter kernel in the custom directory..." # Adding Custom Kernel Directory
custom_kernel_dir="${warp_directory}/warpenv" # Replace with your desired path
$PYTHON -m ipykernel install --name user --prefix="${custom_kernel_dir}"
handle_error

$PYTHON -m pip install --upgrade jupyter_http_over_ws>=0.0.7 && jupyter serverextension enable --py jupyter_http_over_ws
handle_error

# Create symbolic link for libnvrtc
echo "Creating symbolic link for libnvrtc..."
ln -sf $HOME/.local/lib/$PYTHON/site-packages/torch/lib/libnvrtc-672ee683.so.11.2 $(pwd)/warpenv/lib/$PYTHON/site-packages/torch/lib/libnvrtc.so
handle_error

echo "Creating run_linux.sh..."

# Appending the content of the original run_linux.sh into the new run_linux.sh file
cat << 'EOF' >> run_linux.sh
#!/bin/bash

# Define directories and files
VENV_DIR=$(pwd)/warpenv

# Check if Git is installed
if git --version >/dev/null 2>&1; then
    echo "Skipping git as it's installed."
else
    echo "Git not installed. Please install Git first."
    echo "Exiting."
    exit 1
fi

# Check if Virtual Environment is installed
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "Virtual env not installed. Please run install.sh"
    echo "Exiting."
    exit 1
fi

# Setting variables to skip install inside the notebook
export IS_DOCKER=1
export IS_LOCAL_INSTALL=1

# Activate virtual environment
echo "Activating virtual environment."
source ${VENV_DIR}/bin/activate

# Check for required $PYTHON packages
$PYTHON -c "import torch; from xformers import ops; assert torch.cuda.is_available(), 'Cuda not available, please check your apt repositories'"
if [ $? -eq 1 ]; then
    exit 1
fi

# Change into 'content' directory
cd content || { echo "Directory content not found. Exiting."; exit 1; }

# Set JUPYTER_CONFIG_DIR to specify the configuration directory
export JUPYTER_CONFIG_DIR=$(pwd)/.jupyter

# Launch Jupyter server
echo "Launching Jupyter server."
echo "-----"
echo "After the server has launched, go to https://colab.research.google.com"
echo "Click File -> Upload Notebook and upload the *.ipynb file"
echo "Click on the dropdown menu near 'Connect' or 'Reconnect' button on the top-right part of the interface."
echo "Select 'connect to a local runtime' and paste the URL that will be generated below,"
echo "which looks like 'http://localhost:8888/?token=somenumbers'"
echo "Click 'Connect' and CTRL+F9 to run all cells."
echo "------"
jupyter notebook  --allow-root --ServerApp.open_browser=False --no-browser --ServerApp.allow_remote_access=True --ServerApp.allow_origin='https://colab.research.google.com' --port=8888 --ServerApp.port_retries=0 --ip=0.0.0.0

# Deactivate virtual environment
echo "Deactivating virtual environment..."
deactivate
# Return to WarpFusion Directory from content directory
cd ..


EOF

chmod +x run_linux.sh
