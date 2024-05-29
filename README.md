# WarpFusion
WarpFusion

Latest public version: 
- [v0.21-AGPL](https://github.com/Sxela/WarpFusion/blob/v0.21-AGPL/stable_warpfusion.ipynb)
- [![Run v0.21 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github//Sxela/WarpFusion/blob/v0.21-AGPL/stable_warpfusion.ipynb)
- [Release Notes](https://github.com/Sxela/WarpFusion/releases/tag/v0.21)

If you find a public version elsewhere, before running as admin/root, make sure to check it for malware by comparing it to the latest notebook in this repo.

Greatly inspired by [Cameron Smith's](https://github.com/cysmith) [neural-style-tf](https://github.com/cysmith/neural-style-tf)

## Example videos 
[Example videos and settings](examples/readme.md)

## Guides made by users:

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

- 25.08.2023, ComfyWarp v0.1 [WarpFusion: Warp and Consistency explanation in ComfyUI](https://www.youtube.com/watch?v=ZuPBDRjwtu0&list=PL2cEnissQhlCko-T3gPH9ltMLMRjabpIS&index=3&ab_channel=DerpLearning)

- 2.09.2023, ComfyWarp v0.2 [WarpFusion: ComfyWarp iteration 2.](https://www.youtube.com/watch?v=vRpmx5Iusdo&list=PL2cEnissQhlCko-T3gPH9ltMLMRjabpIS&index=4&t=2s&ab_channel=DerpLearning)

- 3.09.2023, v0.16+ [WarpFusion - Multiple Masked Prompts Guide](https://www.youtube.com/watch?v=t_e-YRfLR7Y&list=PL2cEnissQhlCko-T3gPH9ltMLMRjabpIS&index=5&t=2s&ab_channel=DerpLearning)

- 20.09.2023, v0.19 [Warp Fusion Tutorial | Video to AI Video | Stable diffusion (Hindi)](https://www.youtube.com/watch?v=JeyRUFRPCXE&ab_channel=VFXMinds)

- 10.10.2023, ComfyWarp v0.4 [WarpFusion: ComfyWarp v0.4.2 (schedulers, flow_blend)](https://www.youtube.com/watch?v=CdP8fus_vNg&list=PL2cEnissQhlCko-T3gPH9ltMLMRjabpIS&index=6&t=1s&ab_channel=DerpLearning)

- 4.11.2023, ComfyWarp v0.5 [WarpFusion: ComfyWarp v0.5 - FixedQueue](https://www.youtube.com/watch?v=aAz3ELgYkqo&list=PL2cEnissQhlCko-T3gPH9ltMLMRjabpIS&index=7&t=1s&ab_channel=DerpLearning)

- 13.11.2023, v0.27
[Stable WarpFusion v0.27 - Changelog](https://www.youtube.com/watch?v=VXS-bpWy7CA&lc=Ugz0GUHVSNt41cl-aD54AaABAg&ab_channel=DerpLearning)

- 17.01.2024, v0.30
[Stable WarpFusion v0.30 - Changelog](https://www.youtube.com/watch?v=OlPTcIZGxRM)

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

# Local installation guide for Linux-Ubuntu 22.04 (venv): 

## Pre-requisites:

- Make sure that Ubuntu packages for CUDA toolkit and the latest NVIDIA utils are installed. Check using nvidia-smi command.

- ⚠️ **Warning: Destructive Process Ahead** ⚠️
  
  **Clean Python Environment**: 
  - If you haven't followed best practices for Python virtual environments, you may want to clean your system. 
  - **Be warned, this is a destructive process** and will remove all Python packages installed in the global environment.
 
    ```bash
    pip freeze > uninstall.txt
    pip uninstall -r uninstall.txt
    sudo pip freeze > uninstall.txt
    sudo pip uninstall -r uninstall.txt
    rm -rf ~/.cache
    ```
## Installation Steps:
1. **Choose Directory**: 
    - Open a terminal and navigate to your home directory or a directory of your choice.
    
    ```bash
    cd $HOME or cd ~
    ```
2. **Clone Repository**: 
    - Clone the WarpFusion repository.
    
    ```bash
    git clone https://github.com/WarpFusion/WarpFusion.git
    ```
3. **Navigate to Folder**: 
    - Enter the WarpFusion directory.
    
    ```bash
    cd WarpFusion
    ```
4. **Run the Linux Installer**: 
    - Make the script executable and run it.
    
    ```bash
    chmod +x linux_install.sh
    ./linux_install.sh
    ```
    - Script will prompt you to enter a "version" to create your working folder, this can be any name you choose as at will append to "WarpFusion", ie: "WarpFusion0.23.11
## Run to launch
1. Navigate to your `WarpFusion(version)` folder and execute the run script:
    ```bash
    cd $HOME/WarpFusion(version)
    ./run.sh
    ```
2. After the server has launched, go to https://colab.research.google.com
3. Click File -> Upload Notebook and upload the *.ipynb file
4. Click on the dropdown menu near "Connect" or "Reconnect" button on the topright part of the interface.
5. Select "connect to a local runtime" and paste the URL that will be generated below, which looks like "http://localhost:8888/?token=somenumbers" 
6. Click "Connect" and CTRL+F9 to run all cells. 

## Troubleshoot python virtual environment issues
- Delete your python virtual environment "warpenv" and re-run the running the script (backup your models, images and videos just in case).
    ```bash
    cd $HOME/WarpFusion(version)
    rm -rf warpenv
    ```

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

## Credits

This notebook uses:

[Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis & StabilityAI\
[K-diffusion wrapper](https://github.com/crowsonkb/k-diffusion) by Katherine Crowson\
[RAFT model](https://github.com/princeton-vl/RAFT) by princeton-vl  \
Consistency Checking (legacy) from [maua](https://github.com/maua-maua-maua/maua) \
Color correction from [pengbo-learn](https://github.com/pengbo-learn/python-color-transfer)\
Auto brightness adjustment from [progrockdiffusion](https://github.com/lowfuel/progrockdiffusion)

[AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui): weighted prompt keywords, lora, embeddings, attention hacks  \
Reconstructed noise - based on changes [suggested](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/736) by briansemrau 

[ControlNet](https://github.com/lllyasviel/ControlNet) \
TemporalNet, Controlnet Face and lots of other controlnets (check model list)\
[BLIP](https://github.com/salesforce/BLIP) by SalesForce \
[RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) (as external [cli package](https://github.com/Sxela/RobustVideoMattingCLI))  \
[CLIP](https://github.com/openai/CLIP) \
[FreeU Hack](https://huggingface.co/papers/2309.11497)  \
[Experimental ffmpeg Deflicker](https://video.stackexchange.com/questions/23384/remove-flickering-due-to-artificial-light-with-ffmpeg)  \
[Dw pose estimator](https://github.com/IDEA-Research/DWPose)  \
[SAMTrack Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything) (with [cli my wrapper and edits](https://github.com/Sxela/Segment-and-Track-Anything-CLI)) \
[ComfyUI](https://github.com/comfyanonymous/ComfyUI): sdxl controlnet loaders, control loras \
[animatediff](https://github.com/guoyww/animatediff) base \
animatediff wrapper for compvis models from [comfyui-animatediff](https://github.com/ArtVentureX/comfyui-animatediff) \
IP Adapters implementation from [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)


\
DiscoDiffusion legacy credits:

Original notebook by [Somnai](https://twitter.com/Somnai_dreams), [Adam Letts](https://twitter.com/gandamu_ml) and lots of other awesome people!

Turbo feature by [Chris Allen](https://twitter.com/zippy731)

Improvements to ability to run on local systems, Windows support, and dependency installation by [HostsServer](https://twitter.com/HostsServer)

Warp and custom model support by [Alex Spirin](https://twitter.com/devdef)



## Citation

If you find this code useful for your research, please cite:
```
@misc{Spirin2022,
  author = {Spirin, Alex},
  title = {warpfusion},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Sxela/WarpFusion}},
}
```
