#@title ### 1.4 Install and import dependencies
!pip install opencv-python==4.5.5.64
!pip uninstall torchtext -y
!pip install pandas matplotlib

# if is_colab:
#   gitclone('https://github.com/vacancy/PyPatchMatch', recursive=True)
#   !make ./PyPatchMatch
#   from PyPatchMatch import patch_match

import pathlib, shutil, os, sys

if not is_colab:
  # If running locally, there's a good chance your env will need this in order to not crash upon np.matmul() or similar operations.
  os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

PROJECT_DIR = os.path.abspath(os.getcwd())
USE_ADABINS = False

if is_colab:
  if google_drive is not True:
    root_path = f'/content'
    model_path = '/content/models' 
else:
  root_path = os.getcwd()
  model_path = f'{root_path}/models'


multipip_res = subprocess.run(['pip', 'install', 'lpips', 'datetime', 'timm', 'ftfy', 'einops', 'pytorch-lightning', 'omegaconf'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(multipip_res)

if is_colab:
  subprocess.run(['apt', 'install', 'imagemagick'], stdout=subprocess.PIPE).stdout.decode('utf-8')

# try:
#   from CLIP import clip
# except:
#   if not os.path.exists("CLIP"):
#     gitclone("https://github.com/openai/CLIP")
#   sys.path.append(f'{PROJECT_DIR}/CLIP')

try:
  from guided_diffusion.script_util import create_model_and_diffusion
except:
  if not os.path.exists("guided-diffusion"):
    gitclone("https://github.com/crowsonkb/guided-diffusion")
  sys.path.append(f'{PROJECT_DIR}/guided-diffusion')

try:
  from resize_right import resize
except:
  if not os.path.exists("ResizeRight"):
    gitclone("https://github.com/assafshocher/ResizeRight.git")
  sys.path.append(f'{PROJECT_DIR}/ResizeRight')

if not os.path.exists("BLIP"):
    gitclone("https://github.com/salesforce/BLIP")
    sys.path.append(f'{PROJECT_DIR}/BLIP')
    # !pip install -r "{PROJECT_DIR}/BLIP/requirements.txt"

import torch
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import gc
import io
import math
import timm
from IPython import display
import lpips
from PIL import Image, ImageOps, ImageDraw
import requests
from glob import glob
import json
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
# from CLIP import clip
from resize_right import resize
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import Output
import hashlib
from functools import partial
if is_colab:
  os.chdir('/content')
  from google.colab import files
else:
  os.chdir(f'{PROJECT_DIR}')
from IPython.display import Image as ipyimg
from numpy import asarray
from einops import rearrange, repeat
import torch, torchvision
import time
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
device = DEVICE # At least one of the modules expects this name..

if torch.cuda.get_device_capability(DEVICE) == (8,0): ## A100 fix thanks to Emad
  print('Disabling CUDNN for A100 gpu', file=sys.stderr)
  torch.backends.cudnn.enabled = False


pipi('prettytable')
pipi('basicsr')