#@title 1.2 Install SD Dependencies
!pip install mediapipe
!pip install safetensors lark
os.makedirs('./embeddings', exist_ok=True)
import os
gitclone('https://github.com/Sxela/sxela-stablediffusion')
gitclone('https://github.com/Sxela/ControlNet')
try:
  os.rename('./sxela-stablediffusion', './stablediffusion')
except Exception as e: 
  print(e)
  if os.path.exists('./stablediffusion'):
    print('pulling a fresh stablediffusion')
    os.chdir( f'./stablediffusion')
    subprocess.run(['git', 'pull'])
    os.chdir( f'../')
try: 
    if os.path.exists('./ControlNet'):
      print('pulling a fresh ControlNet')
      os.chdir( f'./ControlNet')
      subprocess.run(['git', 'pull'])
      os.chdir( f'../')
except: pass


if True:
  
  !pip install --ignore-installed Pillow==9.0.0
  !pip install -e ./stablediffusion
  !pip install ipywidgets==7.7.1
  !pip install transformers==4.19.2

  !pip install omegaconf
  !pip install einops
  !pip install "pytorch_lightning>1.4.1,<=1.7.7"
  !pip install scikit-image
  !pip install opencv-python
  !pip install ai-tools
  !pip install cognitive-face
  !pip install zprint
  !pip install kornia==0.5.0

  !pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
  !pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

  !pip install lpips
  !pip install keras
  
  gitclone('https://github.com/Sxela/k-diffusion')
  os.chdir( f'./k-diffusion')
  subprocess.run(['git', 'pull'])
  !pip install -e .
  os.chdir( f'../')
  import sys
  sys.path.append('./k-diffusion')

!pip install wget
!pip install webdataset

try:
  import open_clip
except: 
  !pip install open_clip_torch
  import open_clip