#@title Install xformers
#@markdown Sometimes it detects the os incorrectly. If you see it mention the wrong os, try forcing the correct one and running this cell again.\
#@markdown If torch version needs to be donwgraded, the environment will be restarted. 
#@markdown # If you see "you session has crashed" message in this cell, just press CTRL+F10 or Runtime->Run all
#@markdown Do not delete the environment, it is an expected behavior. 
# import torch
import subprocess, sys
gpu = None
def get_version(package):
  proc = subprocess.run(['pip','show', package], stdout=subprocess.PIPE)
  out = proc.stdout.decode('UTF-8')
  returncode = proc.returncode
  if returncode != 0:
    return -1
  return out.split('Version:')[-1].split('\n')[0]
import os, platform
force_os = 'off' #@param ['off','Windows','Linux']

force_torch_reinstall = False #@param {'type':'boolean'}
force_xformers_reinstall = False #@param {'type':'boolean'}
if force_torch_reinstall:
  print('Reinstalling torch...')
  !pip uninstall torch  torchvision  torchaudio  cudatoolkit -y
  !conda uninstall pytorch  torchvision  torchaudio  cudatoolkit -y
  !pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113
  print('Reinstalled torch.')
if force_xformers_reinstall:
  print('Reinstalling xformers...')
  !pip uninstall xformers -y
  print('Reinstalled xformers.')
if platform.system() != 'Linux' or force_os == 'Windows':
  if not os.path.exists('ffmpeg.exe'):
    !pip install requests
    import requests

    url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
    print('ffmpeg.exe not found, downloading...')
    r = requests.get(url, allow_redirects=True)
    print('downloaded, extracting')
    open('ffmpeg-master-latest-win64-gpl.zip', 'wb').write(r.content)
    import zipfile
    with zipfile.ZipFile('ffmpeg-master-latest-win64-gpl.zip', 'r') as zip_ref:
        zip_ref.extractall('./')
    from shutil import copy 
    copy('./ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe', './')
  try:
    import xformers
    
  except:
    print('Failed to import xformers, installing.')
    if "3.10" in sys.version:
      if get_version('torch') == -1:
        print('Torch not found, installing. Will download >1GB and may take a while.')
        !pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113
      if "1.12" in get_version('torch'):
        print('Trying to install local xformers on Windows. Works only with pytorch 1.12.* and python 3.10.')
        !pip install https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
      elif "1.13" in get_version('torch'): 
        print('Trying to install local xformers on Windows. Works only with pytorch 1.13.* and python 3.10.')
        !pip install https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/torch13/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
     
if is_colab or (platform.system() == 'Linux') or force_os == 'Linux':
  print('Installing xformers on Linux/Colab.')
  # !wget https://github.com/Sxela/sxela-stablediffusion/releases/download/v1.0.0/xformers_install.zip
  # !unzip -o xformers_install.zip
  # !mv /content/xformers_install/* /usr/local/lib/python3.8/dist-packages/
  from subprocess import getoutput
  from IPython.display import HTML
  from IPython.display import clear_output
  import time
  #https://github.com/TheLastBen/fast-stable-diffusion
  s = getoutput('nvidia-smi')
  if 'T4' in s:
    gpu = 'T4'
  elif 'P100' in s:
    gpu = 'P100'
  elif 'V100' in s:
    gpu = 'V100'
  elif 'A100' in s:
    gpu = 'A100'
  
  for g in ['A4000','A5000','A6000']:
    if g in s:
      gpu = 'A100'

  for g in ['2080','2070','2060']:
    if g in s:
      gpu = 'T4'

  while True:
      try: 
          gpu=='T4'or gpu=='P100'or gpu=='V100'or gpu=='A100'
          break
      except:
          pass
      print(' it seems that your GPU is not supported at the moment')
      time.sleep(5)

  # if gpu == 'A100':
  #   !wget https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/A100
  #   !7z x /content/A100 -aoa -o/usr/local/lib/python3.8/dist-packages/

  # clear_output()
  try:
    import xformers.ops
  except:
    #fix thanks to kye#8384
    !pip install --upgrade pip
    if gpu == 'T4': 
      print('Downgrading torch for T4 to avoid CUDA lauch blocking errors with torch v2.')
      !pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
      !pip install xformers==0.0.16 
    else: 
      !pip install triton xformers
    # 
  print(' DONE !')