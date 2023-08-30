#@markdown specify path to your Stable Diffusion checkpoint (the "original" flavor)
#@title define SD + K functions, load model
from safetensors import safe_open
import argparse
import math,os,time
os.chdir( f'{root_dir}/src/taming-transformers')
import taming
os.chdir( f'{root_dir}')
import wget
import accelerate
import torch
import torch.nn as nn
from tqdm.notebook import trange, tqdm
sys.path.append('./k-diffusion')
os.chdir( f'{root_dir}/k-diffusion')
# import taming
import k_diffusion as K
os.chdir( f'{root_dir}')

from pytorch_lightning import seed_everything
from k_diffusion.sampling import  sample_euler, sample_euler_ancestral, sample_heun, sample_dpm_2, sample_dpm_2_ancestral, sample_lms, sample_dpm_fast, sample_dpm_adaptive, sample_dpmpp_2s_ancestral,  sample_dpmpp_sde, sample_dpmpp_2m

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from torch import autocast
import numpy as np

from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms

def model_to(model, device):
  for param in model.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
  

# import wget
model_version = 'control_multi'#@param ['v1','v1_inpainting','v1_instructpix2pix','v2_512','v2_depth', 'v2_768_v', "control_sd15_canny", "control_sd15_depth","control_sd15_hed",  "control_sd15_mlsd", "control_sd15_normal", "control_sd15_openpose", "control_sd15_scribble", "control_sd15_seg", 'control_multi' ]
if model_version == 'v1' :
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v1-inference.yaml"
if model_version == 'v1_inpainting':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v1-inpainting-inference.yaml"
if model_version == 'v2_512':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v2-inference.yaml"
if model_version == 'v2_768_v':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v2-inference-v.yaml"
if model_version == 'v2_depth':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v2-midas-inference.yaml"
  os.makedirs(f'{root_dir}/midas_models', exist_ok=True)
  if not os.path.exists(f"{root_dir}/midas_models/dpt_hybrid-midas-501f0c75.pt"):
    midas_url = 'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt '
    os.makedirs(f'{root_dir}/midas_models', exist_ok=True)
    wget.download(midas_url,  f"{root_dir}/midas_models/dpt_hybrid-midas-501f0c75.pt")
    # !wget -O  "{root_dir}/midas_models/dpt_hybrid-midas-501f0c75.pt" https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt 
control_helpers = {
    "control_sd15_canny":None,
    "control_sd15_depth":"dpt_hybrid-midas-501f0c75.pt",
    "control_sd15_hed":"network-bsds500.pth",
    "control_sd15_mlsd":"mlsd_large_512_fp32.pth",
    "control_sd15_normal":"dpt_hybrid-midas-501f0c75.pt",
    "control_sd15_openpose":["body_pose_model.pth", "hand_pose_model.pth"],
    "control_sd15_scribble":None,
    "control_sd15_seg":"upernet_global_small.pth",
    "control_sd15_temporalnet":None,
    "control_sd15_face":None
} 

if model_version == 'v1_instructpix2pix':
  config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v1_instruct_pix2pix.yaml"
vae_ckpt = '' #@param {'type':'string'}
if vae_ckpt == '': vae_ckpt = None
load_to = 'cpu' #@param ['cpu','gpu']
if load_to == 'gpu': load_to = 'cuda'
quantize = True #@param {'type':'boolean'}
no_half_vae = False #@param {'type':'boolean'}
import gc 
def load_model_from_config(config, ckpt, vae_ckpt=None, controlnet=None, verbose=False):
    with torch.no_grad():
      model = instantiate_from_config(config.model).eval().cuda()
      if gpu != 'A100':
        if no_half_vae:
          model.model.half()
          model.cond_stage_model.half()
          model.control_model.half()
        else: 
          model.half()
      gc.collect()

      print(f"Loading model from {ckpt}")
      if ckpt.endswith('.safetensors'):
        pl_sd = {}
        with safe_open(ckpt, framework="pt", device=load_to) as f:
          for key in f.keys():
              pl_sd[key] = f.get_tensor(key)
      else: pl_sd = torch.load(ckpt, map_location=load_to)

      if "global_step" in pl_sd:
          print(f"Global Step: {pl_sd['global_step']}")
      if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
      else: sd = pl_sd
      del pl_sd
      gc.collect()

      if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location=load_to)
        if "state_dict" in vae_sd:
          vae_sd = vae_sd["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }

      m, u = model.load_state_dict(sd, strict=False)
      if len(m) > 0 and verbose:
          print("missing keys:")
          print(m, len(m))
      if len(u) > 0 and verbose:
          print("unexpected keys:")
          print(u, len(u))

      if controlnet is not None:
        ckpt = controlnet
        print(f"Loading model from {ckpt}")
        if ckpt.endswith('.safetensors'):
          pl_sd = {}
          with safe_open(ckpt, framework="pt", device=load_to) as f:
            for key in f.keys():
                pl_sd[key] = f.get_tensor(key)
        else: pl_sd = torch.load(ckpt, map_location=load_to)

        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
          sd = pl_sd["state_dict"]
        else: sd = pl_sd
        del pl_sd
        gc.collect()
        m, u = model.control_model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m, len(m))
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u, len(u))


      return model

import clip
from kornia import augmentation as KA
from torch.nn import functional as F
from resize_right import resize

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

from einops import rearrange, repeat

def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
        # with torch.no_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs);# print(denoised.requires_grad)
        # with torch.enable_grad():
            # denoised = denoised.detach().requires_grad_()
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach();# print(cond_grad.requires_grad)
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return model_fn

def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
        # with torch.no_grad():
            # x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs);# print(denoised.requires_grad)
        # with torch.enable_grad():
            # print(sigma**0.5, sigma, sigma**2)
            denoised = denoised.detach().requires_grad_()
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach();# print(cond_grad.requires_grad)
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return model_fn


def make_static_thresh_model_fn(model, value=1.):
    def model_fn(x, sigma, **kwargs):
        return model(x, sigma, **kwargs).clamp(-value, value)
    return model_fn

def get_image_embed(x):
        if x.shape[2:4] != clip_size:
            x = resize(x, out_shape=clip_size, pad_mode='reflect')
        # print('clip', x.shape)
        # x = clip_normalize(x).cuda()
        x = clip_model.encode_image(x).float()
        return F.normalize(x)

def load_img_sd(path, size):
    # print(type(path))
    # print('load_sd',path)
    
    image = Image.open(path).convert("RGB")
    # print(f'loaded img with size {image.size}')
    image = image.resize(size, resample=Image.LANCZOS)
    # w, h = image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
   
    # image = image.resize((w, h), resample=Image.LANCZOS)
    if VERBOSE: print(f'resized to {image.size}')
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

# import lpips
# lpips_model = lpips.LPIPS(net='vgg').to(device)

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, image_cond=None):
        cond = prompt_parser.reconstruct_cond_batch(cond, 0)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, 0)
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        # print(cond, uncond)
        cond_in = torch.cat([uncond, cond])
        
        if image_cond is None: 
          uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
          return uncond + (cond - uncond) * cond_scale
        else: 
          if model_version != 'control_multi':
            if img_zero_uncond:
              img_in = torch.cat([torch.zeros_like(image_cond), 
                                          image_cond])
            else: 
              img_in = torch.cat([image_cond]*2)
            uncond, cond = self.inner_model(x_in, sigma_in, cond={"c_crossattn": [cond_in], 
                                                                  'c_concat': [img_in]}).chunk(2)
            return uncond + (cond - uncond) * cond_scale
            
          if model_version == 'control_multi' and controlnet_multimodel_mode != 'external':
            img_in = {}
            for key in image_cond.keys():
              img_in[key] = torch.cat([torch.zeros_like(image_cond[key]), 
                                          image_cond[key]]) if img_zero_uncond else torch.cat([image_cond[key]]*2)
    
            uncond, cond = self.inner_model(x_in, sigma_in, cond={"c_crossattn": [cond_in], 
                                                                  'c_concat': img_in,
                                                                  'controlnet_multimodel':controlnet_multimodel,
                                                                  'loaded_controlnets':loaded_controlnets}).chunk(2)
            return uncond + (cond - uncond) * cond_scale
          if model_version == 'control_multi' and controlnet_multimodel_mode == 'external':

              #wormalize weights
              weights = np.array([controlnet_multimodel[m]["weight"] for m in controlnet_multimodel.keys()])
              weights = weights/weights.sum()
              result = None
              # print(weights)
              for i,controlnet in enumerate(controlnet_multimodel.keys()):
                try:
                  if img_zero_uncond:
                    img_in = torch.cat([torch.zeros_like(image_cond[controlnet]), 
                                          image_cond[controlnet]])
                  else: 
                    img_in = torch.cat([image_cond[controlnet]]*2)
                except: 
                  pass
                
                if weights[i]!=0:
                  controlnet_settings = controlnet_multimodel[controlnet]

                  self.inner_model.inner_model.control_model = loaded_controlnets[controlnet]

                  uncond, cond = self.inner_model(x_in, sigma_in, cond={"c_crossattn": [cond_in], 
                                                                      'c_concat': [img_in]}).chunk(2)
                  if result is None:
                    result = (uncond + (cond - uncond) * cond_scale)*weights[i]
                  else: result = result + (uncond + (cond - uncond) * cond_scale)*weights[i]
              return result




import einops
class InstructPix2PixCFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, cond_scale, image_scale, image_cond):
        # c = cond
        # uc = uncond
        c = prompt_parser.reconstruct_cond_batch(cond, 0)
        uc = prompt_parser.reconstruct_cond_batch(uncond, 0)
        text_cfg_scale = cond_scale
        image_cfg_scale  = image_scale
        # print(image_cond)
        cond = {}
        cond["c_crossattn"] = [c]
        cond["c_concat"] = [image_cond]

        uncond = {}
        uncond["c_crossattn"] = [uc]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)

        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)

dynamic_thresh = 2.
device = 'cuda'
# config_path = f"{root_dir}/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
model_path = "/content/drive/MyDrive/models/protogenV22Anime_22.safetensors" #@param {'type':'string'}
import pickle
#@markdown ---
#@markdown ControlNet download settings
use_small_controlnet  = True 
# #@param {'type':'boolean'}
small_controlnet_model_path = '' 
# #@param {'type':'string'}
download_control_model = True 
# #@param {'type':'boolean'}
force_download = False #@param {'type':'boolean'}
controlnet_models_dir = "./ControlNet/models" #@param {'type':'string'}
if not is_colab and controlnet_models_dir.startswith('/content'):
  controlnet_models_dir = f"{root_dir}/ControlNet/models"
  print('You have a controlnet path set up for google drive, but we are not on Colab. Defaulting controlnet model path to ', controlnet_models_dir)
os.makedirs(controlnet_models_dir, exist_ok=True)
#@markdown ---

control_sd15_canny  = False 
control_sd15_depth  = False 
control_sd15_hed  = True 
control_sd15_mlsd  = False 
control_sd15_normal  = False 
control_sd15_openpose  = False 
control_sd15_scribble  = False
control_sd15_seg  = False 
control_sd15_temporalnet  = False
control_sd15_face  = False 

if model_version == 'control_multi':
  control_versions = []
  if control_sd15_canny: control_versions+=['control_sd15_canny']
  if control_sd15_depth: control_versions+=['control_sd15_depth']
  if control_sd15_hed: control_versions+=['control_sd15_hed']
  if control_sd15_mlsd: control_versions+=['control_sd15_mlsd']
  if control_sd15_normal: control_versions+=['control_sd15_normal']
  if control_sd15_openpose: control_versions+=['control_sd15_openpose']
  if control_sd15_scribble: control_versions+=['control_sd15_scribble']
  if control_sd15_seg: control_versions+=['control_sd15_seg']
  if control_sd15_temporalnet: control_versions+=['control_sd15_temporalnet']
  if control_sd15_face: control_versions+=['control_sd15_face']
else: control_versions = [model_version]

if model_version in ["control_sd15_canny",  
                     "control_sd15_depth",
                     "control_sd15_hed", 
                     "control_sd15_mlsd", 
                     "control_sd15_normal", 
                     "control_sd15_openpose", 
                     "control_sd15_scribble", 
                     "control_sd15_seg", 'control_sd15_face', 'control_multi']:
  

  os.chdir(f"{root_dir}/ControlNet/")
  from annotator.util import resize_image, HWC3

  from cldm.model import create_model, load_state_dict
  os.chdir('../')    
  

  #if download model is on and model path is not found, download full controlnet
  if download_control_model:
    if not os.path.exists(model_path):
      print(f'Model not found at {model_path}')
      if model_version == 'control_multi': model_ver = control_versions[0]
      else: model_ver = model_version
      
      model_path = f"{controlnet_models_dir}/{model_ver}.pth"
      # model_path = f"{root_dir}/ControlNet/models/{model_ver}.pth"
      model_url = f'https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/{model_ver}.pth'
      if not os.path.exists(model_path) or force_download:
        try:
          pathlib.Path(model_path).unlink()
        except: pass
        print('Downloading full controlnet model... ')
        wget.download(model_url,  model_path)
        print('Downloaded full controlnet model.')
    #if model found, assume it's a working checkpoint, download small controlnet only:

    for model_ver in control_versions:
      small_controlnet_model_path = f"{controlnet_models_dir}/{model_ver}_small.safetensors"
      if use_small_controlnet and os.path.exists(model_path) and not os.path.exists(small_controlnet_model_path):
        print(f'Model found at {model_path}. Small model not found at {small_controlnet_model_path}.')
        
        controlnet_small_hf_url = 'https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_MODE-fp16.safetensors'
        small_url = controlnet_small_hf_url.replace('MODE', model_ver.split('_')[-1])

        if model_ver == 'control_sd15_temporalnet':
          small_url = 'https://huggingface.co/CiaraRowles/TemporalNet/resolve/main/diff_control_sd15_temporalnet_fp16.safetensors'
        if model_ver == 'control_sd15_face':
          small_url = 'https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace/resolve/main/control_v2p_sd15_mediapipe_face.safetensors'
          # small_url = 'https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace/resolve/main/control_mediapipe_face_sd15_v2.safetensors'

        if not os.path.exists(small_controlnet_model_path) or force_download:
          try:
            pathlib.Path(small_controlnet_model_path).unlink()
          except: pass
          print(f'Downloading small controlnet model from {small_url}... ')
          wget.download(small_url,  small_controlnet_model_path)
          print('Downloaded small controlnet model.')


      helper_names = control_helpers[model_ver]
      if helper_names is not None:
          if type(helper_names) == str: helper_names = [helper_names]
          for helper_name in helper_names:
            helper_model_url = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/'+helper_name
            helper_model_path = f'{root_dir}/ControlNet/annotator/ckpts/'+helper_name
            if not os.path.exists(helper_model_path) or force_download:
              try:
                pathlib.Path(helper_model_path).unlink()
              except: pass
              wget.download(helper_model_url, helper_model_path)
  assert os.path.exists(model_path), f'Model not found at path: {model_path}. Please enter a valid path to the checkpoint file.'
  
  if os.path.exists(small_controlnet_model_path):
    smallpath = small_controlnet_model_path
  else: 
    smallpath = None
  config = OmegaConf.load(f"{root_dir}/ControlNet/models/cldm_v15.yaml")
  sd_model = load_model_from_config(config=config, 
                                    ckpt=model_path, vae_ckpt=vae_ckpt, controlnet=smallpath, 
                                    verbose=True)

  #legacy
  # sd_model = create_model(f"{root_dir}/ControlNet/models/cldm_v15.yaml").cuda()
  # sd_model.load_state_dict(load_state_dict(model_path, location=load_to), strict=False)
  sd_model.cond_stage_model.half()
  sd_model.model.half()
  sd_model.control_model.half()
  sd_model.cuda()
  
  gc.collect()
else:
  assert os.path.exists(model_path), f'Model not found at path: {model_path}. Please enter a valid path to the checkpoint file.'
  if model_path.endswith('.pkl'):
      with open(model_path, 'rb') as f:
        sd_model = pickle.load(f).cuda().eval()
        if gpu == 'A100':
          sd_model = sd_model.float()
  else:
      config = OmegaConf.load(config_path)
      sd_model = load_model_from_config(config, model_path, vae_ckpt=vae_ckpt, verbose=True).cuda()

sys.path.append('./stablediffusion/')
from modules import prompt_parser, sd_hijack
# sd_model.first_stage_model = torch.compile(sd_model.first_stage_model)
# sd_model.model = torch.compile(sd_model.model)
if sd_model.parameterization == "v":
  model_wrap = K.external.CompVisVDenoiser(sd_model, quantize=quantize )
else: 
  model_wrap = K.external.CompVisDenoiser(sd_model, quantize=quantize)
sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
model_wrap_cfg = CFGDenoiser(model_wrap)
if model_version == 'v1_instructpix2pix':
  model_wrap_cfg = InstructPix2PixCFGDenoiser(model_wrap) 

#@markdown If you're having crashes (CPU out of memory errors) while running this cell on standard colab env, consider saving the model as pickle.\
#@markdown You can save the pickled model on your google drive and use it instead of the usual stable diffusion model.\
#@markdown To do that, run the notebook with a high-ram env, run all cells before and including this cell as well, and save pickle in the next cell. Then you can switch to a low-ram env and load the pickled model.