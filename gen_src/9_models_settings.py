#@markdown ####**Models Settings:**
#@markdown #####temporarily off
diffusion_model = "stable_diffusion"
use_secondary_model = False
diffusion_sampling_mode = 'ddim'
##@markdown #####**Custom model:**
custom_path = ''

##@markdown #####**CLIP settings:**
ViTB32 = False
ViTB16 = False 
ViTL14 = False
ViTL14_336 = False 
RN101 = False 
RN50 = False 
RN50x4 = False 
RN50x16 = False
RN50x64 = False

## @markdown If you're having issues with model downloads, check this to compare SHA's:
check_model_SHA = False
use_checkpoint = True
model_256_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'
model_512_SHA = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
model_256_comics_SHA = 'f587fd6d2edb093701931e5083a13ab6b76b3f457b60efd1aa873d60ee3d6388'
model_secondary_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'

model_256_link = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
model_512_link = 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'
model_256_comics_link = 'https://github.com/Sxela/DiscoDiffusion-Warp/releases/download/v0.1.0/256x256_openai_comics_faces_by_alex_spirin_084000.pt'
model_secondary_link = 'https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth'

model_256_path = f'{model_path}/256x256_diffusion_uncond.pt'
model_512_path = f'{model_path}/512x512_diffusion_uncond_finetune_008100.pt'
model_256_comics_path = f'{model_path}/256x256_openai_comics_faces_by_alex_spirin_084000.pt'
model_secondary_path = f'{model_path}/secondary_model_imagenet_2.pth'

model_256_downloaded = False
model_512_downloaded = False
model_secondary_downloaded = False
model_256_comics_downloaded = False

# Download the diffusion model
if diffusion_model == '256x256_diffusion_uncond':
  if os.path.exists(model_256_path) and check_model_SHA:
    print('Checking 256 Diffusion File')
    with open(model_256_path,"rb") as f:
        bytes = f.read() 
        hash = hashlib.sha256(bytes).hexdigest();
    if hash == model_256_SHA:
      print('256 Model SHA matches')
      model_256_downloaded = True
    else: 
      print("256 Model SHA doesn't match, redownloading...")
      wget(model_256_link, model_path)
      model_256_downloaded = True
  elif os.path.exists(model_256_path) and not check_model_SHA or model_256_downloaded == True:
    print('256 Model already downloaded, check check_model_SHA if the file is corrupt')
  else:  
    wget(model_256_link, model_path)
    model_256_downloaded = True
elif diffusion_model == '512x512_diffusion_uncond_finetune_008100':
  if os.path.exists(model_512_path) and check_model_SHA:
    print('Checking 512 Diffusion File')
    with open(model_512_path,"rb") as f:
        bytes = f.read() 
        hash = hashlib.sha256(bytes).hexdigest();
    if hash == model_512_SHA:
      print('512 Model SHA matches')
      model_512_downloaded = True
    else:  
      print("512 Model SHA doesn't match, redownloading...")
      wget(model_512_link, model_path)
      model_512_downloaded = True
  elif os.path.exists(model_512_path) and not check_model_SHA or model_512_downloaded == True:
    print('512 Model already downloaded, check check_model_SHA if the file is corrupt')
  else:  
    wget(model_512_link, model_path)
    model_512_downloaded = True
elif diffusion_model == '256x256_openai_comics_faces_by_alex_spirin_084000':
  if os.path.exists(model_256_comics_path) and check_model_SHA:
    print('Checking 256 Comics Diffusion File')
    with open(model_256_comics_path,"rb") as f:
        bytes = f.read() 
        hash = hashlib.sha256(bytes).hexdigest();
    if hash == model_256_comics_SHA:
      print('256 Comics Model SHA matches')
      model_256_comics_downloaded = True
    else:  
      print("256 Comics SHA doesn't match, redownloading...")
      wget(model_256_comics_link, model_path)
      model_256_comics_downloaded = True
  elif os.path.exists(model_256_comics_path) and not check_model_SHA or model_256_comics_downloaded == True:
    print('256 Comics Model already downloaded, check check_model_SHA if the file is corrupt')
  else:  
    wget(model_256_comics_link, model_path)
    model_256_comics_downloaded = True


# Download the secondary diffusion model v2
if use_secondary_model == True:
  if os.path.exists(model_secondary_path) and check_model_SHA:
    print('Checking Secondary Diffusion File')
    with open(model_secondary_path,"rb") as f:
        bytes = f.read() 
        hash = hashlib.sha256(bytes).hexdigest();
    if hash == model_secondary_SHA:
      print('Secondary Model SHA matches')
      model_secondary_downloaded = True
    else:  
      print("Secondary Model SHA doesn't match, redownloading...")
      wget(model_secondary_link, model_path)
      model_secondary_downloaded = True
  elif os.path.exists(model_secondary_path) and not check_model_SHA or model_secondary_downloaded == True:
    print('Secondary Model already downloaded, check check_model_SHA if the file is corrupt')
  else:  
    wget(model_secondary_link, model_path)
    model_secondary_downloaded = True

model_config = model_and_diffusion_defaults()
if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 512,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': use_checkpoint,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
elif diffusion_model == '256x256_diffusion_uncond':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': use_checkpoint,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
elif diffusion_model == '256x256_openai_comics_faces_by_alex_spirin_084000':
    model_config.update({
          'attention_resolutions': '16',
          'class_cond': False,
          'diffusion_steps': 1000,
          'rescale_timesteps': True,
          'timestep_respacing': 'ddim100',
          'image_size': 256,
          'learn_sigma': True,
          'noise_schedule': 'linear',
          'num_channels': 128,
          'num_heads': 1,
          'num_res_blocks': 2,
          'use_checkpoint': use_checkpoint,
          'use_fp16': True,
          'use_scale_shift_norm': False,
      })

model_default = model_config['image_size']

if use_secondary_model:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(torch.load(f'{model_path}/secondary_model_imagenet_2.pth', map_location='cpu'))
    secondary_model.eval().requires_grad_(False).to(device)

clip_models = []
if ViTB32 is True: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)) 
if ViTB16 is True: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device) ) 
if ViTL14 is True: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device) ) 
if ViTL14_336 is True: clip_models.append(clip.load('ViT-L/14@336px', jit=False)[0].eval().requires_grad_(False).to(device) ) 
if RN50 is True: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x4 is True: clip_models.append(clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device)) 
if RN50x16 is True: clip_models.append(clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device)) 
if RN50x64 is True: clip_models.append(clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device)) 
if RN101 is True: clip_models.append(clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device)) 

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)

if diffusion_model == 'custom':
  model_config.update({
          'attention_resolutions': '16',
          'class_cond': False,
          'diffusion_steps': 1000,
          'rescale_timesteps': True,
          'timestep_respacing': 'ddim100',
          'image_size': 256,
          'learn_sigma': True,
          'noise_schedule': 'linear',
          'num_channels': 128,
          'num_heads': 1,
          'num_res_blocks': 2,
          'use_checkpoint': use_checkpoint,
          'use_fp16': True,
          'use_scale_shift_norm': False,
      })