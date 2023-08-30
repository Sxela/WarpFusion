#@title CLIP guidance settings
#@markdown You can use clip guidance to further push style towards your text input.\
#@markdown Please note that enabling it (by using clip_guidance_scale>0) will greatly increase render times and VRAM usage.\
#@markdown For now it does 1 sample of the whole image per step (similar to 1 outer_cut in discodiffusion).

# clip_type, clip_pretrain = 'ViT-B-32-quickgelu', 'laion400m_e32'
# clip_type, clip_pretrain ='ViT-L-14', 'laion2b_s32b_b82k'
clip_type = 'ViT-H-14' #@param ['ViT-L-14','ViT-B-32-quickgelu', 'ViT-H-14']
if clip_type == 'ViT-H-14' : clip_pretrain = 'laion2b_s32b_b79k'
if clip_type == 'ViT-L-14' : clip_pretrain = 'laion2b_s32b_b82k'
if clip_type == 'ViT-B-32-quickgelu' : clip_pretrain = 'laion400m_e32'

clip_guidance_scale = 0 #@param {'type':"number"}
if clip_guidance_scale > 0: 
  clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_type, pretrained=clip_pretrain)
  _=clip_model.half().cuda().eval()
  clip_size = clip_model.visual.image_size
  for param in clip_model.parameters():
      param.requires_grad = False
else:
  try:
    del clip_model
    gc.collect()
  except: pass