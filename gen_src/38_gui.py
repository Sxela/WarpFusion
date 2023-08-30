#@title gui

#@markdown Load default settings
default_settings_path = '' #@param {'type':'string'}
settings_out = batchFolder+f"/settings"
from  ipywidgets import HTML, IntRangeSlider, jslink, Layout, VBox, HBox, Tab, Label, IntText, Dropdown, Text, Accordion, Button, Output, Textarea, FloatSlider, FloatText, Checkbox, SelectionSlider, Valid

def desc_widget(widget, desc, width=80, h=True):
    if isinstance(widget, Checkbox): return widget
    if isinstance(width, str):
        if width.endswith('%') or width.endswith('px'):
            layout = Layout(width=width)
    else: layout = Layout(width=f'{width}') 

    text = Label(desc, layout = layout, tooltip = widget.tooltip, description_tooltip = widget.description_tooltip)
    return HBox([text, widget]) if h else VBox([text, widget])

#try keep settings on occasional run cell 
try:
  user_comment= get_value('user_comment',guis)
  blend_json_schedules=get_value('blend_json_schedules',guis)
  VERBOSE=get_value('VERBOSE',guis)
  use_background_mask=get_value('use_background_mask',guis)
  invert_mask=get_value('invert_mask',guis)
  background=get_value('background',guis)
  background_source=get_value('background_source',guis)
  (mask_clip_low, mask_clip_high) = get_value('mask_clip',guis) 

  #turbo 
  turbo_mode=get_value('turbo_mode',guis)
  turbo_steps=get_value('turbo_steps',guis)
  colormatch_turbo=get_value('colormatch_turbo',guis)
  turbo_frame_skips_steps=get_value('turbo_frame_skips_steps',guis)
  soften_consistency_mask_for_turbo_frames=get_value('soften_consistency_mask_for_turbo_frames',guis)

  #warp
  flow_warp= get_value('flow_warp',guis)
  apply_mask_after_warp=get_value('apply_mask_after_warp',guis)
  warp_num_k=get_value('warp_num_k',guis)
  warp_forward=get_value('warp_forward',guis)
  warp_strength=get_value('warp_strength',guis)
  flow_override_map=eval(get_value('flow_override_map',guis))
  warp_mode=get_value('warp_mode',guis)
  warp_towards_init=get_value('warp_towards_init',guis)

  #cc
  check_consistency=get_value('check_consistency',guis)
  missed_consistency_weight=get_value('missed_consistency_weight',guis)
  overshoot_consistency_weight=get_value('overshoot_consistency_weight',guis)
  edges_consistency_weight=get_value('edges_consistency_weight',guis)
  consistency_blur=get_value('consistency_blur',guis)
  padding_ratio=get_value('padding_ratio',guis)
  padding_mode=get_value('padding_mode',guis)
  match_color_strength=get_value('match_color_strength',guis)
  soften_consistency_mask=get_value('soften_consistency_mask',guis)
  mask_result=get_value('mask_result',guis)
  use_patchmatch_inpaiting=get_value('use_patchmatch_inpaiting',guis)

  #diffusion
  text_prompts=eval(get_value('text_prompts',guis))
  negative_prompts=eval(get_value('negative_prompts',guis))
  depth_source=get_value('depth_source',guis)
  set_seed=get_value('set_seed',guis)
  clamp_grad=get_value('clamp_grad',guis)
  clamp_max=get_value('clamp_max',guis)
  sat_scale=get_value('sat_scale',guis)
  init_grad=get_value('init_grad',guis)
  grad_denoised=get_value('grad_denoised',guis)
  blend_latent_to_init=get_value('blend_latent_to_init',guis)
  fixed_code=get_value('fixed_code',guis)
  blend_code=get_value('blend_code',guis)
  normalize_code=get_value('normalize_code',guis)
  dynamic_thresh=get_value('dynamic_thresh',guis)
  sampler = get_value('sampler',guis)
  use_karras_noise = get_value('use_karras_noise',guis)
  inpainting_mask_weight = get_value('inpainting_mask_weight',guis)
  inverse_inpainting_mask = get_value('inverse_inpainting_mask',guis)
  inpainting_mask_source = get_value('mask_source',guis)

  #colormatch
  normalize_latent=get_value('normalize_latent',guis)
  normalize_latent_offset=get_value('normalize_latent_offset',guis)
  latent_fixed_mean=eval(str(get_value('latent_fixed_mean',guis)))
  latent_fixed_std=eval(str(get_value('latent_fixed_std',guis)))
  latent_norm_4d=get_value('latent_norm_4d',guis)
  colormatch_frame=get_value('colormatch_frame',guis)
  color_match_frame_str=get_value('color_match_frame_str',guis)
  colormatch_offset=get_value('colormatch_offset',guis)
  colormatch_method=get_value('colormatch_method',guis)
  colormatch_regrain=get_value('colormatch_regrain',guis)
  colormatch_after=get_value('colormatch_after',guis)
  image_prompts = {}

  fixed_seed = get_value('fixed_seed',guis)

  rec_cfg = get_value('rec_cfg',guis)
  rec_steps_pct = get_value('rec_steps_pct',guis)
  rec_prompts = eval(get_value('rec_prompts',guis))
  rec_randomness = get_value('rec_randomness',guis)
  use_predicted_noise = get_value('use_predicted_noise',guis)

  controlnet_preprocess = get_value('controlnet_preprocess',guis)
  detect_resolution  = get_value('detect_resolution',guis)
  bg_threshold = get_value('bg_threshold',guis)
  low_threshold = get_value('low_threshold',guis)
  high_threshold = get_value('high_threshold',guis)
  value_threshold = get_value('value_threshold',guis)
  distance_threshold = get_value('distance_threshold',guis)
  temporalnet_source = get_value('temporalnet_source',guis)
  temporalnet_skip_1st_frame = get_value('temporalnet_skip_1st_frame',guis)
  controlnet_multimodel_mode = get_value('controlnet_multimodel_mode',guis)
  max_faces = get_value('max_faces',guis)

  do_softcap = get_value('do_softcap',guis)
  softcap_thresh = get_value('softcap_thresh',guis)
  softcap_q = get_value('softcap_q',guis)

  masked_guidance = get_value('masked_guidance',guis)
  mask_callback = get_value('mask_callback',guis)
except: 
  pass

gui_misc = {
    "user_comment": Textarea(value=user_comment,layout=Layout(width=f'80%'),  description = 'user_comment:',  description_tooltip = 'Enter a comment to differentiate between save files.'),
    "blend_json_schedules": Checkbox(value=blend_json_schedules, description='blend_json_schedules',indent=True, description_tooltip = 'Smooth values between keyframes.', tooltip = 'Smooth values between keyframes.'),
    "VERBOSE": Checkbox(value=VERBOSE,description='VERBOSE',indent=True, description_tooltip = 'Print all logs'),
    "offload_model": Checkbox(value=offload_model,description='offload_model',indent=True, description_tooltip = 'Offload unused models to CPU and back to GPU to save VRAM. May reduce speed.'),
    "do_softcap": Checkbox(value=do_softcap,description='do_softcap',indent=True, description_tooltip = 'Softly clamp latent excessive values. Reduces feedback loop effect a bit.'),
    "softcap_thresh":FloatSlider(value=softcap_thresh, min=0, max=1, step=0.05, description='softcap_thresh:', readout=True, readout_format='.1f', description_tooltip='Scale down absolute values above that threshold (latents are being clamped at [-1:1] range, so 0.9 will downscale values above 0.9 to fit into that range, [-1.5:1.5] will be scaled to [-1:1], but only absolute values over 0.9 will be affected'),
    "softcap_q":FloatSlider(value=softcap_q, min=0, max=1, step=0.05, description='softcap_q:', readout=True, readout_format='.1f', description_tooltip='Percentile to downscale. 1-downscle full range with outliers, 0.9 - downscale only 90%  values above thresh, clamp 10%'),

}

gui_mask = {
    "use_background_mask":Checkbox(value=use_background_mask,description='use_background_mask',indent=True, description_tooltip='Enable masking. In order to use it, you have to either extract or provide an existing mask in Video Masking cell.\n'),
    "invert_mask":Checkbox(value=invert_mask,description='invert_mask',indent=True, description_tooltip='Inverts the mask, allowing to process either backgroung or characters, depending on your mask.'),
    "background": Dropdown(description='background', 
                           options = ['image', 'color', 'init_video'], value = background, 
                           description_tooltip='Background type. Image - uses static image specified in background_source, color - uses fixed color specified in background_source, init_video - uses raw init video for masked areas.'), 
    "background_source": Text(value=background_source, description = 'background_source', description_tooltip='Specify image path or color name of hash.'),
    "apply_mask_after_warp": Checkbox(value=apply_mask_after_warp,description='apply_mask_after_warp',indent=True, description_tooltip='On to reduce ghosting. Apply mask after warping and blending warped image with current raw frame. If off, only current frame will be masked, previous frame will be warped and blended wuth masked current frame.'),
    "mask_clip" : IntRangeSlider(
      value=mask_clip,
      min=0,
      max=255,
      step=1,
      description='Mask clipping:',
      description_tooltip='Values below the selected range will be treated as black mask, values above - as white.',
      disabled=False,
      continuous_update=False,
      orientation='horizontal',
      readout=True)
    
}

gui_turbo = {
    "turbo_mode":Checkbox(value=turbo_mode,description='turbo_mode',indent=True, description_tooltip='Turbo mode skips diffusion process on turbo_steps number of frames. Frames are still being warped and blended. Speeds up the render at the cost of possible trails an ghosting.' ),
    "turbo_steps": IntText(value = turbo_steps, description='turbo_steps:', description_tooltip='Number of turbo frames'),
    "colormatch_turbo":Checkbox(value=colormatch_turbo,description='colormatch_turbo',indent=True, description_tooltip='Apply frame color matching during turbo frames. May increease rendering speed, but may add minor flickering.'),
    "turbo_frame_skips_steps" :  SelectionSlider(description='turbo_frame_skips_steps', 
                                                 options = ['70%','75%','80%','85%', '80%', '95%', '100% (don`t diffuse turbo frames, fastest)'], value = '100% (don`t diffuse turbo frames, fastest)', description_tooltip='Skip steps for turbo frames. Select 100% to skip diffusion rendering for turbo frames completely.'),
    "soften_consistency_mask_for_turbo_frames": FloatSlider(value=soften_consistency_mask_for_turbo_frames, min=0, max=1, step=0.05, description='soften_consistency_mask_for_turbo_frames:', readout=True, readout_format='.1f', description_tooltip='Clips the consistency mask, reducing it`s effect'),
  
}

gui_warp = {
    "flow_warp":Checkbox(value=flow_warp,description='flow_warp',indent=True, description_tooltip='Blend current raw init video frame with previously stylised frame with respect to consistency mask. 0 - raw frame, 1 - stylized frame'),
    
    "flow_blend_schedule" : Textarea(value=str(flow_blend_schedule),layout=Layout(width=f'80%'),  description = 'flow_blend_schedule:',  description_tooltip='Blend current raw init video frame with previously stylised frame with respect to consistency mask. 0 - raw frame, 1 - stylized frame'),
    "warp_num_k": IntText(value = warp_num_k, description='warp_num_k:', description_tooltip='Nubmer of clusters in forward-warp mode. The more - the smoother is the motion. Lower values move larger chunks of image at a time.'),
    "warp_forward": Checkbox(value=warp_forward,description='warp_forward',indent=True,  description_tooltip='Experimental. Enable patch-based flow warping. Groups pixels by motion direction and moves them together, instead of moving individual pixels.'),
    # "warp_interp": Textarea(value='Image.LANCZOS',layout=Layout(width=f'80%'),  description = 'warp_interp:'),
    "warp_strength": FloatText(value = warp_strength, description='warp_strength:', description_tooltip='Experimental. Motion vector multiplier. Provides a glitchy effect.'),
    "flow_override_map":  Textarea(value=str(flow_override_map),layout=Layout(width=f'80%'),  description = 'flow_override_map:', description_tooltip='Experimental. Motion vector maps mixer. Allows changing frame-motion vetor indexes or repeating motion, provides a glitchy effect.'),
    "warp_mode": Dropdown(description='warp_mode', options = ['use_latent', 'use_image'],
                          value = warp_mode, description_tooltip='Experimental. Apply warp to latent vector. May get really blurry, but reduces feedback loop effect for slow movement'), 
    "warp_towards_init": Dropdown(description='warp_towards_init',
                                  options = ['stylized', 'off'] , value = warp_towards_init, description_tooltip='Experimental. After a frame is stylized, computes the difference between output and input for that frame, and warps the output back to input, preserving its shape.'),
    "padding_ratio": FloatSlider(value=padding_ratio, min=0, max=1, step=0.05, description='padding_ratio:', readout=True, readout_format='.1f', description_tooltip='Amount of padding. Padding is used to avoid black edges when the camera is moving out of the frame.'),
    "padding_mode": Dropdown(description='padding_mode', options = ['reflect','edge','wrap'],
                             value = padding_mode),
}

# warp_interp = Image.LANCZOS

gui_consistency = {
    "check_consistency":Checkbox(value=check_consistency,description='check_consistency',indent=True, description_tooltip='Enables consistency checking (CC). CC is used to avoid ghosting and trails, that appear due to lack of information while warping frames. It allows replacing motion edges, frame borders, incorrectly moved areas with raw init frame data.'),
    "missed_consistency_weight":FloatSlider(value=missed_consistency_weight, min=0, max=1, step=0.05, description='missed_consistency_weight:', readout=True, readout_format='.1f', description_tooltip='Multiplier for incorrectly predicted\moved areas. For example, if an object moves and background appears behind it. We can predict what to put in that spot, so we can either duplicate the object, resulting in trail, or use init video data for that region.'),
    "overshoot_consistency_weight":FloatSlider(value=overshoot_consistency_weight, min=0, max=1, step=0.05, description='overshoot_consistency_weight:', readout=True, readout_format='.1f', description_tooltip='Multiplier for areas that appeared out of the frame. We can either leave them black or use raw init video.'),
    "edges_consistency_weight":FloatSlider(value=edges_consistency_weight, min=0, max=1, step=0.05, description='edges_consistency_weight:', readout=True, readout_format='.1f', description_tooltip='Multiplier for motion edges. Moving objects are most likely to leave trails, this option together with missed consistency weight helps prevent that, but in a more subtle manner.'),
    "soften_consistency_mask" :  FloatSlider(value=soften_consistency_mask, min=0, max=1, step=0.05, description='soften_consistency_mask:', readout=True, readout_format='.1f'),
    "consistency_blur": FloatText(value = consistency_blur, description='consistency_blur:'),
    "barely used": Label(' '),
    "match_color_strength" : FloatSlider(value=match_color_strength, min=0, max=1, step=0.05, description='match_color_strength:', readout=True, readout_format='.1f', description_tooltip='Enables colormathing raw init video pixls in inconsistent areas only to the stylized frame. May reduce flickering for inconsistent areas.'),
    "mask_result": Checkbox(value=mask_result,description='mask_result',indent=True, description_tooltip='Stylizes only inconsistent areas. Takes consistent areas from the previous frame.'),
    "use_patchmatch_inpaiting": FloatSlider(value=use_patchmatch_inpaiting, min=0, max=1, step=0.05, description='use_patchmatch_inpaiting:', readout=True, readout_format='.1f', description_tooltip='Uses patchmatch inapinting for inconsistent areas. Is slow.'),
}

gui_diffusion = {
    "use_karras_noise":Checkbox(value=use_karras_noise,description='use_karras_noise',indent=True, description_tooltip='Enable for samplers that have K at their name`s end.'),
    "sampler": Dropdown(description='sampler',options= [('sample_euler', sample_euler), 
                                  ('sample_euler_ancestral',sample_euler_ancestral), 
                                  ('sample_heun',sample_heun),
                                  ('sample_dpm_2', sample_dpm_2),
                                  ('sample_dpm_2_ancestral',sample_dpm_2_ancestral),
                                  ('sample_lms', sample_lms),
                                  ('sample_dpm_fast', sample_dpm_fast),
                                  ('sample_dpm_adaptive',sample_dpm_adaptive),
                                  ('sample_dpmpp_2s_ancestral', sample_dpmpp_2s_ancestral),
                                  ('sample_dpmpp_sde', sample_dpmpp_sde),
                                  ('sample_dpmpp_2m', sample_dpmpp_2m)], value = sampler),
    "text_prompts" : Textarea(value=str(text_prompts),layout=Layout(width=f'80%'),  description = 'Prompt:'),
    "negative_prompts" :  Textarea(value=str(negative_prompts), layout=Layout(width=f'80%'), description = 'Negative Prompt:'),
    "depth_source":Dropdown(description='depth_source', options = ['init', 'stylized','cond_video'] , 
                            value = depth_source, description_tooltip='Depth map source for depth model. It can either take raw init video frame or previously stylized frame.'), 
    "inpainting_mask_source":Dropdown(description='inpainting_mask_source', options = ['none', 'consistency_mask', 'cond_video'] , 
                           value = inpainting_mask_source, description_tooltip='Inpainting model mask source. none - full white mask (inpaint whole image), consistency_mask - inpaint inconsistent areas only'),
    "inverse_inpainting_mask":Checkbox(value=inverse_inpainting_mask,description='inverse_inpainting_mask',indent=True, description_tooltip='Inverse inpainting mask'),
    "inpainting_mask_weight":FloatSlider(value=inpainting_mask_weight, min=0, max=1, step=0.05, description='inpainting_mask_weight:', readout=True, readout_format='.1f', 
                                         description_tooltip= 'Inpainting mask weight. 0 - Disables inpainting mask.'),
    "set_seed": IntText(value = set_seed, description='set_seed:', description_tooltip='Seed. Use -1 for random.'),
    "clamp_grad":Checkbox(value=clamp_grad,description='clamp_grad',indent=True, description_tooltip='Enable limiting the effect of external conditioning per diffusion step'),
    "clamp_max": FloatText(value = clamp_max, description='clamp_max:',description_tooltip='limit the effect of external conditioning per diffusion step'),
    "latent_scale_schedule":Textarea(value=str(latent_scale_schedule),layout=Layout(width=f'80%'),  description = 'latent_scale_schedule:', description_tooltip='Latents scale defines how much minimize difference between output and input stylized image in latent space.'),
    "init_scale_schedule": Textarea(value=str(init_scale_schedule),layout=Layout(width=f'80%'),  description = 'init_scale_schedule:', description_tooltip='Init scale defines how much minimize difference between output and input stylized image in RGB space.'),
    "sat_scale": FloatText(value = sat_scale, description='sat_scale:', description_tooltip='Saturation scale limits oversaturation.'),
    "init_grad": Checkbox(value=init_grad,description='init_grad',indent=True,  description_tooltip='On - compare output to real frame, Off - to stylized frame'),
    "grad_denoised" : Checkbox(value=grad_denoised,description='grad_denoised',indent=True, description_tooltip='Fastest, On by default, calculate gradients with respect to denoised image instead of input image per diffusion step.' ),
    "steps_schedule" : Textarea(value=str(steps_schedule),layout=Layout(width=f'80%'),  description = 'steps_schedule:', 
                               description_tooltip= 'Total diffusion steps schedule. Use list format like [50,70], where each element corresponds to a frame, last element being repeated forever, or dictionary like {0:50, 20:70} format to specify keyframes only.'),
    "style_strength_schedule" : Textarea(value=str(style_strength_schedule),layout=Layout(width=f'80%'),  description = 'style_strength_schedule:',
                                          description_tooltip= 'Diffusion (style) strength. Actual number of diffusion steps taken (at 50 steps with 0.3 or 30% style strength you get 15 steps, which also means 35 0r 70% skipped steps). Inverse of skep steps. Use list format like [0.5,0.35], where each element corresponds to a frame, last element being repeated forever, or dictionary like {0:0.5, 20:0.35} format to specify keyframes only.'),
    "cfg_scale_schedule": Textarea(value=str(cfg_scale_schedule),layout=Layout(width=f'80%'),  description = 'cfg_scale_schedule:', description_tooltip= 'Guidance towards text prompt. 7 is a good starting value, 1 is off (text prompt has no effect).'),
    "image_scale_schedule": Textarea(value=str(image_scale_schedule),layout=Layout(width=f'80%'),  description = 'image_scale_schedule:', description_tooltip= 'Only used with InstructPix2Pix Model. Guidance towards text prompt. 1.5 is a good starting value'),
    "blend_latent_to_init": FloatSlider(value=blend_latent_to_init, min=0, max=1, step=0.05, description='blend_latent_to_init:', readout=True, readout_format='.1f', description_tooltip = 'Blend latent vector with raw init'),
    # "use_karras_noise": Checkbox(value=False,description='use_karras_noise',indent=True),
    # "end_karras_ramp_early": Checkbox(value=False,description='end_karras_ramp_early',indent=True),
    "fixed_seed": Checkbox(value=fixed_seed,description='fixed_seed',indent=True, description_tooltip= 'Fixed seed.'),
    "fixed_code":  Checkbox(value=fixed_code,description='fixed_code',indent=True, description_tooltip= 'Fixed seed analog. Fixes diffusion noise.'),
    "blend_code": FloatSlider(value=blend_code, min=0, max=1, step=0.05, description='blend_code:', readout=True, readout_format='.1f', description_tooltip= 'Fixed seed amount/effect strength.'),
    "normalize_code":Checkbox(value=normalize_code,description='normalize_code',indent=True, description_tooltip= 'Whether to normalize the noise after adding fixed seed.'),
    "dynamic_thresh": FloatText(value = dynamic_thresh, description='dynamic_thresh:', description_tooltip= 'Limit diffusion model prediction output. Lower values may introduce clamping/feedback effect'),
    "use_predicted_noise":Checkbox(value=use_predicted_noise,description='use_predicted_noise',indent=True, description_tooltip='Reconstruct initial noise from init / stylized image.'),
    "rec_prompts" : Textarea(value=str(rec_prompts),layout=Layout(width=f'80%'),  description = 'Rec Prompt:'),
    "rec_randomness":   FloatSlider(value=rec_randomness, min=0, max=1, step=0.05, description='rec_randomness:', readout=True, readout_format='.1f', description_tooltip= 'Reconstructed noise randomness. 0 - reconstructed noise only. 1 - random noise.'),
    "rec_cfg": FloatText(value = rec_cfg, description='rec_cfg:', description_tooltip= 'CFG scale for noise reconstruction. 1-1.9 are the best values.'),
    "rec_source": Dropdown(description='rec_source', options = ['init', 'stylized'] , 
                            value = rec_source, description_tooltip='Source for noise reconstruction. Either raw init frame or stylized frame.'), 
    "rec_steps_pct":FloatSlider(value=rec_steps_pct, min=0, max=1, step=0.05, description='rec_steps_pct:', readout=True, readout_format='.2f', description_tooltip= 'Reconstructed noise steps in relation to total steps. 1 = 100% steps.'),

    "masked_guidance":Checkbox(value=masked_guidance,description='masked_guidance',indent=True, 
                               description_tooltip= 'Use mask for init/latent guidance to ignore inconsistencies and only guide based on the consistent areas.'),
    "mask_callback": FloatSlider(value=mask_callback, min=0, max=1, step=0.05, 
                                 description='mask_callback:', readout=True, readout_format='.2f', description_tooltip= '0 - off. 0.5-0.7 are good values. Make inconsistent area passes only before this % of actual steps, then diffuse whole image.'),

    
}
gui_colormatch = {
    "normalize_latent": Dropdown(description='normalize_latent',
                                 options = ['off', 'user_defined', 'color_video', 'color_video_offset',
    'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset'], value =normalize_latent ,description_tooltip= 'Normalize latent to prevent it from overflowing. User defined: use fixed input values (latent_fixed_*) Stylized/init frame - match towards stylized/init frame with a fixed number (specified in the offset field below). Stylized\init frame offset - match to a frame with a number = current frame - offset (specified in the offset filed below).'),
    "normalize_latent_offset":IntText(value = normalize_latent_offset, description='normalize_latent_offset:', description_tooltip= 'Offset from current frame number for *_frame_offset mode, or fixed frame number for *frame mode.'),
    "latent_fixed_mean": FloatText(value = latent_fixed_mean, description='latent_fixed_mean:', description_tooltip= 'User defined mean value for normalize_latent=user_Defined mode'),
    "latent_fixed_std": FloatText(value = latent_fixed_std, description='latent_fixed_std:', description_tooltip= 'User defined standard deviation value for normalize_latent=user_Defined mode'),
    "latent_norm_4d": Checkbox(value=latent_norm_4d,description='latent_norm_4d',indent=True, description_tooltip= 'Normalize on a per-channel basis (on by default)'),
    "colormatch_frame": Dropdown(description='colormatch_frame', options = ['off', 'stylized_frame', 'color_video', 'color_video_offset', 'init_frame', 'stylized_frame_offset', 'init_frame_offset'], 
                                 value = colormatch_frame,
                                 description_tooltip= 'Match frame colors to prevent it from overflowing.  Stylized/init frame - match towards stylized/init frame with a fixed number (specified in the offset filed below). Stylized\init frame offset - match to a frame with a number = current frame - offset (specified in the offset field below).'),
    "color_match_frame_str": FloatText(value = color_match_frame_str, description='color_match_frame_str:', description_tooltip= 'Colormatching strength. 0 - no colormatching effect.'),
    "colormatch_offset":IntText(value =colormatch_offset, description='colormatch_offset:', description_tooltip= 'Offset from current frame number for *_frame_offset mode, or fixed frame number for *frame mode.'),
    "colormatch_method": Dropdown(description='colormatch_method', options = ['LAB', 'PDF', 'mean'], value =colormatch_method ),
    # "colormatch_regrain": Checkbox(value=False,description='colormatch_regrain',indent=True),
    "colormatch_after":Checkbox(value=colormatch_after,description='colormatch_after',indent=True, description_tooltip= 'On - Colormatch output frames when saving to disk, may differ from the preview. Off - colormatch before stylizing.'),
    
}

gui_controlnet = {
    "controlnet_preprocess": Checkbox(value=controlnet_preprocess,description='controlnet_preprocess',indent=True, 
                                      description_tooltip= 'preprocess input conditioning image for controlnet. If false, use raw conditioning as input to the model without detection/preprocessing.'),
    "detect_resolution":IntText(value = detect_resolution, description='detect_resolution:', description_tooltip= 'Control net conditioning image resolution. The size of the image passed into controlnet preprocessors. Suggest keeping this as high as you can fit into your VRAM for more details.'),
    "bg_threshold":FloatText(value = bg_threshold, description='bg_threshold:', description_tooltip='Control net depth/normal bg cutoff threshold'),
    "low_threshold":IntText(value = low_threshold, description='low_threshold:', description_tooltip= 'Control net canny filter parameters'), 
    "high_threshold":IntText(value = high_threshold, description='high_threshold:', description_tooltip= 'Control net canny filter parameters'),
    "value_threshold":FloatText(value = value_threshold, description='value_threshold:', description_tooltip='Control net mlsd filter parameters'),
    "distance_threshold":FloatText(value = distance_threshold, description='distance_threshold:', description_tooltip='Control net mlsd filter parameters'),
    "temporalnet_source":Dropdown(description ='temporalnet_source', options = ['init', 'stylized'] , 
                            value = temporalnet_source, description_tooltip='Temporalnet guidance source. Previous init or previous stylized frame'),
    "temporalnet_skip_1st_frame": Checkbox(value = temporalnet_skip_1st_frame,description='temporalnet_skip_1st_frame',indent=True, 
                                      description_tooltip='Skip temporalnet for 1st frame (if not skipped, will use raw init for guidance'),
    "controlnet_multimodel_mode":Dropdown(description='controlnet_multimodel_mode', options = ['internal','external'], value =controlnet_multimodel_mode, description_tooltip='internal - sums controlnet values before feeding those into diffusion model, external - sum outputs of differnet contolnets after passing through diffusion model. external seems slower but smoother.' ), 
    "max_faces":IntText(value = max_faces, description='max_faces:', description_tooltip= 'Max faces to detect. Control net face parameters'),
}

colormatch_regrain = False

guis = [gui_diffusion, gui_controlnet, gui_warp, gui_consistency, gui_turbo, gui_mask, gui_colormatch, gui_misc]

class FilePath(HBox):
    def __init__(self,  **kwargs):
        self.model_path = Text(value='',  continuous_update = True,**kwargs)
        self.path_checker = Valid(
        value=False, layout=Layout(width='2000px')
        )
        
        self.model_path.observe(self.on_change)
        super().__init__([self.model_path, self.path_checker])
    
    def __getattr__(self, attr):
        if attr == 'value':
            return self.model_path.value
        else:
            return super.__getattr__(attr)
    
    def on_change(self, change):
        if change['name'] == 'value':
            if os.path.exists(change['new']):
                self.path_checker.value = True
                self.path_checker.description = ''
            else: 
                self.path_checker.value = False
                self.path_checker.description = 'The file does not exist. Please specify the correct path.'


def add_labels_dict(gui):
    style = {'description_width': '250px' }
    layout = Layout(width='500px')
    gui_labels = {}
    for key in gui.keys():
        # if not isinstance( gui [key],Checkbox ):
        gui [key].style = style
        if not isinstance(gui[key], Textarea):
            gui [key].layout = layout
        
        box = gui[key]
        gui_labels[key] = box
 
    return gui_labels

def add_labels_dict(gui):
    style = {'description_width': '250px' }
    layout = Layout(width='500px')
    gui_labels = {}
    for key in gui.keys():
        gui[key].style = style
        # temp = gui[key]
        # temp.observe(dump_gui())
        # gui[key] = temp
        if not isinstance(gui[key], Textarea) and not isinstance( gui[key],Checkbox ):
            gui[key].layout = layout
        if isinstance( gui[key],Checkbox ):
            html_label = HTML(
                description=gui[key].description,
                description_tooltip=gui[key].description_tooltip,  style={'description_width': 'initial' },
                layout = Layout(position='relative', left='-25px'))
            gui_labels[key] = HBox([gui[key],html_label])
            gui[key].description = ''
            # gui_labels[key] = gui[key]

        else:

            gui_labels[key] = gui[key]
        # gui_labels[key].observe(print('smth changed', time.time()))
 
    return gui_labels


gui_diffusion_label, gui_controlnet_label, gui_warp_label, gui_consistency_label, gui_turbo_label, gui_mask_label, gui_colormatch_label, gui_misc_label = [add_labels_dict(o) for o in guis]

cond_keys = ['latent_scale_schedule','init_scale_schedule','clamp_grad','clamp_max','init_grad','grad_denoised','masked_guidance' ]
conditioning_w = Accordion([VBox([gui_diffusion_label[o] for o in cond_keys])])
conditioning_w.set_title(0, 'External Conditioning...')

seed_keys = ['set_seed', 'fixed_seed', 'fixed_code', 'blend_code', 'normalize_code']
seed_w = Accordion([VBox([gui_diffusion_label[o] for o in seed_keys])])
seed_w.set_title(0, 'Seed...')

rec_keys = ['use_predicted_noise','rec_prompts','rec_cfg','rec_randomness', 'rec_source', 'rec_steps_pct']
rec_w = Accordion([VBox([gui_diffusion_label[o] for o in rec_keys])])
rec_w.set_title(0, 'Reconstructed noise...')

prompt_keys = ['text_prompts', 'negative_prompts',
'steps_schedule', 'style_strength_schedule', 
'cfg_scale_schedule', 'blend_latent_to_init', 'dynamic_thresh',  
'depth_source', 'mask_callback']
if model_version == 'v1_instructpix2pix':
  prompt_keys.append('image_scale_schedule')
if  model_version == 'v1_inpainting':
  prompt_keys+=['inpainting_mask_source', 'inverse_inpainting_mask', 'inpainting_mask_weight']
prompt_keys = [o for o in prompt_keys if o not in seed_keys+cond_keys]
prompt_w = [gui_diffusion_label[o] for o in prompt_keys]

gui_diffusion_list = [*prompt_w, gui_diffusion_label['sampler'], 
gui_diffusion_label['use_karras_noise'], conditioning_w, seed_w, rec_w]

control_annotator_keys = ['controlnet_preprocess','detect_resolution','bg_threshold','low_threshold','high_threshold','value_threshold',
                          'distance_threshold', 'max_faces']
control_annotator_w = Accordion([VBox([gui_controlnet_label[o] for o in control_annotator_keys])])
control_annotator_w.set_title(0, 'Controlnet annotator settings...')
control_keys = ['temporalnet_source', 'temporalnet_skip_1st_frame', 'controlnet_multimodel_mode']
control_w = [gui_controlnet_label[o] for o in control_keys]
gui_control_list = [control_annotator_w, *control_w]

#misc
misc_keys = ["user_comment","blend_json_schedules","VERBOSE","offload_model"]
misc_w = [gui_misc_label[o] for o in misc_keys]

softcap_keys = ['do_softcap','softcap_thresh','softcap_q']
softcap_w = Accordion([VBox([gui_misc_label[o] for o in softcap_keys])])
softcap_w.set_title(0, 'Softcap settings...')

load_settings_btn = Button(description='Load settings')
def btn_eventhandler(obj):
  load_settings(load_settings_path.value)
load_settings_btn.on_click(btn_eventhandler)
load_settings_path = FilePath(placeholder='Please specify the path to the settings file to load.', description_tooltip='Please specify the path to the settings file to load.')
settings_w = Accordion([VBox([load_settings_path, load_settings_btn])])
settings_w.set_title(0, 'Load settings...')
gui_misc_list = [*misc_w, softcap_w, settings_w]

guis_labels_source = [gui_diffusion_list]
guis_titles_source = ['diffusion']
if 'control' in model_version:
  guis_labels_source += [gui_control_list]
  guis_titles_source += ['controlnet']
  
guis_labels_source += [gui_warp_label, gui_consistency_label, 
gui_turbo_label, gui_mask_label, gui_colormatch_label, gui_misc_list]
guis_titles_source += ['warp', 'consistency', 'turbo', 'mask', 'colormatch', 'misc']

guis_labels = [VBox([*o.values()]) if isinstance(o, dict) else VBox(o) for o in guis_labels_source]

app = Tab(guis_labels)
for i,title in enumerate(guis_titles_source):
    app.set_title(i, title)

def get_value(key, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            return obj[key].value
        else: 
            for o in obj.keys():
                res = get_value(key, obj[o])
                if res is not None: return res
    if isinstance(obj, list):
        for o in obj:
            res = get_value(key, o)
            if res is not None: return res
    return None

def set_value(key, value, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            obj[key].value = value
        else: 
            for o in obj.keys():
                set_value(key, value, obj[o])
                 
    if isinstance(obj, list):
        for o in obj:
            set_value(key, value, o)

import json
def infer_settings_path(path):
    default_settings_path = path
    if default_settings_path == '-1':
      settings_files = sorted(glob(os.path.join(settings_out, '*.txt')))
      if len(settings_files)>0:
        default_settings_path = settings_files[-1]
      else:
        print('Skipping load latest run settings: no settings files found.')
        return ''
    else:
      try: 
        if type(eval(default_settings_path)) == int:
          files = sorted(glob(os.path.join(settings_out, '*.txt')))
          for f in files: 
            if f'({default_settings_path})' in f:
              default_settings_path = f
      except: pass

    path = default_settings_path
    return path
    
def load_settings(path):
    path = infer_settings_path(path)

    global guis, load_settings_path, output
    if not os.path.exists(path):
        output.clear_output()
        print('Please specify a valid path to a settings file.')
        return 

    print('Loading settings from: ', default_settings_path)
    with open(path, 'rb') as f:
        settings = json.load(f)

    for key in settings: 
        try:
            val = settings[key]
            if key == 'normalize_latent' and val == 'first_latent':
              val = 'init_frame'
              settings['normalize_latent_offset'] = 0
            if key == 'turbo_frame_skips_steps' and val == None:
                val = '100% (don`t diffuse turbo frames, fastest)'
            if key == 'seed':
                key = 'set_seed'
            if key == 'grad_denoised ':
                key = 'grad_denoised'
            if type(val) in [dict,list]:
                if type(val) in [dict]:
                  temp = {}
                  for k in val.keys():
                    temp[int(k)] = val[k]
                  val = temp
                val = json.dumps(val)
            if key == 'mask_clip':
              val = eval(val)
            if key == 'sampler':
              val = getattr(K.sampling, val) 
            
            set_value(key, val, guis)
        except Exception as e: 
            print(key), print(settings[key] )
            print(e)
    output.clear_output()
    print('Successfully loaded settings from ', path )

def dump_gui():
  print('smth changed', time.time())
#   with open('gui.pkl', 'wb') as f:
#     pickle.dump(app.get_state(), f)

# app.observe(dump_gui())
output = Output()
if default_settings_path != '':
  load_settings(default_settings_path)

display.display(app)