#@title ##Warp Turbo Smooth Settings
#@markdown Skip steps for turbo frames. Select 100% to skip diffusion rendering for turbo frames completely.
turbo_frame_skips_steps = '100% (don`t diffuse turbo frames, fastest)' #@param ['70%','75%','80%','85%', '90%', '95%', '100% (don`t diffuse turbo frames, fastest)']

if turbo_frame_skips_steps == '100% (don`t diffuse turbo frames, fastest)':
  turbo_frame_skips_steps = None 
else: 
  turbo_frame_skips_steps = int(turbo_frame_skips_steps.split('%')[0])/100
#None - disable and use default skip steps

#@markdown ###Consistency mask postprocessing
#@markdown ####Soften consistency mask
#@markdown Lower values mean less stylized frames and more raw video input in areas with fast movement, but fewer trails add ghosting.\
#@markdown Gives glitchy datamoshing look.\
#@markdown Higher values keep stylized frames, but add trails and ghosting.

soften_consistency_mask = 0 #@param {type:"slider", min:0, max:1, step:0.1}
forward_weights_clip = soften_consistency_mask
#0 behaves like consistency on, 1 - off, in between - blends
soften_consistency_mask_for_turbo_frames = 0 #@param {type:"slider", min:0, max:1, step:0.1}
forward_weights_clip_turbo_step = soften_consistency_mask_for_turbo_frames
#None - disable and use forward_weights_clip for turbo frames, 0 behaves like consistency on, 1 - off, in between - blends
#@markdown ####Blur consistency mask.
#@markdown Softens transition between raw video init and stylized frames in occluded areas.
consistency_blur = 1 #@param


# disable_cc_for_turbo_frames = False #@param {"type":"boolean"} 
#disable consistency for turbo frames, the same as forward_weights_clip_turbo_step = 1, but a bit faster

#@markdown ###Frame padding
#@markdown Increase padding if you have a shaky\moving camera footage and are getting black borders.

padding_ratio = 0.2 #@param {type:"slider", min:0, max:1, step:0.1}
#relative to image size, in range 0-1
padding_mode = 'reflect' #@param ['reflect','edge','wrap']


#safeguard the params
if turbo_frame_skips_steps is not None:
  turbo_frame_skips_steps = min(max(0,turbo_frame_skips_steps),1)
forward_weights_clip = min(max(0,forward_weights_clip),1)
if forward_weights_clip_turbo_step is not None:
  forward_weights_clip_turbo_step = min(max(0,forward_weights_clip_turbo_step),1)
padding_ratio = min(max(0,padding_ratio),1)
##@markdown ###Inpainting
##@markdown Inpaint occluded areas on top of raw frames. 0 - 0% inpainting opacity (no inpainting), 1 - 100% inpainting opacity. Other values blend between raw and inpainted frames.

inpaint_blend = 0 
##@param {type:"slider", min:0,max:1,value:1,step:0.1}

#@markdown ###Color matching
#@markdown Match color of inconsistent areas to unoccluded ones, after inconsistent areas were replaced with raw init video or inpainted\
#@markdown 0 - off, other values control effect opacity

match_color_strength = 0 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.1'}

disable_cc_for_turbo_frames = False