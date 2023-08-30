#@title Frame correction
#@markdown Match frame pixels or latent to other frames to preven oversaturation and feedback loop artifacts
#@markdown ###Latent matching
#@markdown Match the range of latent vector towards the 1st frame or a user defined range. Doesn't restrict colors, but may limit contrast.
normalize_latent = 'off' #@param ['off', 'color_video', 'color_video_offset', 'user_defined', 'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset']
#@markdown in offset mode, specifies the offset back from current frame, and 0 means current frame. In non-offset mode specifies the fixed frame number. 0 means the 1st frame. 

normalize_latent_offset = 0  #@param {'type':'number'}
#@markdown User defined stats to normalize the latent towards
latent_fixed_mean = 0.  #@param {'type':'raw'}
latent_fixed_std = 0.9  #@param {'type':'raw'}
#@markdown Match latent on per-channel basis 
latent_norm_4d = True  #@param {'type':'boolean'}
#@markdown ###Color matching
#@markdown Color match frame towards stylized or raw init frame. Helps prevent images going deep purple. As a drawback, may lock colors to the selected fixed frame. Select stylized_frame with colormatch_offset = 0 to reproduce previous notebooks.
colormatch_frame = 'stylized_frame' #@param ['off', 'color_video', 'color_video_offset','stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset']
#@markdown Color match strength. 1 mimics legacy behavior
color_match_frame_str = 0.2 #@param {'type':'number'}
#@markdown in offset mode, specifies the offset back from current frame, and 0 means current frame. In non-offset mode specifies the fixed frame number. 0 means the 1st frame. 
colormatch_offset = 0  #@param {'type':'number'}
colormatch_method = 'LAB'#@param ['LAB', 'PDF', 'mean']
colormatch_method_fn = PT.lab_transfer
if colormatch_method == 'LAB':
  colormatch_method_fn = PT.pdf_transfer 
if colormatch_method == 'mean': 
  colormatch_method_fn = PT.mean_std_transfer
#@markdown Match source frame's texture
colormatch_regrain = False #@param {'type':'boolean'}