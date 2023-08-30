#@title Flow and turbo settings
#@markdown #####**Video Optical Flow Settings:**
flow_warp = True #@param {type: 'boolean'} 
#cal optical flow from video frames and warp prev frame with flow
flow_blend =  0.999
##@param {type: 'number'} #0 - take next frame, 1 - take prev warped frame
check_consistency = True #@param {type: 'boolean'}
 #cal optical flow from video frames and warp prev frame with flow

#======= TURBO MODE
#@markdown ---
#@markdown ####**Turbo Mode:**
#@markdown (Starts after frame 1,) skips diffusion steps and just uses flow map to warp images for skipped frames.
#@markdown Speeds up rendering by 2x-4x, and may improve image coherence between frames. frame_blend_mode smooths abrupt texture changes across 2 frames.
#@markdown For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo

turbo_mode = False #@param {type:"boolean"}
turbo_steps = "3" #@param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 1 # frames