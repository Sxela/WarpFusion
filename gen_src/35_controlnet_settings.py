steps_schedule = {
    0: 25
} #schedules total steps. useful with low strength, when you end up with only 10 steps at 0.2 strength x50 steps. Increasing max steps for low strength gives model more time to get to your text prompt
style_strength_schedule = [0.9,0.6]#[0.5]+[0.2]*149+[0.3]*3+[0.2] #use this instead of skip steps. It means how many steps we should do. 0.8 = we diffuse for 80% steps, so we skip 20%. So for skip steps 70% use 0.3
flow_blend_schedule = [1] #for example [0.1]*3+[0.999]*18+[0.3] will fade-in for 3 frames, keep style for 18 frames, and fade-out for the rest
cfg_scale_schedule = [7] #text2image strength, 7.5 is a good default
blend_json_schedules = True #True - interpolate values between keyframes. False - use latest keyframe 

dynamic_thresh = 30

fixed_code = False #Aka fixed seed. you can use this with fast moving videos, but be careful with still images 
blend_code = 0.1 # Only affects fixed code. high values make the output collapse
normalize_code = True #Only affects fixed code. 

warp_strength = 1 #leave 1 for no change. 1.01 is already a strong value.
flow_override_map = []#[*range(1,15)]+[16]*10+[*range(17+10,17+10+20)]+[18+10+20]*15+[*range(19+10+20+15,9999)] #map flow to frames. set to [] to disable.  [1]*10+[*range(10,9999)] repeats 1st frame flow 10 times, then continues as usual

blend_latent_to_init = 0

colormatch_after = False #colormatch after stylizing. On in previous notebooks.
colormatch_turbo = False #apply colormatching for turbo frames. On in previous notebooks

user_comment = 'testing cc layers'

mask_result = False #imitates inpainting by leaving only inconsistent areas to be diffused

use_karras_noise = False #Should work better with current sample, needs more testing.
end_karras_ramp_early = False

warp_interp = Image.LANCZOS
VERBOSE = True

use_patchmatch_inpaiting = 0

warp_num_k = 128 # number of patches per frame
warp_forward = False #use k-means patched warping (moves large areas instead of single pixels)

inverse_inpainting_mask = False
inpainting_mask_weight = 1.
mask_source = 'none'
mask_clip = [0, 255]
sampler = sample_euler
image_scale = 2
image_scale_schedule = {0:1.5, 1:2}

inpainting_mask_source = 'none'

fixed_seed = False #fixes seed
offload_model = True #offloads model to cpu defore running decoder. May save a bit of VRAM

use_predicted_noise = False
rec_randomness = 0.
rec_cfg = 1.
rec_prompts = {0: ['a beautiful highly detailed most beautiful (woman) ever']}
rec_source = 'init'
rec_steps_pct = 1

#controlnet settings
controlnet_preprocess = True #preprocess input conditioning image for controlnet. If false, use raw conditioning as input to the model without detection/preprocessing
detect_resolution = 768 #control net conditioning image resolution
bg_threshold = 0.4 #controlnet depth/normal bg cutoff threshold
low_threshold = 100 #canny filter parameters
high_threshold = 200 #canny filter parameters
value_threshold = 0.1 #mlsd model settings
distance_threshold = 0.1 #mlsd model settings

temporalnet_source = 'stylized'
temporalnet_skip_1st_frame = True
controlnet_multimodel_mode = 'internal' #external or internal. internal - sums controlnet values before feeding those into diffusion model, external - sum outputs of differnet contolnets after passing through diffusion model. external seems slower but smoother.)

do_softcap = False #softly clamp latent excessive values. reduces feedback loop effect a bit 
softcap_thresh = 0.9 # scale down absolute values above that threshold (latents are being clamped at [-1:1] range, so 0.9 will downscale values above 0.9 to fit into that range, [-1.5:1.5] will be scaled to [-1:1], but only absolute values over 0.9 will be affected)
softcap_q = 1. # percentile to downscale. 1-downscle full range with outliers, 0.9 - downscale only 90%  values above thresh, clamp 10%)

max_faces = 10
masked_guidance = False #use mask for init/latent guidance to ignore inconsistencies and only guide based on the consistent areas 
mask_callback = 0.5  # 0 - off. 0.5-0.7 are good values. make inconsistent area passes only before this % of actual steps, then diffuse whole image