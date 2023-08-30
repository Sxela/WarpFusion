#these variables are not in the GUI and are not being loaded.

torch.backends.cudnn.enabled = False # disabling this may increase performance on Ampere and Ada GPUs

diffuse_inpaint_mask_blur = 25 #used in mask result to extent the mask
diffuse_inpaint_mask_thresh = 0.8 #used in mask result to extent the mask

add_noise_to_latent = True #add noise to latent vector during latent guidance
noise_upscale_ratio = 1 #noise upscale ratio for latent noise during latent guidance
guidance_use_start_code = True #fix latent noise across steps during latent guidance
init_latent_fn = spherical_dist_loss #function to compute latent distance, l1_loss, rmse, spherical_dist_loss
use_scale = False #use gradient scaling (for mixed precision)
g_invert_mask = False #invert guidance mask

cb_noise_upscale_ratio = 1 #noise in masked diffusion callback
cb_add_noise_to_latent = True #noise in masked diffusion callback
cb_use_start_code = True #fix noise per frame in masked diffusion callback
cb_fixed_code = False #fix noise across all animation in masked diffusion callback (overcooks fast af)
cb_norm_latent = False #norm cb latent to normal ditribution stats in masked diffusion callback

img_zero_uncond = False #by default image conditioned models use same image for negative conditioning (i.e. both positive and negative image conditings are the same. you can use empty negative condition by enabling this)

controlnet_multimodel = {
  "control_sd15_depth": {
    "weight": 1,
    "start": 0,
    "end": 0.8
  },
  "control_sd15_canny": {
    "weight": 0,
    "start": 0.7,
    "end": 1
  },
  "control_sd15_hed": {
    "weight": 1,
    "start": 0,
    "end": 1
  },
  "control_sd15_mlsd": {
    "weight": 0,
    "start": 0,
    "end": 0
  },
  "control_sd15_normal": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_openpose": {
    "weight": 0,
    "start": 0,
    "end": 0.8
  },
  "control_sd15_scribble": {
    "weight": 0,
    "start": 0,
    "end": 0.6
  },
  "control_sd15_seg": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_temporalnet": {
    "weight": 0,
    "start": 0,
    "end": 1
  },
  "control_sd15_face": {
    "weight": 1,
    "start": 0,
    "end": 1
  }
}