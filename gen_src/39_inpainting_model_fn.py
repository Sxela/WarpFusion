#@title inpainting model fn
# frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
# weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg" 
# forward_weights = load_cc(weights_path, blur=consistency_blur)

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1, inpainting_mask_weight=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    if mask is not None:
      mask = np.array(mask.convert("L"))
      mask = mask.astype(np.float32)/255.0
      mask = mask[None,None]
      mask[mask < 0.5] = 0
      mask[mask >= 0.5] = 1
      mask = torch.from_numpy(mask)
    else: 
      mask = image.new_ones(1, 1, *image.shape[-2:])

    # masked_image = image * (mask < 0.5)

    masked_image = torch.lerp(
            image,
            image * (mask < 0.5),
            inpainting_mask_weight
        )

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch

def inpainting_conditioning(source_image, image_mask = None, inpainting_mask_weight = 1, sd_model=sd_model):
        #based on https://github.com/AUTOMATIC1111/stable-diffusion-webui

        # Handle the different mask inputs
        if image_mask is not None:
            
            if torch.is_tensor(image_mask):

                conditioning_mask = image_mask[:,:1,...]
                # print('mask conditioning_mask', conditioning_mask.shape)
            else:
                print(image_mask.shape, source_image.shape)
                # conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = image_mask[...,0].astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None]).float()

                # Inpainting model uses a discretized mask as input, so we round to either 1.0 or 0.0
                conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = source_image.new_ones(1, 1, *source_image.shape[-2:])
        print(conditioning_mask.shape, source_image.shape)
        # Create another latent image, this time with a masked version of the original input.
        # Smoothly interpolate between the masked and unmasked latent conditioning image using a parameter.
        conditioning_mask = conditioning_mask.to(source_image.device).to(source_image.dtype)
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            inpainting_mask_weight
        )
        
        # Encode the new masked image using first stage of network.
        conditioning_image =  sd_model.get_first_stage_encoding( sd_model.encode_first_stage(conditioning_image))

        # Create the concatenated conditioning tensor to be fed to `c_concat`
        conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=conditioning_image.shape[-2:])
        conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
        image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
        image_conditioning = image_conditioning.to('cuda').type( sd_model.dtype)

        return image_conditioning