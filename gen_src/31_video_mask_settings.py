#@title Video mask settings
#@markdown Check to enable background masking during render. Not recommended, better use masking when creating the output video for more control and faster testing.
use_background_mask = False #@param {'type':'boolean'}
#@markdown Check to invert the mask.
invert_mask = False #@param {'type':'boolean'}
#@markdown Apply mask right before feeding init image to the model. Unchecking will only mask current raw init frame.
apply_mask_after_warp = True #@param {'type':'boolean'}
#@markdown Choose background source to paste masked stylized image onto: image, color, init video.
background = "init_video" #@param ['image', 'color', 'init_video']
#@markdown Specify the init image path or color depending on your background source choice.
background_source = 'red' #@param {'type':'string'}