#@markdown ###Automatic Brightness Adjustment
#@markdown Automatically adjust image brightness when its mean value reaches a certain threshold\
#@markdown Ratio means the vaue by which pixel values are multiplied when the thresjold is reached\
#@markdown Fix amount is being directly added to\subtracted from pixel values to prevent oversaturation due to multiplications\
#@markdown Fix amount is also being applied to border values defined by min\max threshold, like 1 and 254 to keep the image from having burnt out\pitch black areas while still being within set high\low thresholds


#@markdown The idea comes from https://github.com/lowfuel/progrockdiffusion

enable_adjust_brightness = False #@param {'type':'boolean'}
high_brightness_threshold = 180 #@param {'type':'number'}
high_brightness_adjust_ratio = 0.97 #@param {'type':'number'}
high_brightness_adjust_fix_amount = 2 #@param {'type':'number'}
max_brightness_threshold = 254 #@param {'type':'number'}
low_brightness_threshold = 40 #@param {'type':'number'}
low_brightness_adjust_ratio = 1.03 #@param {'type':'number'}
low_brightness_adjust_fix_amount = 2 #@param {'type':'number'}
min_brightness_threshold = 1 #@param {'type':'number'}