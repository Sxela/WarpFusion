#@title  ####**Advanced Settings:**

set_seed = '4275770367' #@param{type: 'string'}


#@markdown *Clamp grad is used with any of the init_scales or sat_scale above 0*\
#@markdown Clamp grad limits the amount various criterions, controlled by *_scale parameters, are pushing the image towards the desired result.\
#@markdown For example, high scale values may cause artifacts, and clamp_grad removes this effect.
#@markdown 0.7 is a good clamp_max value.
eta = 0.55
clamp_grad = True #@param{type: 'boolean'}
clamp_max = 2 #@param{type: 'number'}