# DD-style losses, renders 2 times slower (!) and more memory intensive :D

latent_scale_schedule = [0,0] #controls coherency with previous frame in latent space. 0 is a good starting value. 1+ render slower, but may improve image coherency. 100 is a good value if you decide to turn it on.
init_scale_schedule = [0,0] #controls coherency with previous frame in pixel space. 0 - off, 1000 - a good starting value if you decide to turn it on.
sat_scale = 0

init_grad = False #True - compare result to real frame, False - to stylized frame
grad_denoised = True #fastest, on by default, calc grad towards denoised x instead of input x