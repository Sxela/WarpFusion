, threshold
#@title Create schedules from frame difference
def adjust_schedule(diff, normal_val, new_scene_val, thresh, falloff_frames, sched=None):
  diff_array = np.array(diff)

  diff_new = np.zeros_like(diff_array)
  diff_new = diff_new+normal_val

  for i in range(len(diff_new)):
    el = diff_array[i]
    if sched is not None:
      diff_new[i] = get_scheduled_arg(i, sched)
    if el>thresh or i==0: 
      diff_new[i] = new_scene_val
      if falloff_frames>0:
        for j in range(falloff_frames):
          if i+j>len(diff_new)-1: break
          # print(j,(falloff_frames-j)/falloff_frames, j/falloff_frames )
          falloff_val = normal_val
          if sched is not None:
            falloff_val = get_scheduled_arg(i+falloff_frames, sched)
          diff_new[i+j] = new_scene_val*(falloff_frames-j)/falloff_frames+falloff_val*j/falloff_frames
  return diff_new

def check_and_adjust_sched(sched, template, diff, respect_sched=True):
  if template is None or template == '' or template == []:
    return sched
  normal_val, new_scene_val, thresh, falloff_frames = template
  sched_source = None
  if respect_sched:
    sched_source = sched
  return list(adjust_schedule(diff, normal_val, new_scene_val, thresh, falloff_frames, sched_source).astype('float').round(3))

#@markdown fill in templates for schedules you'd like to create from frames' difference\
#@markdown leave blank to use schedules from previous cells\
#@markdown format: **[normal value, high difference value, difference threshold, falloff from high to normal (number of frames)]**\
#@markdown For example, setting flow blend template to [0.999, 0.3, 0.5, 5] will use 0.999 everywhere unless a scene has changed (frame difference >0.5) and then set flow_blend for this frame to 0.3 and gradually fade to 0.999 in 5 frames

latent_scale_template = '' #@param {'type':'raw'}
init_scale_template = '' #@param {'type':'raw'}
steps_template = '' #@param {'type':'raw'}
style_strength_template = [0.5, 0.9, 0.5, 5] #@param {'type':'raw'}
flow_blend_template = [0.3, 0., 0.5, 2] #@param {'type':'raw'}
cfg_scale_template = None #@param {'type':'raw'}
image_scale_template = None #@param {'type':'raw'}

#@markdown Turning this off will disable templates and will use schedules set in previous cell
make_schedules = False #@param {'type':'boolean'}
#@markdown Turning this on will respect previously set schedules and only alter the frames with peak difference
respect_sched = True #@param {'type':'boolean'}
diff_override = [] #@param {'type':'raw'}

#shift+1 required