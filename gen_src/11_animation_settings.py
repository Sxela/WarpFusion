#@title ## Animation Settings
#@markdown Create a looping video from single init image\
#@markdown Use this if you just want to test settings. This will create a small video (1 sec = 24 frames)\
#@markdown This way you will be able to iterate faster without the need to process flow maps for a long final video before even getting to testing prompts.
#@markdown You'll need to manually input the resulting video path into the next cell.

use_looped_init_image = False #@param {'type':'boolean'}
video_duration_sec = 2 #@param {'type':'number'}
if use_looped_init_image:
  !ffmpeg -loop 1 -i "{init_image}" -c:v libx264 -t "{video_duration_sec}" -pix_fmt yuv420p -vf scale={side_x}:{side_y} "{root_dir}/out.mp4" -y
  print('Video saved to ', f"{root_dir}/out.mp4")