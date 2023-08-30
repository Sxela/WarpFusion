import PIL
#@title ### **Create video**
#@markdown Video file will save in the same folder as your images.
from tqdm.notebook import trange
skip_video_for_run_all = False #@param {type: 'boolean'}
#@markdown ### **Video masking (post-processing)**
#@markdown Use previously generated background mask during video creation
use_background_mask_video = False #@param {type: 'boolean'}
invert_mask_video = False #@param {type: 'boolean'}
#@markdown Choose background source: image, color, init video.
background_video = "init_video" #@param ['image', 'color', 'init_video']
#@markdown Specify the init image path or color depending on your background video source choice.
background_source_video = 'red' #@param {type: 'string'}
blend_mode = "optical flow" #@param ['None', 'linear', 'optical flow']
# if (blend_mode == "optical flow") & (animation_mode != 'Video Input Legacy'):
#@markdown ### **Video blending (post-processing)**
#   print('Please enable Video Input mode and generate optical flow maps to use optical flow blend mode')
blend =  0.5#@param {type: 'number'}
check_consistency = True #@param {type: 'boolean'}
postfix = ''

def try_process_frame(i, func):
    try:
        func(i)
    except:
        print('Error processing frame ', i)



if use_background_mask_video:
  postfix+='_mask'

#@markdown #### Multithreading settings
#@markdown Suggested range - from 1 to number of cores on SSD and double number of cores - on HDD. Mostly limited by your drive bandwidth.
#@markdown Results for 500 frames @ 6 cores: 5 threads - 2:38, 10 threads - 0:55, 20 - 0:56, 1: 5:53
threads = 12#@param {type:"number"}
threads = max(min(threads, 64),1)
frames = []

#@markdown ### **Video settings**

if skip_video_for_run_all == True:
  print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')

else:
  # import subprocess in case this cell is run without the above cells
  import subprocess
  from base64 import b64encode

  from multiprocessing.pool import ThreadPool as Pool
  
  pool = Pool(threads)

  latest_run = batchNum

  folder = batch_name #@param
  run =  latest_run#@param
  final_frame = 'final_frame'
  

  init_frame = 1#@param {type:"number"} This is the frame where the video will start
  last_frame = final_frame#@param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
  fps = 24#@param {type:"number"}
  output_format = 'mp4' #@param ['mp4','mov']
  # view_video_in_cell = True #@param {type: 'boolean'}

  # tqdm.write('Generating video...')

  if last_frame == 'final_frame':
    last_frame = len(glob(batchFolder+f"/{folder}({run})_*.png"))
    print(f'Total frames: {last_frame}')

  image_path = f"{outDirPath}/{folder}/{folder}({run})_%06d.png"
  filepath = f"{outDirPath}/{folder}/{folder}({run}).{output_format}"

  if (blend_mode == 'optical flow') & (True) :
    image_path = f"{outDirPath}/{folder}/flow/{folder}({run})_%06d.png"
    postfix += '_flow'
    video_out = batchFolder+f"/video"
    os.makedirs(video_out, exist_ok=True)
    filepath = f"{video_out}/{folder}({run})_{postfix}.{output_format}"
    if last_frame == 'final_frame':
      last_frame = len(glob(batchFolder+f"/flow/{folder}({run})_*.png"))
    flo_out = batchFolder+f"/flow"
    # !rm -rf {flo_out}/* 
    
    # !mkdir "{flo_out}"
    os.makedirs(flo_out, exist_ok=True)
    
    frames_in = sorted(glob(batchFolder+f"/{folder}({run})_*.png"))

    frame0 = Image.open(frames_in[0])
    if use_background_mask_video:   
      frame0 = apply_mask(frame0, 0, background_video, background_source_video, invert_mask_video)
    frame0.save(flo_out+'/'+frames_in[0].replace('\\','/').split('/')[-1])

    def process_flow_frame(i):
        frame1_path = frames_in[i-1]
        frame2_path = frames_in[i]

        frame1 = Image.open(frame1_path)
        frame2 = Image.open(frame2_path)
        frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4])+1):06}.jpg"
        flo_path = f"{flo_folder}/{frame1_stem}.npy"
        weights_path = None
        if check_consistency: 
          if reverse_cc_order:
            weights_path = f"{flo_folder}/{frame1_stem}-21_cc.jpg" 
          else: 
            weights_path = f"{flo_folder}/{frame1_stem}_12-21_cc.jpg" 
        tic = time.time()
        printf('process_flow_frame warp')
        frame = warp(frame1, frame2, flo_path, blend=blend, weights_path=weights_path, 
            pad_pct=padding_ratio, padding_mode=padding_mode, inpaint_blend=0, video_mode=True)
        if use_background_mask_video:
          frame = apply_mask(frame, i, background_video, background_source_video, invert_mask_video)
        frame.save(batchFolder+f"/flow/{folder}({run})_{i:06}.png")

    with Pool(threads) as p:
      fn = partial(try_process_frame, func=process_flow_frame)
      total_frames = range(init_frame, min(len(frames_in), last_frame))
      result = list(tqdm(p.imap(fn, total_frames), total=len(total_frames)))

  if blend_mode == 'linear':
    image_path = f"{outDirPath}/{folder}/blend/{folder}({run})_%06d.png"
    postfix += '_blend'
    video_out = batchFolder+f"/video"
    os.makedirs(video_out, exist_ok=True)
    filepath = f"{video_out}/{folder}({run})_{postfix}.{output_format}"
    if last_frame == 'final_frame':
      last_frame = len(glob(batchFolder+f"/blend/{folder}({run})_*.png"))
    blend_out = batchFolder+f"/blend"
    os.makedirs(blend_out, exist_ok = True)
    frames_in = glob(batchFolder+f"/{folder}({run})_*.png")
    
    frame0 = Image.open(frames_in[0])
    if use_background_mask_video:   
      frame0 = apply_mask(frame0, 0, background_video, background_source_video, invert_mask_video)
    frame0.save(flo_out+'/'+frames_in[0].replace('\\','/').split('/')[-1])

    def process_blend_frame(i):
      frame1_path = frames_in[i-1]
      frame2_path = frames_in[i]

      frame1 = Image.open(frame1_path)
      frame2 = Image.open(frame2_path)
      frame = Image.fromarray((np.array(frame1)*(1-blend) + np.array(frame2)*(blend)).round().astype('uint8'))
      if use_background_mask_video:
        frame = apply_mask(frame, i, background_video, background_source_video, invert_mask_video)
      frame.save(batchFolder+f"/blend/{folder}({run})_{i:06}.png")

    with Pool(threads) as p:
      fn = partial(try_process_frame, func=process_blend_frame)
      total_frames = range(init_frame, min(len(frames_in), last_frame))
      result = list(tqdm(p.imap(fn, total_frames), total=len(total_frames)))
  if output_format == 'mp4':
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec',
        'png',
        '-r',
        str(fps),
        '-start_number',
        str(init_frame),
        '-i',
        image_path,
        '-frames:v',
        str(last_frame+1),
        '-c:v',
        'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt',
        'yuv420p',
        '-crf',
        '17',
        '-preset',
        'veryslow',
        filepath
    ]
  if output_format == 'mov':
      cmd = [
      'ffmpeg',
      '-y',
      '-vcodec',
      'png',
      '-r',
      str(fps),
      '-start_number',
      str(init_frame),
      '-i',
      image_path,
      '-frames:v',
      str(last_frame+1),
      '-c:v',
      'qtrle',
      '-vf',
      f'fps={fps}',
      filepath
  ]


  process = subprocess.Popen(cmd, cwd=f'{batchFolder}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  if process.returncode != 0:
      print(stderr)
      raise RuntimeError(stderr)
  else:
      print("The video is ready and saved to the images folder")

  # if view_video_in_cell:
  #     mp4 = open(filepath,'rb').read()
  #     data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  #     display.HTML(f'<video width=400 controls><source src="{data_url}" type="video/mp4"></video>')