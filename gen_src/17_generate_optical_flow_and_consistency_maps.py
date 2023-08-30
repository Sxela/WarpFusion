#@title Generate optical flow and consistency maps
#@markdown Run once per init video\
#@markdown If you are getting **"AttributeError: module 'PIL.TiffTags' has no attribute 'IFD'"** error,\
#@markdown just click **"Runtime" - "Restart and Run All"** once per session.
#hack to get pillow to work w\o restarting
#if you're running locally, just restart this runtime, no need to edit PIL files.
flow_warp = True
check_consistency = True
force_flow_generation = False #@param {type:'boolean'}
def hstack(images):
  if isinstance(next(iter(images)), str):
    images = [Image.open(image).convert('RGB') for image in images]
  widths, heights = zip(*(i.size for i in images))
  for image in images: 
    draw = ImageDraw.Draw(image)
    draw.rectangle(((0, 00), (image.size[0], image.size[1])), outline="black", width=3)
  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
  return new_im

import locale
def getpreferredencoding(do_setlocale = True):
            return "UTF-8"
if is_colab: locale.getpreferredencoding = getpreferredencoding

def vstack(images):
  if isinstance(next(iter(images)), str):
    images = [Image.open(image).convert('RGB') for image in images]
  widths, heights = zip(*(i.size for i in images))

  total_height = sum(heights)
  max_width = max(widths)

  new_im = Image.new('RGB', (max_width, total_height))

  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]
  return new_im

if is_colab:
  for i in [7,8,9,10]:
    try:
      filedata = None
      with open(f'/usr/local/lib/python3.{i}/dist-packages/PIL/TiffImagePlugin.py', 'r') as file :
        filedata = file.read()
      filedata = filedata.replace('(TiffTags.IFD, "L", "long"),', '#(TiffTags.IFD, "L", "long"),')
      with open(f'/usr/local/lib/python3.{i}/dist-packages/PIL/TiffImagePlugin.py', 'w') as file :
        file.write(filedata)
      with open(f'/usr/local/lib/python3.7/dist-packages/PIL/TiffImagePlugin.py', 'w') as file :
        file.write(filedata)
    except:
      pass
      # print(f'Error writing /usr/local/lib/python3.{i}/dist-packages/PIL/TiffImagePlugin.py')

class flowDataset():
  def __init__(self, in_path, half=True, normalize=False):
    frames = sorted(glob(in_path+'/*.*'));
    assert len(frames)>2, f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.'
    self.frames = frames
      
  def __len__(self):
    return len(self.frames)-1 

  def load_img(self, img, size):
    img = Image.open(img).convert('RGB').resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...]

  def __getitem__(self, i):
    frame1, frame2 = self.frames[i], self.frames[i+1]
    frame1 = self.load_img(frame1, width_height)
    frame2 = self.load_img(frame2, width_height)
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    batch = torch.cat([frame1, frame2])
    if normalize:
      batch = 2 * (batch / 255.0) - 1.0
    return batch

from torch.utils.data import DataLoader

def save_preview(flow21, out_flow21_fn):
  Image.fromarray(flow_to_image(flow21)).save(out_flow21_fn, quality=90)

#copyright Alex Spirin @ 2022
def blended_roll(img_copy, shift, axis):
  if int(shift) == shift: 
    return np.roll(img_copy, int(shift), axis=axis)

  max = math.ceil(shift)
  min = math.floor(shift)
  if min != 0 :
    img_min = np.roll(img_copy, min, axis=axis)
  else: 
    img_min = img_copy
  img_max = np.roll(img_copy, max, axis=axis)
  blend = max-shift
  img_blend = img_min*blend + img_max*(1-blend)
  return img_blend

#copyright Alex Spirin @ 2022
def move_cluster(img,i,res2, center, mode='blended_roll'):
  img_copy = img.copy()
  motion = center[i]
  mask = np.where(res2==motion, 1, 0)[...,0][...,None]
  y, x = motion
  if mode=='blended_roll':
    img_copy = blended_roll(img_copy, x, 0)
    img_copy = blended_roll(img_copy, y, 1)
  if mode=='int_roll':    
    img_copy = np.roll(img_copy, int(x), axis=0)
    img_copy = np.roll(img_copy, int(y), axis=1)
  return img_copy, mask

import cv2


def get_k(flow, K):
  Z = flow.reshape((-1,2))
  # convert to np.float32
  Z = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
  # Now convert back into uint8, and make original image
  res = center[label.flatten()]
  res2 = res.reshape((flow.shape))
  return res2, center

def k_means_warp(flo, img, num_k):
  # flo = np.load(flo)
  img = np.array((img).convert('RGB'))
  num_k = 8 
  
  # print(img.shape)
  res2, center = get_k(flo, num_k)
  center = sorted(list(center), key=lambda x: abs(x).mean())

  img = cv2.resize(img, (res2.shape[:-1][::-1]))
  img_out = np.ones_like(img)*255.

  for i in range(num_k):
    img_rolled, mask_i = move_cluster(img,i,res2,center)
    img_out = img_out*(1-mask_i) + img_rolled*(mask_i)

  # cv2_imshow(img_out)
  return Image.fromarray(img_out.astype('uint8'))

def flow_batch(i, batch, pool):
  with torch.cuda.amp.autocast():
          batch = batch[0]
          frame_1 = batch[0][None,...].cuda()
          frame_2 = batch[1][None,...].cuda()
          frame1 = ds.frames[i]
          frame1 = frame1.replace('\\','/')
          out_flow21_fn = f"{flo_fwd_folder}/{frame1.split('/')[-1]}"
          if flow_lq:   frame_1, frame_2 = frame_1, frame_2
          if use_jit_raft:
            _, flow21 = raft_model(frame_2, frame_1)
          else:
            flow21 = raft_model(frame_2, frame_1, num_flow_updates=num_flow_updates)[-1]
          flow21 = flow21[0].permute(1, 2, 0).detach().cpu().numpy()
          
          if flow_save_img_preview or i in range(0,len(ds),len(ds)//10): 
            pool.apply_async(save_preview, (flow21, out_flow21_fn+'.jpg') )
          pool.apply_async(np.save, (out_flow21_fn, flow21))
          if check_consistency:
            if use_jit_raft:
              _, flow12 = raft_model(frame_1, frame_2)
            else:
              flow12 = raft_model(frame_1, frame_2)[-1]
            # _, flow12 = raft_model(frame_1, frame_2)
            flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()
            if flow_save_img_preview: 
              pool.apply_async(save_preview, (flow12, out_flow21_fn+'_12'+'.jpg'))
            pool.apply_async(np.save, (out_flow21_fn+'_12', flow12))

from multiprocessing.pool import ThreadPool as Pool
import gc
threads = 4 #@param {'type':'number'}
#@markdown If you're having "process died" error on Windows, set num_workers to 0
num_workers = 0 #@param {'type':'number'}

#@markdown Use lower quality model (half-precision).\
#@markdown Uses half the vram, allows fitting 1500x1500+ frames into 16gigs, which the original full-precision RAFT can't do.
flow_lq = True #@param {type:'boolean'}
#@markdown Save human-readable flow images along with motion vectors. Check /{your output dir}/videoFrames/out_flo_fwd folder.
flow_save_img_preview = False  #@param {type:'boolean'}
in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
flo_folder = in_path+'_out_flo_fwd'
# #@markdown reverse_cc_order - on - default value (like in older notebooks). off - reverses consistency computation
reverse_cc_order = True  #
# #@param {type:'boolean'}
if not flow_warp: print('flow_wapr not set, skipping')
try: raft_model
except: raft_model = None
#@markdown Use previous pre-compile raft version (won't work with pytorch 2.0)
use_jit_raft = False#@param {'type':'boolean'}
#@markdown Compile raft model (only with use_raft_jit = False). Compiles the model (~about 2 minutes) for ~30% speedup. Use for very long runs.
compile_raft = False#@param {'type':'boolean'}
#@markdown Flow estimation quality (number of iterations, 12 - default. higher - better and slower)
num_flow_updates = 20 #@param {'type':'number'}
if (animation_mode == 'Video Input') and (flow_warp):
  flows = glob(flo_folder+'/*.*')
  if (len(flows)>0) and not force_flow_generation: print(f'Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {flo_folder}.\nIf you wish to generate new flow files, check force_flow_generation and run this cell again.')

  if (len(flows)==0) or force_flow_generation:
    ds = flowDataset(in_path, normalize=not use_jit_raft)
  
    frames = sorted(glob(in_path+'/*.*'));
    if len(frames)<2: 
      print(f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.')
    if len(frames)>=2:
      if __name__ == '__main__':
        
        dl = DataLoader(ds, num_workers=num_workers)  
        if use_jit_raft: 
          if flow_lq:
            raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
          # raft_model = torch.nn.DataParallel(RAFT(args2))
          else: raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()
          # raft_model.load_state_dict(torch.load(f'{root_path}/RAFT/models/raft-things.pth'))
          # raft_model = raft_model.module.cuda().eval()
        else:
          if raft_model is None or not compile_raft:
            from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
            from torchvision.models.optical_flow import raft_large, raft_small
            raft_weights = Raft_Large_Weights.C_T_SKHT_V1
            raft_device = "cuda" if torch.cuda.is_available() else "cpu"

            raft_model = raft_large(weights=raft_weights, progress=False).to(raft_device)
            # raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(raft_device)
            raft_model = raft_model.eval()
            if gpu != 'T4' and compile_raft: raft_model = torch.compile(raft_model)
            if flow_lq:
              raft_model = raft_model.half()


        for f in pathlib.Path(f'{flo_fwd_folder}').glob('*.*'):
          f.unlink()

        temp_flo = in_path+'_temp_flo'
        flo_fwd_folder = in_path+'_out_flo_fwd'

        os.makedirs(flo_fwd_folder, exist_ok=True)
        os.makedirs(temp_flo, exist_ok=True)
        cc_path = f'{root_dir}/flow_tools/check_consistency.py'
        with torch.no_grad():
          p = Pool(threads)
          for i,batch in enumerate(tqdm(dl)):
              flow_batch(i, batch, p)
          p.close()
          p.join()

        # del raft_model 
        gc.collect()
        if is_colab: locale.getpreferredencoding = getpreferredencoding
        if check_consistency:
          fwd = f"{flo_fwd_folder}/*jpg.npy"
          bwd = f"{flo_fwd_folder}/*jpg_12.npy"
          
          if reverse_cc_order:
            #old version, may be incorrect
            print('Doing bwd->fwd cc check')
            !python "{cc_path}" --flow_fwd "{fwd}" --flow_bwd "{bwd}" --output "{flo_fwd_folder}/" --image_output --output_postfix="-21_cc" --blur=0. --save_separate_channels --skip_numpy_output
          else:
            print('Doing fwd->bwd cc check')
            !python "{cc_path}" --flow_fwd "{bwd}" --flow_bwd "{fwd}" --output "{flo_fwd_folder}/" --image_output --output_postfix="-21_cc" --blur=0. --save_separate_channels --skip_numpy_output
          # delete forward flow
          # for f in pathlib.Path(flo_fwd_folder).glob('*jpg_12.npy'):
          #   f.unlink()

# previews_flow = glob(f'{flo_fwd_folder}/*.jpg.jpg'); len(previews_flow)
# rowsz = 5
# imgs_flow = vstack([hstack(previews_flow[i*rowsz:(i+1)*rowsz]) for i in range(len(previews_flow)//rowsz)])

# previews_cc = glob(f'{flo_fwd_folder}/*.jpg-21_cc.jpg')
# previews_cc = previews_cc[::len(previews_cc)//10]; len(previews_cc)
# rowsz = 5
# imgs_cc = vstack([hstack(previews_cc[i*rowsz:(i+1)*rowsz]) for i in range(len(previews_cc)//rowsz)])

# imgs = vstack([imgs_flow, imgs_cc.convert('L')])
print('Samples from raw. alpha, consistency, and flow maps')
# fit(imgs, 1024)

flo_imgs = glob(flo_fwd_folder+'/*.jpg.jpg')[:5]
vframes = []
for flo_img in flo_imgs:
  hframes = []
  flo_img = flo_img.replace('\\','/')
  frame = Image.open(videoFramesFolder + '/' + flo_img.split('/')[-1][:-4])
  hframes.append(frame)
  try:
    alpha = Image.open(videoFramesAlpha + '/' + flo_img.split('/')[-1][:-4]).resize(frame.size)
    hframes.append(alpha)
  except: 
    pass
  try:
    cc_img = Image.open(flo_img[:-4]+'-21_cc.jpg').convert('L').resize(frame.size)
    hframes.append(cc_img)
  except: 
    pass
  try:  
    flo_img = Image.open(flo_img).resize(frame.size)
    hframes.append(flo_img)
  except: 
    pass
  v_imgs = vstack(hframes)
  vframes.append(v_imgs)
preview = hstack(vframes)
fit(preview, 1024)