#@title Content-aware scheduing
#@markdown Allows automated settings scheduling based on video frames difference. If a scene changes, it will be detected and reflected in the schedule.\
#@markdown rmse function is faster than lpips, but less precise.\
#@markdown After the analysis is done, check the graph and pick a threshold that works best for your video. 0.5 is a good one for lpips, 1.2 is a good one for rmse. Don't forget to adjust the templates with new threshold in the cell below.

def load_img_lpips(path, size=(512,512)):
    image = Image.open(path).convert("RGB")
    image = image.resize(size, resample=Image.LANCZOS)
    # print(f'resized to {image.size}')
    image = np.array(image).astype(np.float32) / 127
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = normalize(image)
    return image.cuda()

diff = None
analyze_video = False #@param {'type':'boolean'}

diff_function = 'lpips' #@param ['rmse','lpips','rmse+lpips']

def l1_loss(x,y):
  return torch.sqrt(torch.mean((x-y)**2))


def rmse(x,y):
  return torch.abs(torch.mean(x-y))

def joint_loss(x,y):
  return rmse(x,y)*lpips_model(x,y)

diff_func = rmse
if  diff_function == 'lpips':
  diff_func = lpips_model
if diff_function == 'rmse+lpips':
  diff_func = joint_loss

if analyze_video:
  diff = [0]
  frames = sorted(glob(f'{videoFramesFolder}/*.jpg'))
  from tqdm.notebook import trange
  for i in trange(1,len(frames)):
    with torch.no_grad():
      diff.append(diff_func(load_img_lpips(frames[i-1]), load_img_lpips(frames[i])).sum().mean().detach().cpu().numpy())

  import numpy as np
  import matplotlib.pyplot as plt

  plt.rcParams["figure.figsize"] = [12.50, 3.50]
  plt.rcParams["figure.autolayout"] = True

  y = diff
  plt.title(f"{diff_function} frame difference")
  plt.plot(y, color="red")
  calc_thresh = np.percentile(np.array(diff), 97)
  plt.axhline(y=calc_thresh, color='b', linestyle='dashed')

  plt.show()
  print(f'suggested threshold: {calc_thresh.round(2)}')