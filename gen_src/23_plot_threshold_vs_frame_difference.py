#@title Plot threshold vs frame difference
#@markdown The suggested threshold may be incorrect, so you can plot your value and see if it covers the peaks.
if diff is not None:
  import numpy as np
  import matplotlib.pyplot as plt

  plt.rcParams["figure.figsize"] = [12.50, 3.50]
  plt.rcParams["figure.autolayout"] = True

  y = diff
  plt.title(f"{diff_function} frame difference")
  plt.plot(y, color="red")
  calc_thresh = np.percentile(np.array(diff), 97)
  plt.axhline(y=calc_thresh, color='b', linestyle='dashed')
  user_threshold = 0.5 #@param {'type':'raw'}
  plt.axhline(y=user_threshold, color='r')

  plt.show()
  peaks = []
  for i,d in enumerate(diff):
    if d>user_threshold:
      peaks.append(i)
  print(f'Peaks at frames: {peaks} for user_threshold of {user_threshold}')
else: print('Please analyze frames in the previous cell  to plot graph')