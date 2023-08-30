#@title Define optical flow functions for Video input animation mode only
# if animation_mode == 'Video Input Legacy':
DEBUG = False

# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


from torch import Tensor

# if True:
if animation_mode == 'Video Input':
  in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
  flo_folder = in_path+'_out_flo_fwd'
  #the main idea comes from neural-style-tf frame warping with optical flow maps
  #https://github.com/cysmith/neural-style-tf
  # path = f'{root_dir}/RAFT/core'
  # import sys
  # sys.path.append(f'{root_dir}/RAFT/core')
  # %cd {path}

  # from utils.utils import InputPadder

  class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

  # from raft import RAFT
  import numpy as np
  import argparse, PIL, cv2
  from PIL import Image
  from tqdm.notebook import tqdm
  from glob import glob
  import torch
  import scipy.ndimage

  args2 = argparse.Namespace()
  args2.small = False
  args2.mixed_precision = True

  TAG_CHAR = np.array([202021.25], np.float32)

  def writeFlow(filename,uv,v=None):
      """ 
      https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py
      Copyright 2017 NVIDIA CORPORATION

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.
      
      Write optical flow to file.
      
      If v is None, uv is assumed to contain both u and v channels,
      stacked in depth.
      Original code by Deqing Sun, adapted from Daniel Scharstein.
      """
      nBands = 2

      if v is None:
          assert(uv.ndim == 3)
          assert(uv.shape[2] == 2)
          u = uv[:,:,0]
          v = uv[:,:,1]
      else:
          u = uv

      assert(u.shape == v.shape)
      height,width = u.shape
      f = open(filename,'wb')
      # write the header
      f.write(TAG_CHAR)
      np.array(width).astype(np.int32).tofile(f)
      np.array(height).astype(np.int32).tofile(f)
      # arrange into matrix form
      tmp = np.zeros((height, width*nBands))
      tmp[:,np.arange(width)*2] = u
      tmp[:,np.arange(width)*2 + 1] = v
      tmp.astype(np.float32).tofile(f)
      f.close()
  


  # def load_cc(path, blur=2):
  #   weights = np.load(path)
  #   if blur>0: weights = scipy.ndimage.gaussian_filter(weights, [blur, blur])
  #   weights = np.repeat(weights[...,None],3, axis=2)
    
  #   if DEBUG: print('weight min max mean std', weights.shape, weights.min(), weights.max(), weights.mean(), weights.std())
  #   return weights

  def load_cc(path, blur=2):
    multilayer_weights = np.array(Image.open(path))/255
    weights = np.ones_like(multilayer_weights[...,0])
    weights*=multilayer_weights[...,0].clip(1-missed_consistency_weight,1)
    weights*=multilayer_weights[...,1].clip(1-overshoot_consistency_weight,1)
    weights*=multilayer_weights[...,2].clip(1-edges_consistency_weight,1)

    if blur>0: weights = scipy.ndimage.gaussian_filter(weights, [blur, blur])
    weights = np.repeat(weights[...,None],3, axis=2)
    
    if DEBUG: print('weight min max mean std', weights.shape, weights.min(), weights.max(), weights.mean(), weights.std())
    return weights
  
  

  def load_img(img, size):
    img = Image.open(img).convert('RGB').resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...].cuda()

  def get_flow(frame1, frame2, model, iters=20, half=True):
          # print(frame1.shape, frame2.shape)
          padder = InputPadder(frame1.shape)
          frame1, frame2 = padder.pad(frame1, frame2)
          if half: frame1, frame2 = frame1, frame2
          # print(frame1.shape, frame2.shape)
          _, flow12 = model(frame1, frame2)
          flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

          return flow12

  def warp_flow(img, flow, mul=1.):
      h, w = flow.shape[:2]
      flow = flow.copy()
      flow[:, :, 0] += np.arange(w)
      flow[:, :, 1] += np.arange(h)[:, np.newaxis]
      # print('flow stats', flow.max(), flow.min(), flow.mean())
      # print(flow)
      flow*=mul
      # print('flow stats mul', flow.max(), flow.min(), flow.mean())
      # res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
      res = cv2.remap(img, flow, None, cv2.INTER_LANCZOS4)
       
      return res

  def makeEven(_x):
    return _x if (_x % 2 == 0) else _x+1

  def fit(img,maxsize=512):
    maxdim = max(*img.size)
    if maxdim>maxsize:
    # if True:
      ratio = maxsize/maxdim
      x,y = img.size
      size = (makeEven(int(x*ratio)),makeEven(int(y*ratio))) 
      img = img.resize(size, warp_interp)
    return img


  def warp(frame1, frame2, flo_path, blend=0.5, weights_path=None, forward_clip=0., 
           pad_pct=0.1, padding_mode='reflect', inpaint_blend=0., video_mode=False, warp_mul=1.):
    printf('blend warp', blend)
    
    if isinstance(flo_path, str):
      flow21 = np.load(flo_path)
    else: flow21 = flo_path
    # print('loaded flow from ', flo_path, ' witch shape ', flow21.shape)
    pad = int(max(flow21.shape)*pad_pct)
    flow21 = np.pad(flow21, pad_width=((pad,pad),(pad,pad),(0,0)),mode='constant')
    # print('frame1.size, frame2.size, padded flow21.shape')
    # print(frame1.size, frame2.size, flow21.shape)
    

    frame1pil = np.array(frame1.convert('RGB'))#.resize((flow21.shape[1]-pad*2,flow21.shape[0]-pad*2),warp_interp))
    frame1pil = np.pad(frame1pil, pad_width=((pad,pad),(pad,pad),(0,0)),mode=padding_mode)
    if video_mode:
      warp_mul=1.
    frame1_warped21 = warp_flow(frame1pil, flow21, warp_mul)
    frame1_warped21 = frame1_warped21[pad:frame1_warped21.shape[0]-pad,pad:frame1_warped21.shape[1]-pad,:]

    frame2pil = np.array(frame2.convert('RGB').resize((flow21.shape[1]-pad*2,flow21.shape[0]-pad*2),warp_interp))
    # if not video_mode: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
    if weights_path: 
      forward_weights = load_cc(weights_path, blur=consistency_blur)
      # print('forward_weights')
      # print(forward_weights.shape)
      if not video_mode and match_color_strength>0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
        
      forward_weights = forward_weights.clip(forward_clip,1.)
      if use_patchmatch_inpaiting>0 and warp_mode == 'use_image':
        if not is_colab: print('Patchmatch only working on colab/linux')
        else: print('PatchMatch disabled.')
        # if not video_mode and is_colab: 
        #       print('patchmatching')
        #       # print(np.array(blended_w).shape, forward_weights[...,0][...,None].shape )
        #       patchmatch_mask = (forward_weights[...,0][...,None]*-255.+255).astype('uint8')
        #       frame2pil = np.array(frame2pil)*(1-use_patchmatch_inpaiting)+use_patchmatch_inpaiting*np.array(patch_match.inpaint(frame1_warped21, patchmatch_mask, patch_size=5))
        #       # blended_w = Image.fromarray(blended_w)
      blended_w = frame2pil*(1-blend) + blend*(frame1_warped21*forward_weights+frame2pil*(1-forward_weights))
    else: 
      if not video_mode and match_color_strength>0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
      blended_w = frame2pil*(1-blend) + frame1_warped21*(blend)

    

    blended_w = Image.fromarray(blended_w.round().astype('uint8'))
    # if use_patchmatch_inpaiting and warp_mode == 'use_image':
    #           print('patchmatching')
    #           print(np.array(blended_w).shape, forward_weights[...,0][...,None].shape )
    #           patchmatch_mask = (forward_weights[...,0][...,None]*-255.+255).astype('uint8')
    #           blended_w = patch_match.inpaint(blended_w, patchmatch_mask, patch_size=5)
    #           blended_w = Image.fromarray(blended_w)
    if not video_mode: 
      if enable_adjust_brightness: blended_w = adjust_brightness(blended_w)
    return  blended_w

  def warp_lat(frame1, frame2, flo_path, blend=0.5, weights_path=None, forward_clip=0., 
           pad_pct=0.1, padding_mode='reflect', inpaint_blend=0., video_mode=False, warp_mul=1.):
    warp_downscaled = True 
    flow21 = np.load(flo_path)
    pad = int(max(flow21.shape)*pad_pct)
    if warp_downscaled: 
      flow21 = flow21.transpose(2,0,1)[None,...]
      flow21 = torch.nn.functional.interpolate(torch.from_numpy(flow21).float(), scale_factor = 1/8, mode = 'bilinear')
      flow21 = flow21.numpy()[0].transpose(1,2,0)/8
      # flow21 = flow21[::8,::8,:]/8

    flow21 = np.pad(flow21, pad_width=((pad,pad),(pad,pad),(0,0)),mode='constant')

    if not warp_downscaled:
      frame1 = torch.nn.functional.interpolate(frame1, scale_factor = 8)
    frame1pil = frame1.cpu().numpy()[0].transpose(1,2,0)
  
    frame1pil = np.pad(frame1pil, pad_width=((pad,pad),(pad,pad),(0,0)),mode=padding_mode)
    if video_mode:
      warp_mul=1.
    frame1_warped21 = warp_flow(frame1pil, flow21, warp_mul)
    frame1_warped21 = frame1_warped21[pad:frame1_warped21.shape[0]-pad,pad:frame1_warped21.shape[1]-pad,:]
    if not warp_downscaled:
      frame2pil = frame2.convert('RGB').resize((flow21.shape[1]-pad*2,flow21.shape[0]-pad*2),warp_interp)
    else:
      frame2pil = frame2.convert('RGB').resize(((flow21.shape[1]-pad*2)*8,(flow21.shape[0]-pad*2)*8),warp_interp)
    frame2pil = np.array(frame2pil)
    frame2pil = (frame2pil/255.)[None,...].transpose(0, 3, 1, 2)
    frame2pil = 2*torch.from_numpy(frame2pil).float().cuda()-1.
    frame2pil = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame2pil))
    if not warp_downscaled: frame2pil = torch.nn.functional.interpolate(frame2pil, scale_factor = 8)
    frame2pil = frame2pil.cpu().numpy()[0].transpose(1,2,0)
    # if not video_mode: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
    if weights_path: 
      forward_weights = load_cc(weights_path, blur=consistency_blur)
      print(forward_weights[...,:1].shape, 'forward_weights.shape')
      forward_weights = np.repeat(forward_weights[...,:1],4, axis=-1)
      # print('forward_weights')
      # print(forward_weights.shape)
      print('frame2pil.shape, frame1_warped21.shape, flow21.shape', frame2pil.shape, frame1_warped21.shape, flow21.shape)  
      forward_weights = forward_weights.clip(forward_clip,1.)
      if warp_downscaled: forward_weights = forward_weights[::8,::8,:]; print(forward_weights.shape, 'forward_weights.shape')
      blended_w = frame2pil*(1-blend) + blend*(frame1_warped21*forward_weights+frame2pil*(1-forward_weights))
    else: 
      if not video_mode and not warp_mode == 'use_latent' and match_color_strength>0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
      blended_w = frame2pil*(1-blend) + frame1_warped21*(blend)
    blended_w = blended_w.transpose(2,0,1)[None,...]
    blended_w = torch.from_numpy(blended_w).float()
    if not warp_downscaled:  
      # blended_w = blended_w[::8,::8,:]
      blended_w = torch.nn.functional.interpolate(blended_w, scale_factor = 1/8, mode='bilinear')
    
    
    return blended_w# torch.nn.functional.interpolate(torch.from_numpy(blended_w), scale_factor = 1/8)


  in_path = videoFramesFolder if not flow_video_init_path else flowVideoFramesFolder
  flo_folder = in_path+'_out_flo_fwd'

  temp_flo = in_path+'_temp_flo'
  flo_fwd_folder = in_path+'_out_flo_fwd'
  flo_bck_folder = in_path+'_out_flo_bck'

  %cd {root_path}