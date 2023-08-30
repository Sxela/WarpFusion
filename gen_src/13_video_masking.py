#@title Video Masking

#@markdown Generate background mask from your init video or use a video as a mask
mask_source = 'init_video' #@param ['init_video','mask_video']
#@markdown Check to rotoscope the video and create a mask from it. If unchecked, the raw monochrome video will be used as a mask.
extract_background_mask = False #@param {'type':'boolean'}
#@markdown Specify path to a mask video for mask_video mode.
mask_video_path = '' #@param {'type':'string'}
if extract_background_mask:
  os.chdir(root_dir)
  !pip install av pims
  gitclone('https://github.com/Sxela/RobustVideoMattingCLI')
  if mask_source == 'init_video':
    videoFramesAlpha = videoFramesFolder+'Alpha'
    createPath(videoFramesAlpha)
    !python "{root_dir}/RobustVideoMattingCLI/rvm_cli.py" --input_path "{videoFramesFolder}" --output_alpha "{root_dir}/alpha.mp4"
    extractFrames(f"{root_dir}/alpha.mp4", f"{videoFramesAlpha}", 1, 0, 999999999)
  if mask_source == 'mask_video':
    videoFramesAlpha = videoFramesFolder+'Alpha'
    createPath(videoFramesAlpha)
    maskVideoFrames = videoFramesFolder+'Mask'
    createPath(maskVideoFrames)
    extractFrames(mask_video_path, f"{maskVideoFrames}", extract_nth_frame, start_frame, end_frame)
    !python "{root_dir}/RobustVideoMattingCLI/rvm_cli.py" --input_path "{maskVideoFrames}" --output_alpha "{root_dir}/alpha.mp4"
    extractFrames(f"{root_dir}/alpha.mp4", f"{videoFramesAlpha}", 1, 0, 999999999)
else:
  if mask_source == 'init_video':
    videoFramesAlpha = videoFramesFolder
  if mask_source == 'mask_video':
    videoFramesAlpha = videoFramesFolder+'Alpha'
    createPath(videoFramesAlpha)
    extractFrames(mask_video_path, f"{videoFramesAlpha}", extract_nth_frame, start_frame, end_frame)
    #extract video