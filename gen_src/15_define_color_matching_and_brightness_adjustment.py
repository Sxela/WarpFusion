#@title Define color matching and brightness adjustment
os.chdir(f"{root_dir}/python-color-transfer")
from python_color_transfer.color_transfer import ColorTransfer, Regrain
os.chdir(root_path)

PT = ColorTransfer()
RG = Regrain()

def match_color(stylized_img, raw_img, opacity=1.):
  if opacity > 0:
    img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'),cv2.COLOR_RGB2BGR)
    img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'),cv2.COLOR_RGB2BGR)
    # img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
    img_arr_col = PT.pdf_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
    img_arr_reg = RG.regrain     (img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
    img_arr_reg = img_arr_reg*opacity+img_arr_in*(1-opacity)
    img_arr_reg = cv2.cvtColor(img_arr_reg.round().astype('uint8'),cv2.COLOR_BGR2RGB)
    return img_arr_reg
  else: return raw_img

from PIL import Image, ImageOps, ImageStat, ImageEnhance

def get_stats(image):
   stat = ImageStat.Stat(image)
   brightness = sum(stat.mean) / len(stat.mean)
   contrast = sum(stat.stddev) / len(stat.stddev)
   return brightness, contrast

#implemetation taken from https://github.com/lowfuel/progrockdiffusion

def adjust_brightness(image):

  brightness, contrast = get_stats(image)
  if brightness > high_brightness_threshold:
    print(" Brightness over threshold. Compensating!")
    filter = ImageEnhance.Brightness(image)
    image = filter.enhance(high_brightness_adjust_ratio)
    image = np.array(image)
    image = np.where(image>high_brightness_threshold, image-high_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
    image = Image.fromarray(image)
  if brightness < low_brightness_threshold:
    print(" Brightness below threshold. Compensating!")
    filter = ImageEnhance.Brightness(image)
    image = filter.enhance(low_brightness_adjust_ratio)
    image = np.array(image)
    image = np.where(image<low_brightness_threshold, image+low_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
    image = Image.fromarray(image)

  image = np.array(image)
  image = np.where(image>max_brightness_threshold, image-high_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
  image = np.where(image<min_brightness_threshold, image+low_brightness_adjust_fix_amount, image).clip(0,255).round().astype('uint8')
  image = Image.fromarray(image)
  return image