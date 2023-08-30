#@title 1.5 Define necessary functions

from typing import Mapping

import mediapipe as mp
import numpy
from PIL import Image


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection  # Only for counting faces.
mp_face_mesh = mp.solutions.face_mesh
mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS
mp_body_connections = mp.solutions.pose_connections.POSE_CONNECTIONS

DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
PoseLandmark = mp.solutions.drawing_styles.PoseLandmark

f_thick = 2
f_rad = 1
right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

# mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec = {}
for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec[edge] = head_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
    face_connection_spec[edge] = left_eye_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
    face_connection_spec[edge] = left_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
#    face_connection_spec[edge] = left_iris_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
    face_connection_spec[edge] = right_eye_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
    face_connection_spec[edge] = right_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
#    face_connection_spec[edge] = right_iris_draw
for edge in mp_face_mesh.FACEMESH_LIPS:
    face_connection_spec[edge] = mouth_draw
iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}


def draw_pupils(image, landmark_list, drawing_spec, halfwidth: int = 2):
    """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
    landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be H,W,C.")
    image_rows, image_cols, image_channels = image.shape
    if image_channels != 3:  # BGR channels
        raise ValueError('Input image must contain three channel bgr data.')
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
                (landmark.HasField('visibility') and landmark.visibility < 0.9) or
                (landmark.HasField('presence') and landmark.presence < 0.5)
        ):
            continue
        if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
            continue
        image_x = int(image_cols*landmark.x)
        image_y = int(image_rows*landmark.y)
        draw_color = None
        if isinstance(drawing_spec, Mapping):
            if drawing_spec.get(idx) is None:
                continue
            else:
                draw_color = drawing_spec[idx].color
        elif isinstance(drawing_spec, DrawingSpec):
            draw_color = drawing_spec.color
        image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
    # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
    return image[:, :, ::-1]


def generate_annotation(
        input_image: Image.Image,
        max_faces: int,
        min_face_size_pixels: int = 0,
        return_annotation_data: bool = False
):
    """
    Find up to 'max_faces' inside the provided input image.
    If min_face_size_pixels is provided and nonzero it will be used to filter faces that occupy less than this many
    pixels in the image.
    If return_annotation_data is TRUE (default: false) then in addition to returning the 'detected face' image, three
    additional parameters will be returned: faces before filtering, faces after filtering, and an annotation image.
    The faces_before_filtering return value is the number of faces detected in an image with no filtering.
    faces_after_filtering is the number of faces remaining after filtering small faces.
    :return:
      If 'return_annotation_data==True', returns (numpy array, numpy array, int, int).
      If 'return_annotation_data==False' (default), returns a numpy array.
    """
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
    ) as facemesh:
        img_rgb = numpy.asarray(input_image)
        results = facemesh.process(img_rgb).multi_face_landmarks
        if results is None:
          return None
        faces_found_before_filtering = len(results)

        # Filter faces that are too small
        filtered_landmarks = []
        for lm in results:
            landmarks = lm.landmark
            face_rect = [
                landmarks[0].x,
                landmarks[0].y,
                landmarks[0].x,
                landmarks[0].y,
            ]  # Left, up, right, down.
            for i in range(len(landmarks)):
                face_rect[0] = min(face_rect[0], landmarks[i].x)
                face_rect[1] = min(face_rect[1], landmarks[i].y)
                face_rect[2] = max(face_rect[2], landmarks[i].x)
                face_rect[3] = max(face_rect[3], landmarks[i].y)
            if min_face_size_pixels > 0:
                face_width = abs(face_rect[2] - face_rect[0])
                face_height = abs(face_rect[3] - face_rect[1])
                face_width_pixels = face_width * input_image.size[0]
                face_height_pixels = face_height * input_image.size[1]
                face_size = min(face_width_pixels, face_height_pixels)
                if face_size >= min_face_size_pixels:
                    filtered_landmarks.append(lm)
            else:
                filtered_landmarks.append(lm)

        faces_remaining_after_filtering = len(filtered_landmarks)

        # Annotations are drawn in BGR for some reason, but we don't need to flip a zero-filled image at the start.
        empty = numpy.zeros_like(img_rgb)

        # Draw detected faces:
        for face_landmarks in filtered_landmarks:
            mp_drawing.draw_landmarks(
                empty,
                face_landmarks,
                connections=face_connection_spec.keys(),
                landmark_drawing_spec=None,
                connection_drawing_spec=face_connection_spec
            )
            draw_pupils(empty, face_landmarks, iris_landmark_spec, 2)

        # Flip BGR back to RGB.
        empty = reverse_channels(empty)

        # We might have to generate a composite.
        if return_annotation_data:
            # Note that we're copying the input image AND flipping the channels so we can draw on top of it.
            annotated = reverse_channels(numpy.asarray(input_image)).copy()
            for face_landmarks in filtered_landmarks:
                mp_drawing.draw_landmarks(
                    empty,
                    face_landmarks,
                    connections=face_connection_spec.keys(),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=face_connection_spec
                )
                draw_pupils(empty, face_landmarks, iris_landmark_spec, 2)
            annotated = reverse_channels(annotated)

        if not return_annotation_data:
            return empty
        else:
            return empty, annotated, faces_found_before_filtering, faces_remaining_after_filtering



# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
import PIL


def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)

def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out

def regen_perlin():
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()

def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2]//4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn//4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts

cutout_debug = False
padargs = {}

class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size,
                 Overview=4, 
                 InnerCrop = 0, IC_Size_Pow=0.5, IC_Grey_P = 0.2
                 ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        if args.animation_mode == 'None':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.5),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.1),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          ])
        elif args.animation_mode == 'Video Input Legacy':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.5),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomPerspective(distortion_scale=0.4, p=0.7),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.15),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          ])
        elif  args.animation_mode == '2D' or args.animation_mode == 'Video Input':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.4),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.1),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
          ])
          

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1,3,self.cut_size,self.cut_size] 
        output_shape_2 = [1,3,self.cut_size+2,self.cut_size+2]
        pad_input = F.pad(input,((sideY-max_size)//2,(sideY-max_size)//2,(sideX-max_size)//2,(sideX-max_size)//2), **padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview>0:
            if self.Overview<=4:
                if self.Overview>=1:
                    cutouts.append(cutout)
                if self.Overview>=2:
                    cutouts.append(gray(cutout))
                if self.Overview>=3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview==4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                if is_colab:
                    TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("/content/cutout_overview0.jpg",quality=99)
                else:
                    TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("cutout_overview0.jpg",quality=99)

                              
        if self.InnerCrop >0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                if is_colab:
                    TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("/content/cutout_InnerCrop.jpg",quality=99)
                else:
                    TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("cutout_InnerCrop.jpg",quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True: cutouts=self.augs(cutouts)
        return cutouts

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)     

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

def get_image_from_lat(lat):
    img = sd_model.decode_first_stage(lat.cuda())[0]
    return TF.to_pil_image(img.add(1).div(2).clamp(0, 1))


def get_lat_from_pil(frame):
    print(frame.shape, 'frame2pil.shape')
    frame = np.array(frame)
    frame = (frame/255.)[None,...].transpose(0, 3, 1, 2)
    frame = 2*torch.from_numpy(frame).float().cuda()-1.
    return sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame))


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0/200.0

def get_sched_from_json(frame_num, sched_json, blend=False):
  
  frame_num = int(frame_num)
  frame_num = max(frame_num, 0)
  sched_int = {}
  for key in sched_json.keys():
    sched_int[int(key)] = sched_json[key]
  sched_json = sched_int
  keys = sorted(list(sched_json.keys())); #print(keys)
  try: 
    frame_num = min(frame_num,max(keys)) #clamp frame num to 0:max(keys) range
  except:
    pass
    
  # print('clamped frame num ', frame_num)
  if frame_num in keys:
    return sched_json[frame_num]; #print('frame in keys')
  if frame_num not in keys:
    for i in range(len(keys)-1):
      k1 = keys[i]
      k2 = keys[i+1]
      if frame_num > k1 and frame_num < k2:
        if not blend: 
            print('frame between keys, no blend')
            return sched_json[k1]
        if blend:
            total_dist = k2-k1
            dist_from_k1 = frame_num - k1
            return sched_json[k1]*(1 - dist_from_k1/total_dist) + sched_json[k2]*(dist_from_k1/total_dist)
      #else: print(f'frame {frame_num} not in {k1} {k2}')
  return 0

def get_scheduled_arg(frame_num, schedule):
    if isinstance(schedule, list):
      return schedule[frame_num] if frame_num<len(schedule) else schedule[-1]
    if isinstance(schedule, dict):
      return get_sched_from_json(frame_num, schedule, blend=blend_json_schedules)



def img2tensor(img, size=None):
    img = img.convert('RGB')
    if size: img = img.resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...].cuda()

def warp_towards_init_fn(sample_pil, init_image):
  print('sample, init', type(sample_pil), type(init_image))
  size = sample_pil.size
  sample = img2tensor(sample_pil)
  init_image = img2tensor(init_image, size)
  flo = get_flow(init_image, sample, raft_model, half=flow_lq)
  # flo = get_flow(sample, init_image, raft_model, half=flow_lq)
  warped = warp(sample_pil, sample_pil, flo_path=flo, blend=1, weights_path=None, 
                          forward_clip=0, pad_pct=padding_ratio, padding_mode=padding_mode, 
                          inpaint_blend=inpaint_blend, warp_mul=warp_strength)
  return warped




def do_3d_step(img_filepath, frame_num, forward_clip):
            global warp_mode, filename, match_frame, first_frame
            global first_frame_source
            if warp_mode == 'use_image':
              prev = Image.open(img_filepath)
            # if warp_mode == 'use_latent':
            #   prev = torch.load(img_filepath[:-4]+'_lat.pt')
            
            frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
            frame2 = Image.open(f'{videoFramesFolder}/{frame_num+1:06}.jpg')
            
              
            flo_path = f"{flo_folder}/{frame1_path.split('/')[-1]}.npy"

            if flow_override_map not in [[],'', None]:
                 mapped_frame_num = int(get_scheduled_arg(frame_num, flow_override_map))
                 frame_override_path = f'{videoFramesFolder}/{mapped_frame_num:06}.jpg'
                 flo_path = f"{flo_folder}/{frame_override_path.split('/')[-1]}.npy"

            if use_background_mask and not apply_mask_after_warp:
              # if turbo_mode & (frame_num % int(turbo_steps) != 0):
              #   print('disabling mask for turbo step, will be applied during turbo blend')
              # else:
                if VERBOSE:print('creating bg mask for frame ', frame_num)
                frame2 = apply_mask(frame2, frame_num, background, background_source, invert_mask)
                # frame2.save(f'frame2_{frame_num}.jpg')
            # init_image = 'warped.png'
            flow_blend = get_scheduled_arg(frame_num, flow_blend_schedule)
            printf('flow_blend: ', flow_blend, 'frame_num:', frame_num, 'len(flow_blend_schedule):', len(flow_blend_schedule))
            weights_path = None
            forward_clip = forward_weights_clip
            if check_consistency: 
              if reverse_cc_order:
                weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg" 
              else: 
                weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"

            if turbo_mode & (frame_num % int(turbo_steps) != 0):
              if forward_weights_clip_turbo_step: 
                forward_clip = forward_weights_clip_turbo_step
              if disable_cc_for_turbo_frames:
                if VERBOSE:print('disabling cc for turbo frames')
                weights_path = None
            if warp_mode == 'use_image':
              prev = Image.open(img_filepath)
              
              if not warp_forward:
                printf('warping')
                warped = warp(prev, frame2, flo_path, blend=flow_blend, weights_path=weights_path, 
                          forward_clip=forward_clip, pad_pct=padding_ratio, padding_mode=padding_mode, 
                          inpaint_blend=inpaint_blend, warp_mul=warp_strength)
              else: 
                flo_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12.npy"
                flo = np.load(flo_path)
                warped = k_means_warp(flo, prev, warp_num_k)
              if colormatch_frame != 'off' and not colormatch_after:
                if not turbo_mode & (frame_num % int(turbo_steps) != 0) or colormatch_turbo:
                  try:
                    print('Matching color before warp to:')
                    filename = get_frame_from_color_mode(colormatch_frame, colormatch_offset, frame_num)
                    match_frame = Image.open(filename)
                    first_frame = match_frame
                    first_frame_source = filename

                  except: 
                    print(traceback.format_exc())
                    print(f'Frame with offset/position {colormatch_offset} not found')
                    if 'init' in colormatch_frame:
                      try: 
                        filename = f'{videoFramesFolder}/{1:06}.jpg'
                        match_frame = Image.open(filename)
                        first_frame = match_frame
                        first_frame_source = filename
                      except: pass
                  print(f'Color matching the 1st frame before warp.')
                  print('Colormatch source - ', first_frame_source)
                  warped = Image.fromarray(match_color_var(first_frame, warped, opacity=color_match_frame_str, f=colormatch_method_fn, regrain=colormatch_regrain))
            if warp_mode == 'use_latent':
              prev = torch.load(img_filepath[:-4]+'_lat.pt')
              warped = warp_lat(prev, frame2, flo_path, blend=flow_blend, weights_path=weights_path, 
                          forward_clip=forward_clip, pad_pct=padding_ratio, padding_mode=padding_mode, 
                          inpaint_blend=inpaint_blend, warp_mul=warp_strength)
            # warped = warped.resize((side_x,side_y), warp_interp)
            

            if use_background_mask and apply_mask_after_warp:
              # if turbo_mode & (frame_num % int(turbo_steps) != 0):
              #   print('disabling mask for turbo step, will be applied during turbo blend')
              #   return warped
              if VERBOSE: print('creating bg mask for frame ', frame_num)
              if warp_mode == 'use_latent':
                warped = apply_mask(warped, frame_num, background, background_source, invert_mask, warp_mode)
              else:
                warped = apply_mask(warped, frame_num, background, background_source, invert_mask, warp_mode)
              # warped.save(f'warped_{frame_num}.jpg')
            
            return warped
   
from tqdm.notebook import trange
import copy

def get_frame_from_color_mode(mode, offset, frame_num):
                      if mode == 'color_video':
                        if VERBOSE:print(f'the color video frame number {offset}.')
                        filename = f'{colorVideoFramesFolder}/{offset+1:06}.jpg'
                      if mode == 'color_video_offset':
                        if VERBOSE:print(f'the color video frame with offset {offset}.')
                        filename = f'{colorVideoFramesFolder}/{frame_num-offset+1:06}.jpg'
                      if mode == 'stylized_frame_offset':
                        if VERBOSE:print(f'the stylized frame with offset {offset}.')
                        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-offset:06}.png'
                      if mode == 'stylized_frame':
                        if VERBOSE:print(f'the stylized frame number {offset}.')
                        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{offset:06}.png'
                      if mode == 'init_frame_offset':
                        if VERBOSE:print(f'the raw init frame with offset {offset}.')
                        filename = f'{videoFramesFolder}/{frame_num-offset+1:06}.jpg'
                      if mode == 'init_frame':
                        if VERBOSE:print(f'the raw init frame number {offset}.')
                        filename = f'{videoFramesFolder}/{offset+1:06}.jpg'
                      return filename

def apply_mask(init_image, frame_num, background, background_source, invert_mask=False, warp_mode='use_image', ):
  global mask_clip_low, mask_clip_high
  if warp_mode == 'use_image':
    size = init_image.size
  if warp_mode == 'use_latent':
    print(init_image.shape)
    size = init_image.shape[-1], init_image.shape[-2]
    size = [o*8 for o in size]
    print('size',size)
  init_image_alpha = Image.open(f'{videoFramesAlpha}/{frame_num+1:06}.jpg').resize(size).convert('L')
  if invert_mask:
    init_image_alpha = ImageOps.invert(init_image_alpha)
  if mask_clip_high < 255 or mask_clip_low > 0:
    arr = np.array(init_image_alpha)
    if mask_clip_high < 255:
      arr = np.where(arr<mask_clip_high, arr, 255)
    if mask_clip_low > 0:
      arr = np.where(arr>mask_clip_low, arr, 0)
    init_image_alpha = Image.fromarray(arr)

  if background == 'color':
    bg = Image.new('RGB', size, background_source)
  if background == 'image':
    bg = Image.open(background_source).convert('RGB').resize(size)
  if background == 'init_video':
    bg = Image.open(f'{videoFramesFolder}/{frame_num+1:06}.jpg').resize(size)
  # init_image.putalpha(init_image_alpha)
  if warp_mode == 'use_image':
    bg.paste(init_image, (0,0), init_image_alpha)
  if warp_mode == 'use_latent':
    #convert bg to latent

    bg = np.array(bg)
    bg = (bg/255.)[None,...].transpose(0, 3, 1, 2)
    bg = 2*torch.from_numpy(bg).float().cuda()-1.
    bg = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(bg))
    bg = bg.cpu().numpy()#[0].transpose(1,2,0)
    init_image_alpha = np.array(init_image_alpha)[::8,::8][None, None, ...]
    init_image_alpha = np.repeat(init_image_alpha, 4, axis = 1)/255
    print(bg.shape, init_image.shape, init_image_alpha.shape, init_image_alpha.max(), init_image_alpha.min())
    bg = init_image*init_image_alpha + bg*(1-init_image_alpha)
  return bg

def softcap(arr, thresh=0.8, q=0.95):
  cap = torch.quantile(abs(arr).float(), q)
  printf('q -----', torch.quantile(abs(arr).float(), torch.Tensor([0.25,0.5,0.75,0.9,0.95,0.99,1]).cuda()))
  cap_ratio = (1-thresh)/(cap-thresh)
  arr = torch.where(arr>thresh, thresh+(arr-thresh)*cap_ratio, arr)
  arr = torch.where(arr<-thresh, -thresh+(arr+thresh)*cap_ratio, arr)
  return arr

def do_run():
  seed = args.seed
  print(range(args.start_frame, args.max_frames))
  if args.animation_mode != "None":
    batchBar = tqdm(total=args.max_frames, desc ="Frames")

  # if (args.animation_mode == 'Video Input') and (args.midas_weight > 0.0):
      # midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model)
  for frame_num in range(args.start_frame, args.max_frames):
      if stop_on_next_loop:
        break
      
      # display.clear_output(wait=True)

      # Print Frame progress if animation mode is on
      if args.animation_mode != "None":
        display.display(batchBar.container)
        batchBar.n = frame_num
        batchBar.update(1)
        batchBar.refresh()
        # display.display(batchBar.container)
        


      
      # Inits if not video frames
      if args.animation_mode != "Video Input Legacy":
        if args.init_image == '':
          init_image = None
        else:
          init_image = args.init_image
        init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
        # init_scale = args.init_scale
        steps = int(get_scheduled_arg(frame_num, steps_schedule))
        style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
        skip_steps = int(steps-steps*style_strength)
        # skip_steps = args.skip_steps

      if args.animation_mode == 'Video Input':
        if frame_num == args.start_frame: 
            steps = int(get_scheduled_arg(frame_num, steps_schedule))
            style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
            skip_steps = int(steps-steps*style_strength)
            # skip_steps = args.skip_steps
            
            # init_scale = args.init_scale
            init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
            # init_latent_scale = args.init_latent_scale
            init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)
            init_image = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
            if use_background_mask:
              init_image_pil = Image.open(init_image)
              init_image_pil = apply_mask(init_image_pil, frame_num, background, background_source, invert_mask)
              init_image_pil.save(f'init_alpha_{frame_num}.png')
              init_image = f'init_alpha_{frame_num}.png'
            if (args.init_image != '') and  args.init_image is not None:
              init_image = args.init_image
              if use_background_mask:
                init_image_pil = Image.open(init_image)
                init_image_pil = apply_mask(init_image_pil, frame_num, background, background_source, invert_mask)
                init_image_pil.save(f'init_alpha_{frame_num}.png')
                init_image = f'init_alpha_{frame_num}.png'
            if VERBOSE:print('init image', args.init_image)
        if frame_num > 0 and frame_num != frame_range[0]:
          # print(frame_num)
 
          first_frame_source = batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.png"
          if os.path.exists(first_frame_source):
              first_frame = Image.open(first_frame_source)
          else: 
              first_frame_source = batchFolder+f"/{batch_name}({batchNum})_{args.start_frame-1:06}.png"
              first_frame = Image.open(first_frame_source)
 
            
          # print(frame_num)
          
          # first_frame = Image.open(batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.png")
          # first_frame_source = batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.png"
          if not fixed_seed:
            seed += 1    
          if resume_run and frame_num == start_frame:
            print('if resume_run and frame_num == start_frame')
            img_filepath = batchFolder+f"/{batch_name}({batchNum})_{start_frame-1:06}.png"
            if turbo_mode and frame_num > turbo_preroll:
              shutil.copyfile(img_filepath, 'oldFrameScaled.png')
            else: 
              shutil.copyfile(img_filepath, 'prevFrame.png')
          else:
            # img_filepath = '/content/prevFrame.png' if is_colab else 'prevFrame.png'
            img_filepath = 'prevFrame.png'

          next_step_pil = do_3d_step(img_filepath, frame_num,  forward_clip=forward_weights_clip)
          if warp_mode == 'use_image':
            next_step_pil.save('prevFrameScaled.png')
          else: 
            # init_image = 'prevFrameScaled_lat.pt'
            # next_step_pil.save('prevFrameScaled.png')
            torch.save(next_step_pil, 'prevFrameScaled_lat.pt')
            
          steps = int(get_scheduled_arg(frame_num, steps_schedule))
          style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
          skip_steps = int(steps-steps*style_strength)
          # skip_steps = args.calc_frames_skip_steps

          ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
          if turbo_mode:
            if frame_num == turbo_preroll: #start tracking oldframe
              if warp_mode == 'use_image': 
                next_step_pil.save('oldFrameScaled.png')#stash for later blending
              if warp_mode == 'use_latent': 
                # lat_from_img = get_lat/_from_pil(next_step_pil) 
                torch.save(next_step_pil, 'oldFrameScaled_lat.pt')        
            elif frame_num > turbo_preroll:
              #set up 2 warped image sequences, old & new, to blend toward new diff image
              if warp_mode == 'use_image': 
                old_frame = do_3d_step('oldFrameScaled.png', frame_num, forward_clip=forward_weights_clip_turbo_step)
                old_frame.save('oldFrameScaled.png')
              if warp_mode == 'use_latent': 
                old_frame = do_3d_step('oldFrameScaled.png', frame_num, forward_clip=forward_weights_clip_turbo_step)

                # lat_from_img = get_lat_from_pil(old_frame) 
                torch.save(old_frame, 'oldFrameScaled_lat.pt')
              if frame_num % int(turbo_steps) != 0: 
                print('turbo skip this frame: skipping clip diffusion steps')
                filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
                blend_factor = ((frame_num % int(turbo_steps))+1)/int(turbo_steps)
                print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                if warp_mode == 'use_image': 
                  newWarpedImg = cv2.imread('prevFrameScaled.png')#this is already updated..
                  oldWarpedImg = cv2.imread('oldFrameScaled.png')
                  blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg,1-blend_factor, 0.0)
                  cv2.imwrite(f'{batchFolder}/{filename}',blendedImage)
                  next_step_pil.save(f'{img_filepath}') # save it also as prev_frame to feed next iteration
                if warp_mode == 'use_latent': 
                  newWarpedImg = torch.load('prevFrameScaled_lat.pt')#this is already updated..
                  oldWarpedImg = torch.load('oldFrameScaled_lat.pt')
                  blendedImage = newWarpedImg*(blend_factor)+oldWarpedImg*(1-blend_factor)
                  blendedImage = get_image_from_lat(blendedImage).save(f'{batchFolder}/{filename}')
                  torch.save(next_step_pil,f'{img_filepath[:-4]}_lat.pt')


                if turbo_frame_skips_steps is not None:
                    if warp_mode == 'use_image': 
                      oldWarpedImg = cv2.imread('prevFrameScaled.png')
                      cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later 
                    print('clip/diff this frame - generate clip diff image')
                    if warp_mode == 'use_latent': 
                      oldWarpedImg = torch.load('prevFrameScaled_lat.pt')
                      torch.save(oldWarpedImg, f'oldFrameScaled_lat.pt',)#swap in for blending later 
                    skip_steps = math.floor(steps * turbo_frame_skips_steps)
                else: continue
              else:
                #if not a skip frame, will run diffusion and need to blend.
                if warp_mode == 'use_image': 
                      oldWarpedImg = cv2.imread('prevFrameScaled.png')
                      cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later 
                print('clip/diff this frame - generate clip diff image')
                if warp_mode == 'use_latent': 
                      oldWarpedImg = torch.load('prevFrameScaled_lat.pt')
                      torch.save(oldWarpedImg, f'oldFrameScaled_lat.pt',)#swap in for blending later 
                # oldWarpedImg = cv2.imread('prevFrameScaled.png')
                # cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later 
                print('clip/diff this frame - generate clip diff image')
          if warp_mode == 'use_image':
            init_image = 'prevFrameScaled.png'
          else: 
            init_image = 'prevFrameScaled_lat.pt'
          if use_background_mask:
            if warp_mode == 'use_latent':
              # pass
              latent = apply_mask(latent.cpu(), frame_num, background, background_source, invert_mask, warp_mode)#.save(init_image)
         
            if warp_mode == 'use_image':
              apply_mask(Image.open(init_image), frame_num, background, background_source, invert_mask).save(init_image)
          # init_scale = args.frames_scale
          init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
          # init_latent_scale = args.frames_latent_scale
          init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)


      loss_values = []
  
      if seed is not None:
          np.random.seed(seed)
          random.seed(seed)
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
          torch.backends.cudnn.deterministic = True
  
      target_embeds, weights = [], []
      
      if args.prompts_series is not None and frame_num >= len(args.prompts_series):
        frame_prompt = args.prompts_series[-1]
        frame_prompt = get_sched_from_json(frame_num, args.prompts_series, blend=False)
      elif args.prompts_series is not None:
        frame_prompt = args.prompts_series[frame_num]
        frame_prompt = get_sched_from_json(frame_num, args.prompts_series, blend=False)
      else:
        frame_prompt = []
      
      if VERBOSE:print(args.image_prompts_series)
      if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
        image_prompt = args.image_prompts_series[-1]
      elif args.image_prompts_series is not None:
        image_prompt = args.image_prompts_series[frame_num]
      else:
        image_prompt = []

      if VERBOSE:print(f'Frame {frame_num} Prompt: {frame_prompt}')


  
      init = None



      image_display = Output()
      for i in range(args.n_batches):
          if args.animation_mode == 'None':
            display.clear_output(wait=True)
            batchBar = tqdm(range(args.n_batches), desc ="Batches")
            batchBar.n = i
            batchBar.refresh()
          print('')
          display.display(image_display)
          gc.collect()
          torch.cuda.empty_cache()
          steps = int(get_scheduled_arg(frame_num, steps_schedule))
          style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
          skip_steps = int(steps-steps*style_strength)


          if perlin_init:
              init = regen_perlin()

          consistency_mask = None
          if (check_consistency or (model_version == 'v1_inpainting')) and frame_num>0:
            frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
            if reverse_cc_order:
              weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
            else: 
              weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"
            consistency_mask = load_cc(weights_path, blur=consistency_blur)

          if diffusion_model == 'stable_diffusion':
            if VERBOSE: print(args.side_x, args.side_y, init_image)
            # init = Image.open(fetch(init_image)).convert('RGB')
            
            # init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
            # init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
            text_prompt = copy.copy(args.prompts_series[frame_num])
            text_prompt = [re.sub('\<(.*?)\>', '', o).strip(' ') for o in text_prompt] #remove loras from prompt
            used_loras, used_loras_weights = get_loras_weights_for_frame(frame_num, new_prompt_loras)
            if VERBOSE:
              print('used_loras, used_loras_weights', used_loras, used_loras_weights)
            # used_loras_weights = [o for o in used_loras_weights if o is not None else 0.]
            load_loras(used_loras,used_loras_weights)
            caption = get_caption(frame_num)
            if caption:
              # print('args.prompt_series',args.prompts_series[frame_num])
              if '{caption}' in text_prompt[0]:
                print('Replacing ', '{caption}', 'with ', caption)
                text_prompt[0] = text_prompt[0].replace('{caption}', caption)
            neg_prompt = get_sched_from_json(frame_num, args.neg_prompts_series, blend=False)
            if args.neg_prompts_series is not None:
              rec_prompt = get_sched_from_json(frame_num, args.rec_prompts_series, blend=False)
              if caption and '{caption}' in rec_prompt[0]:
                  print('Replacing ', '{caption}', 'with ', caption)
                  rec_prompt[0] = rec_prompt[0].replace('{caption}', caption)
            else:
              rec_prompt = copy.copy(text_prompt)

            if VERBOSE:
              print(neg_prompt, 'neg_prompt')
              print('init_scale pre sd run', init_scale)
            # init_latent_scale = args.init_latent_scale
            # if frame_num>0:
            #   init_latent_scale = args.frames_latent_scale
            steps = int(get_scheduled_arg(frame_num, steps_schedule))
            init_scale = get_scheduled_arg(frame_num, init_scale_schedule)
            init_latent_scale = get_scheduled_arg(frame_num, latent_scale_schedule)
            style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
            skip_steps = int(steps-steps*style_strength)
            cfg_scale = get_scheduled_arg(frame_num, cfg_scale_schedule)
            image_scale = get_scheduled_arg(frame_num, image_scale_schedule)
            if VERBOSE:printf('skip_steps b4 run_sd: ', skip_steps)

            deflicker_src = {
                'processed1':f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png',
                'raw1': f'{videoFramesFolder}/{frame_num:06}.jpg',
                'raw2': f'{videoFramesFolder}/{frame_num+1:06}.jpg',
            }

            init_grad_img = None
            if init_grad: init_grad_img = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
            #setup depth source
            if depth_source == 'init': 
              depth_init = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
            if depth_source == 'stylized': 
              depth_init = init_image
            if depth_source == 'cond_video': 
              depth_init = f'{condVideoFramesFolder}/{frame_num+1:06}.jpg'

            #setup temporal source 
            if temporalnet_source =='init':
              prev_frame = f'{videoFramesFolder}/{frame_num:06}.jpg'
            if temporalnet_source == 'stylized':
              prev_frame = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png'
            if temporalnet_source == 'cond_video':
              prev_frame = f'{condVideoFramesFolder}/{frame_num:06}.jpg'
            if not os.path.exists(prev_frame):
              if temporalnet_skip_1st_frame: 
                print('prev_frame not found, replacing 1st videoframe init')
                prev_frame = None
              else: 
                prev_frame = f'{videoFramesFolder}/{frame_num+1:06}.jpg'

            #setup rec noise source
            if rec_source == 'stylized':
              rec_frame = init_image
            elif rec_source == 'init':
              rec_frame = f'{videoFramesFolder}/{frame_num+1:06}.jpg'


            #setгp masks for inpainting model
            if model_version == 'v1_inpainting':
              if inpainting_mask_source == 'consistency_mask':
                depth_init = consistency_mask 
              if inpainting_mask_source in ['none', None,'', 'None', 'off']:
                depth_init = None
              if inpainting_mask_source == 'cond_video': depth_init = f'{condVideoFramesFolder}/{frame_num+1:06}.jpg'
              # print('depth_init0',depth_init)

            sample, latent, depth_img = run_sd(args, init_image=init_image, skip_timesteps=skip_steps, H=args.side_y, 
                             W=args.side_x, text_prompt=text_prompt, neg_prompt=neg_prompt, steps=steps, 
                             seed=seed, init_scale = init_scale, init_latent_scale=init_latent_scale, depth_init=depth_init, 
                             cfg_scale=cfg_scale, image_scale = image_scale, cond_fn=None, 
                             init_grad_img=init_grad_img, consistency_mask=consistency_mask,
                             frame_num=frame_num, deflicker_src=deflicker_src, prev_frame=prev_frame, rec_prompt=rec_prompt, rec_frame=rec_frame)


            # depth_img.save(f'{root_dir}/depth_{frame_num}.png')
            filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
            # if warp_mode == 'use_raw':torch.save(sample,f'{batchFolder}/{filename[:-4]}_raw.pt')
            if warp_mode == 'use_latent':
              torch.save(latent,f'{batchFolder}/{filename[:-4]}_lat.pt')
            samples = sample*(steps-skip_steps)
            samples = [{"pred_xstart": sample} for sample in samples]
            # for j, sample in enumerate(samples):
              # print(j, sample["pred_xstart"].size)
            # raise Exception
            if VERBOSE: print(sample[0][0].shape)
            image = sample[0][0]
            if do_softcap:
              image = softcap(image, thresh=softcap_thresh, q=softcap_q)
            image = image.add(1).div(2).clamp(0, 1)
            image = TF.to_pil_image(image)
            if warp_towards_init != 'off' and frame_num!=0:
              if warp_towards_init == 'init':
                warp_init_filename = f'{videoFramesFolder}/{frame_num+1:06}.jpg'
              else:
                warp_init_filename = init_image
              print('warping towards init')
              init_pil = Image.open(warp_init_filename)
              image = warp_towards_init_fn(image, init_pil)
            
            display.clear_output(wait=True)
            fit(image, display_size).save('progress.png')
            display.display(display.Image('progress.png'))

            if mask_result and check_consistency and frame_num>0:

                        if VERBOSE:print('imitating inpaint')
                        frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
                        weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                        consistency_mask = load_cc(weights_path, blur=consistency_blur)
                         
                        consistency_mask = cv2.GaussianBlur(consistency_mask,
                                                (diffuse_inpaint_mask_blur,diffuse_inpaint_mask_blur),cv2.BORDER_DEFAULT)
                        if diffuse_inpaint_mask_thresh<1:
                          consistency_mask = np.where(consistency_mask<diffuse_inpaint_mask_thresh, 0, 1.)
                        # if dither: 
                        #   consistency_mask = Dither.dither(consistency_mask, 'simple2D', resize=True)
                        
                       
                        

                        # consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
                        if warp_mode == 'use_image':
                          consistency_mask = cv2.GaussianBlur(consistency_mask,
                                                (3,3),cv2.BORDER_DEFAULT)
                          init_img_prev = Image.open(init_image)
                          if VERBOSE:print(init_img_prev.size, consistency_mask.shape, image.size)
                          cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                          image_masked = np.array(image)*(1-consistency_mask) + np.array(init_img_prev)*(consistency_mask)

                          # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                          image_masked = Image.fromarray(image_masked.round().astype('uint8'))
                          # image = image_masked.resize(image.size, warp_interp)
                          image = image_masked
                        if warp_mode == 'use_latent':
                          if invert_mask: consistency_mask = 1-consistency_mask
                          init_lat_prev = torch.load('prevFrameScaled_lat.pt')
                          sample_masked = sd_model.decode_first_stage(latent.cuda())[0]
                          image_prev = TF.to_pil_image(sample_masked.add(1).div(2).clamp(0, 1))


                          cc_small = consistency_mask[::8,::8,0]
                          latent = latent.cpu()*(1-cc_small)+init_lat_prev*cc_small
                          torch.save(latent, 'prevFrameScaled_lat.pt')
                          
                          # image_prev = Image.open(f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png')
                          torch.save(latent, 'prevFrame_lat.pt')
                          # cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                          # image_prev = Image.open('prevFrameScaled.png')
                          image_masked = np.array(image)*(1-consistency_mask) + np.array(image_prev)*(consistency_mask)

                          # # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                          image_masked = Image.fromarray(image_masked.round().astype('uint8'))
                          # image = image_masked.resize(image.size, warp_interp)
                          image = image_masked

            if (frame_num > args.start_frame) or ('color_video' in normalize_latent):
                global first_latent
                global first_latent_source
                def get_frame_from_color_mode(mode, offset, frame_num):
                      if mode == 'color_video':
                        if VERBOSE:print(f'the color video frame number {offset}.')
                        filename = f'{colorVideoFramesFolder}/{offset+1:06}.jpg'
                      if mode == 'color_video_offset':
                        if VERBOSE:print(f'the color video frame with offset {offset}.')
                        filename = f'{colorVideoFramesFolder}/{frame_num-offset+1:06}.jpg'
                      if mode == 'stylized_frame_offset':
                        if VERBOSE:print(f'the stylized frame with offset {offset}.')
                        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-offset:06}.png'
                      if mode == 'stylized_frame':
                        if VERBOSE:print(f'the stylized frame number {offset}.')
                        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{offset:06}.png'
                      if mode == 'init_frame_offset':
                        if VERBOSE:print(f'the raw init frame with offset {offset}.')
                        filename = f'{videoFramesFolder}/{frame_num-offset+1:06}.jpg'
                      if mode == 'init_frame':
                        if VERBOSE:print(f'the raw init frame number {offset}.')
                        filename = f'{videoFramesFolder}/{offset+1:06}.jpg'
                      return filename
                if 'frame' in normalize_latent:
                  def img2latent(img_path):
                    frame2 = Image.open(img_path)
                    frame2pil = frame2.convert('RGB').resize(image.size,warp_interp)
                    frame2pil = np.array(frame2pil)
                    frame2pil = (frame2pil/255.)[None,...].transpose(0, 3, 1, 2)
                    frame2pil = 2*torch.from_numpy(frame2pil).float().cuda()-1.
                    frame2pil = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame2pil))
                    return frame2pil

                  try:
                    if VERBOSE:print('Matching latent to:')
                    filename = get_frame_from_color_mode(normalize_latent, normalize_latent_offset, frame_num)
                    match_latent = img2latent(filename)
                    first_latent = match_latent
                    first_latent_source = filename
                    # print(first_latent_source, first_latent)
                  except: 
                    if VERBOSE:print(traceback.format_exc())
                    print(f'Frame with offset/position {normalize_latent_offset} not found')
                    if 'init' in normalize_latent:
                      try: 
                        filename = f'{videoFramesFolder}/{0:06}.jpg'
                        match_latent = img2latent(filename)
                        first_latent = match_latent
                        first_latent_source = filename
                      except: pass
                    print(f'Color matching the 1st frame.')

                if colormatch_frame != 'off' and colormatch_after:
                  if not turbo_mode & (frame_num % int(turbo_steps) != 0) or colormatch_turbo:
                    try:
                      print('Matching color to:')
                      filename = get_frame_from_color_mode(colormatch_frame, colormatch_offset)
                      match_frame = Image.open(filename)
                      first_frame = match_frame
                      first_frame_source = filename

                    except: 
                      print(f'Frame with offset/position {colormatch_offset} not found')
                      if 'init' in colormatch_frame:
                        try: 
                          filename = f'{videoFramesFolder}/{1:06}.jpg'
                          match_frame = Image.open(filename)
                          first_frame = match_frame
                          first_frame_source = filename
                        except: pass
                      print(f'Color matching the 1st frame.')
                    print('Colormatch source - ', first_frame_source)
                    image = Image.fromarray(match_color_var(first_frame, 
                        image, opacity=color_match_frame_str, f=colormatch_method_fn, 
                        regrain=colormatch_regrain))




            if frame_num == args.start_frame:
              save_settings()
            if args.animation_mode != "None":
                          # sys.exit(os.getcwd(), 'cwd')
              if warp_mode == 'use_image':           
                image.save('prevFrame.png')
              else: 
                torch.save(latent, 'prevFrame_lat.pt')
            filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
            image.save(f'{batchFolder}/{filename}')
            # np.save(latent, f'{batchFolder}/{filename[:-4]}.npy')
            if args.animation_mode == 'Video Input':
                          # If turbo, save a blended image
                          if turbo_mode and frame_num > args.start_frame:
                            # Mix new image with prevFrameScaled
                            blend_factor = (1)/int(turbo_steps)
                            if warp_mode == 'use_image':
                              newFrame = cv2.imread('prevFrame.png') # This is already updated..
                              prev_frame_warped = cv2.imread('prevFrameScaled.png')
                              blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                              cv2.imwrite(f'{batchFolder}/{filename}',blendedImage)
                            if warp_mode == 'use_latent':
                              newFrame = torch.load('prevFrame_lat.pt').cuda()
                              prev_frame_warped = torch.load('prevFrameScaled_lat.pt').cuda()
                              blendedImage = newFrame*(blend_factor)+prev_frame_warped*(1-blend_factor)
                              blendedImage = get_image_from_lat(blendedImage)
                              blendedImage.save(f'{batchFolder}/{filename}')

            else:
                            image.save(f'{batchFolder}/{filename}')
                            image.save('prevFrameScaled.png')

          # with run_display:
          # display.clear_output(wait=True)
          # o = 0
          # for j, sample in enumerate(samples):    
          #   cur_t -= 1
          #   # if (cur_t <= stop_early-2): 
          #   #   print(cur_t)
          #   #   break
          #   intermediateStep = False
          #   if args.steps_per_checkpoint is not None:
          #       if j % steps_per_checkpoint == 0 and j > 0:
          #         intermediateStep = True
          #   elif j in args.intermediate_saves:
          #     intermediateStep = True
          #   with image_display:
          #     if j % args.display_rate == 0 or cur_t == -1 or cur_t == stop_early-1 or intermediateStep == True:
                  
                  
          #         for k, image in enumerate(sample['pred_xstart']):
          #             # tqdm.write(f'Batch {i}, step {j}, output {k}:')
          #             current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
          #             percent = math.ceil(j/total_steps*100)
          #             if args.n_batches > 0:
          #               #if intermediates are saved to the subfolder, don't append a step or percentage to the name
          #               if (cur_t == -1 or cur_t == stop_early-1) and args.intermediates_in_subfolder is True:
          #                 save_num = f'{frame_num:06}' if animation_mode != "None" else i
                      #     filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
                      #   else:
                      #     #If we're working with percentages, append it
                      #     if args.steps_per_checkpoint is not None:
                      #       filename = f'{args.batch_name}({args.batchNum})_{i:06}-{percent:02}%.png'
                      #     # Or else, iIf we're working with specific steps, append those
                      #     else:
                      #       filename = f'{args.batch_name}({args.batchNum})_{i:06}-{j:03}.png'
                      # image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                      # if frame_num > 0:
                      #   print('times per image', o); o+=1
                      #   image = Image.fromarray(match_color_var(first_frame, image, f=PT.lab_transfer))
                      #   # image.save(f'/content/{frame_num}_{cur_t}_{o}.jpg')
                      #   # image = Image.fromarray(match_color_var(first_frame, image))

                      # #reapply init image on top of 
                      # if mask_result and check_consistency and frame_num>0:
                      #   diffuse_inpaint_mask_blur = 15
                      #   diffuse_inpaint_mask_thresh = 220
                      #   print('imitating inpaint')
                      #   frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
                      #   weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                      #   consistency_mask = load_cc(weights_path, blur=consistency_blur)
                      #   consistency_mask = cv2.GaussianBlur(consistency_mask,
                      #                           (diffuse_inpaint_mask_blur,diffuse_inpaint_mask_blur),cv2.BORDER_DEFAULT)
                      #   consistency_mask = np.where(consistency_mask<diffuse_inpaint_mask_thresh/255., 0, 1.)
                      #   consistency_mask = cv2.GaussianBlur(consistency_mask,
                      #                           (3,3),cv2.BORDER_DEFAULT)

                      #   # consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
                      #   init_img_prev = Image.open(init_image)
                      #   print(init_img_prev.size, consistency_mask.shape, image.size)
                      #   cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                      #   image_masked = np.array(image)*(1-consistency_mask) + np.array(init_img_prev)*(consistency_mask)

                      #   # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                      #   image_masked = Image.fromarray(image_masked.round().astype('uint8'))
                      #   # image = image_masked.resize(image.size, warp_interp)
                      #   image = image_masked

                      # if j % args.display_rate == 0 or cur_t == -1 or cur_t == stop_early-1:
                      #   image.save('progress.png')
                      #   display.clear_output(wait=True)
                      #   display.display(display.Image('progress.png'))
                      # if args.steps_per_checkpoint is not None:
                      #   if j % args.steps_per_checkpoint == 0 and j > 0:
                      #     if args.intermediates_in_subfolder is True:
                      #       image.save(f'{partialFolder}/{filename}')
                      #     else:
                      #       image.save(f'{batchFolder}/{filename}')
                      # else:
                      #   if j in args.intermediate_saves:
                      #     if args.intermediates_in_subfolder is True:
                      #       image.save(f'{partialFolder}/{filename}')
                      #     else:
                      #       image.save(f'{batchFolder}/{filename}')
                      # if (cur_t == -1) | (cur_t == stop_early-1):
                      #   if cur_t == stop_early-1: print('early stopping')
                        # if frame_num == 0:
                        #   save_settings()
                        # if args.animation_mode != "None":
                        #   # sys.exit(os.getcwd(), 'cwd')
                        #   image.save('prevFrame.png')
                        # image.save(f'{batchFolder}/{filename}')
                        # if args.animation_mode == 'Video Input':
                        #   # If turbo, save a blended image
                        #   if turbo_mode and frame_num > 0:
                        #     # Mix new image with prevFrameScaled
                        #     blend_factor = (1)/int(turbo_steps)
                        #     newFrame = cv2.imread('prevFrame.png') # This is already updated..
                        #     prev_frame_warped = cv2.imread('prevFrameScaled.png')
                        #     blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                        #     cv2.imwrite(f'{batchFolder}/{filename}',blendedImage)
                        #   else:
                        #     image.save(f'{batchFolder}/{filename}')


                        # if frame_num != args.max_frames-1:
                        #   display.clear_output()
          
          plt.plot(np.array(loss_values), 'r')
  batchBar.close()

def save_settings():
  settings_out = batchFolder+f"/settings"
  os.makedirs(settings_out, exist_ok=True)
  setting_list = {
    'text_prompts': text_prompts,
    'user_comment':user_comment,
    'image_prompts': image_prompts,
    'range_scale': range_scale,
    'sat_scale': sat_scale,
    'max_frames': max_frames,
    'interp_spline': interp_spline,
    'init_image': init_image,
    'clamp_grad': clamp_grad,
    'clamp_max': clamp_max,
    'seed': seed,
    'width': width_height[0],
    'height': width_height[1],
    'diffusion_model': diffusion_model,
    'diffusion_steps': diffusion_steps,
    'max_frames': max_frames,
    'video_init_path':video_init_path,
    'extract_nth_frame':extract_nth_frame,
    'flow_video_init_path':flow_video_init_path,
    'flow_extract_nth_frame':flow_extract_nth_frame,
    'video_init_seed_continuity': video_init_seed_continuity,
    'turbo_mode':turbo_mode,
    'turbo_steps':turbo_steps,
    'turbo_preroll':turbo_preroll,
    'flow_warp':flow_warp,
    'check_consistency':check_consistency,
    'turbo_frame_skips_steps' : turbo_frame_skips_steps,
    'forward_weights_clip' : forward_weights_clip,
    'forward_weights_clip_turbo_step' : forward_weights_clip_turbo_step,
    'padding_ratio':padding_ratio,
    'padding_mode':padding_mode,
    'consistency_blur':consistency_blur,
    'inpaint_blend':inpaint_blend,
    'match_color_strength':match_color_strength,
    'high_brightness_threshold':high_brightness_threshold,
    'high_brightness_adjust_ratio':high_brightness_adjust_ratio,
    'low_brightness_threshold':low_brightness_threshold,
    'low_brightness_adjust_ratio':low_brightness_adjust_ratio,
    'stop_early': stop_early,
    'high_brightness_adjust_fix_amount': high_brightness_adjust_fix_amount,
    'low_brightness_adjust_fix_amount': low_brightness_adjust_fix_amount,
    'max_brightness_threshold':max_brightness_threshold,
    'min_brightness_threshold':min_brightness_threshold,
    'enable_adjust_brightness':enable_adjust_brightness,
    'dynamic_thresh':dynamic_thresh,
    'warp_interp':warp_interp,
    'fixed_code':fixed_code,
    'blend_code':blend_code,
    'normalize_code': normalize_code,
    'mask_result':mask_result,
    'reverse_cc_order':reverse_cc_order,
    'flow_lq':flow_lq,
    'use_predicted_noise':use_predicted_noise,
    'clip_guidance_scale':clip_guidance_scale,
    'clip_type':clip_type,
    'clip_pretrain':clip_pretrain,
    'missed_consistency_weight':missed_consistency_weight,
    'overshoot_consistency_weight':overshoot_consistency_weight,
    'edges_consistency_weight':edges_consistency_weight,
    'style_strength_schedule':style_strength_schedule,
    'flow_blend_schedule':flow_blend_schedule,
    'steps_schedule':steps_schedule,
    'init_scale_schedule':init_scale_schedule,
    'latent_scale_schedule':latent_scale_schedule,
    'latent_scale_template': latent_scale_template,
    'init_scale_template':init_scale_template,
    'steps_template':steps_template,
    'style_strength_template':style_strength_template,
    'flow_blend_template':flow_blend_template,
    'make_schedules':make_schedules,
    'normalize_latent':normalize_latent,
    'normalize_latent_offset':normalize_latent_offset,
    'colormatch_frame':colormatch_frame,
    'use_karras_noise':use_karras_noise,
    'end_karras_ramp_early':end_karras_ramp_early,
    'use_background_mask':use_background_mask,
    'apply_mask_after_warp':apply_mask_after_warp,
    'background':background,
    'background_source':background_source,
    'mask_source':mask_source,
    'extract_background_mask':extract_background_mask,
    'mask_video_path':mask_video_path,
    'negative_prompts':negative_prompts,
    'invert_mask':invert_mask,
    'warp_strength': warp_strength,
    'flow_override_map':flow_override_map,
    'cfg_scale_schedule':cfg_scale_schedule,
    'respect_sched':respect_sched,
    'color_match_frame_str':color_match_frame_str,
    'colormatch_offset':colormatch_offset,
    'latent_fixed_mean':latent_fixed_mean,
    'latent_fixed_std':latent_fixed_std,
    'colormatch_method':colormatch_method,
    'colormatch_regrain':colormatch_regrain,
    'warp_mode':warp_mode,
    'use_patchmatch_inpaiting':use_patchmatch_inpaiting,
    'blend_latent_to_init':blend_latent_to_init,
    'warp_towards_init':warp_towards_init,
    'init_grad':init_grad,
    'grad_denoised':grad_denoised,
    'colormatch_after':colormatch_after,
    'colormatch_turbo':colormatch_turbo,
    'model_version':model_version,
    'depth_source':depth_source,
    'warp_num_k':warp_num_k,
    'warp_forward':warp_forward,
    'sampler':sampler.__name__,
    'mask_clip':(mask_clip_low, mask_clip_high),
    'inpainting_mask_weight':inpainting_mask_weight , 
    'inverse_inpainting_mask':inverse_inpainting_mask,
    'mask_source':mask_source,
    'model_path':model_path,
    'diff_override':diff_override,
    'image_scale_schedule':image_scale_schedule,
    'image_scale_template':image_scale_template,
    'frame_range': frame_range,
    'detect_resolution' :detect_resolution, 
    'bg_threshold':bg_threshold, 
    'diffuse_inpaint_mask_blur':diffuse_inpaint_mask_blur, 
    'diffuse_inpaint_mask_thresh':diffuse_inpaint_mask_thresh,
    'add_noise_to_latent':add_noise_to_latent,
    'noise_upscale_ratio':noise_upscale_ratio,
    'fixed_seed':fixed_seed,
    'init_latent_fn':init_latent_fn.__name__,
    'value_threshold':value_threshold,
    'distance_threshold':distance_threshold,
    'masked_guidance':masked_guidance,
    'mask_callback':mask_callback,
    'quantize':quantize,
    'cb_noise_upscale_ratio':cb_noise_upscale_ratio,  
    'cb_add_noise_to_latent':cb_add_noise_to_latent,
    'cb_use_start_code':cb_use_start_code,
    'cb_fixed_code':cb_fixed_code,
    'cb_norm_latent':cb_norm_latent,
    'guidance_use_start_code':guidance_use_start_code,
    'offload_model':offload_model,
    'controlnet_preprocess':controlnet_preprocess,
    'small_controlnet_model_path':small_controlnet_model_path,
    'use_scale':use_scale,
    'g_invert_mask':g_invert_mask,
    'controlnet_multimodel':json.dumps(controlnet_multimodel),
    'img_zero_uncond':img_zero_uncond,
    'do_softcap':do_softcap,
    'softcap_thresh':softcap_thresh,
    'softcap_q':softcap_q,
    'deflicker_latent_scale':deflicker_latent_scale,
    'deflicker_scale':deflicker_scale,
    'controlnet_multimodel_mode':controlnet_multimodel_mode,
    'no_half_vae':no_half_vae,
    'temporalnet_source':temporalnet_source,
    'temporalnet_skip_1st_frame':temporalnet_skip_1st_frame,
    'rec_randomness':rec_randomness,
    'rec_source':rec_source,
    'rec_cfg':rec_cfg,
    'rec_prompts':rec_prompts,
    'inpainting_mask_source':inpainting_mask_source,
    'rec_steps_pct':rec_steps_pct,
    'max_faces': max_faces,
    'num_flow_updates':num_flow_updates
  }
  try: 
    with open(f"{settings_out}/{batch_name}({batchNum})_settings.txt", "w+") as f:   #save settings
      json.dump(setting_list, f, ensure_ascii=False, indent=4)
  except Exception as e:
    print(e)
    print('Settings:', setting_list)