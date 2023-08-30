#@title LORA & embedding paths

import torch
from glob import glob
import os
import re
import torch
from typing import Union

def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd

def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception as e:
                    pass

        return res

weight_load_location = 'cpu'
from modules import devices, shared

lora_dir = '/content/drive/MyDrive/AI/StableWarpFusion/loras' #@param {'type':'string'}
if not is_colab and lora_dir.startswith('/content'):
  lora_dir = './loras'
  print('Overriding lora dir to ./loras for non-colab env because you path begins with /content. Change path to desired folder')

custom_embed_dir =   '/content/drive/MyDrive/AI/StableWarpFusion/embeddings' #@param {'type':'string'}
if not is_colab and custom_embed_dir.startswith('/content'):
  custom_embed_dir = './embeddings'
  os.makedirs(custom_embed_dir, exist_ok=True)
  print('Overriding embeddings dir to ./embeddings for non-colab env because you path begins with /content. Change path to desired folder')


def torch_load_file(filename, device):
    result = {}
    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result

# safetensors.torch.load_file = torch_load_file

def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        device = map_location or weight_load_location or devices.get_optimal_device_name()
        pl_sd = torch_load_file(checkpoint_file, device=device)
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd

metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}

re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


class LoraOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.metadata = {}

        _, ext = os.path.splitext(filename)
        if ext.lower() == ".safetensors":
            try:
                self.metadata = read_metadata_from_safetensors(filename)
            except Exception as e:
                print(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.ssmd_cover_images = self.metadata.pop('ssmd_cover_images', None)  # those are cover images and they are too big to display in UI as text


class LoraModule:
    def __init__(self, name):
        self.name = name
        self.multiplier = 1.0
        self.modules = {}
        self.mtime = None


class LoraUpDownModule:
    def __init__(self):
        self.up = None
        self.down = None
        self.alpha = None


def assign_lora_names_to_compvis_modules(sd_model):
    lora_layer_mapping = {}

    for name, module in sd_model.cond_stage_model.wrapped.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    for name, module in sd_model.model.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    sd_model.lora_layer_mapping = lora_layer_mapping


def load_lora(name, filename):
    lora = LoraModule(name)
    lora.mtime = os.path.getmtime(filename)

    sd = read_state_dict(filename)

    keys_failed_to_match = {}
    is_sd2 = 'model_transformer_resblocks' in sd_model.lora_layer_mapping

    for key_diffusers, weight in sd.items():
        key_diffusers_without_lora_parts, lora_key = key_diffusers.split(".", 1)
        key = convert_diffusers_name_to_compvis(key_diffusers_without_lora_parts, is_sd2)

        sd_module = sd_model.lora_layer_mapping.get(key, None)

        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = sd_model.lora_layer_mapping.get(m.group(1), None)

        if sd_module is None:
            keys_failed_to_match[key_diffusers] = key
            continue

        lora_module = lora.modules.get(key, None)
        if lora_module is None:
            lora_module = LoraUpDownModule()
            lora.modules[key] = lora_module

        if lora_key == "alpha":
            lora_module.alpha = weight.item()
            continue

        if type(sd_module) == torch.nn.Linear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.MultiheadAttention:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.Conv2d:
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            print(f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}')
            continue
            assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'

        with torch.no_grad():
            module.weight.copy_(weight)

        module.to(device=devices.cpu, dtype=devices.dtype)

        if lora_key == "lora_up.weight":
            lora_module.up = module
        elif lora_key == "lora_down.weight":
            lora_module.down = module
        else:
            assert False, f'Bad Lora layer name: {key_diffusers} - must end in lora_up.weight, lora_down.weight or alpha'

    if len(keys_failed_to_match) > 0:
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match}")

    return lora


def load_loras(names, multipliers=None):
    already_loaded = {}

    for lora in loaded_loras:
        if lora.name in names:
            already_loaded[lora.name] = lora

    loaded_loras.clear()

    loras_on_disk = [available_loras.get(name, None) for name in names]
    if any([x is None for x in loras_on_disk]):
        list_available_loras()

        loras_on_disk = [available_loras.get(name, None) for name in names]

    for i, name in enumerate(names):
        lora = already_loaded.get(name, None)

        lora_on_disk = loras_on_disk[i]
        if lora_on_disk is not None:
            if lora is None or os.path.getmtime(lora_on_disk.filename) > lora.mtime:
                lora = load_lora(name, lora_on_disk.filename)

        if lora is None:
            print(f"Couldn't find Lora with name {name}")
            continue

        lora.multiplier = multipliers[i] if multipliers else 1.0
        loaded_loras.append(lora)


def lora_calc_updown(lora, module, target):
    with torch.no_grad():
        up = module.up.weight.to(target.device, dtype=target.dtype)
        down = module.down.weight.to(target.device, dtype=target.dtype)

        if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
            updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            updown = up @ down

        updown = updown * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)

        return updown


def lora_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of Loras to the weights of torch layer self.
    If weights already have this particular set of loras applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to loras.
    """

    lora_layer_name = getattr(self, 'lora_layer_name', None)
    if lora_layer_name is None:
        return

    current_names = getattr(self, "lora_current_names", ())
    wanted_names = tuple((x.name, x.multiplier) for x in loaded_loras)

    weights_backup = getattr(self, "lora_weights_backup", None)
    if weights_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(devices.cpu, copy=True), self.out_proj.weight.to(devices.cpu, copy=True))
        else:
            weights_backup = self.weight.to(devices.cpu, copy=True)

        self.lora_weights_backup = weights_backup

    if current_names != wanted_names:
        if weights_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.in_proj_weight.copy_(weights_backup[0])
                self.out_proj.weight.copy_(weights_backup[1])
            else:
                self.weight.copy_(weights_backup)

        for lora in loaded_loras:
            module = lora.modules.get(lora_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                self.weight += lora_calc_updown(lora, module, self.weight)
                continue

            module_q = lora.modules.get(lora_layer_name + "_q_proj", None)
            module_k = lora.modules.get(lora_layer_name + "_k_proj", None)
            module_v = lora.modules.get(lora_layer_name + "_v_proj", None)
            module_out = lora.modules.get(lora_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                updown_q = lora_calc_updown(lora, module_q, self.in_proj_weight)
                updown_k = lora_calc_updown(lora, module_k, self.in_proj_weight)
                updown_v = lora_calc_updown(lora, module_v, self.in_proj_weight)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                self.in_proj_weight += updown_qkv
                self.out_proj.weight += lora_calc_updown(lora, module_out, self.out_proj.weight)
                continue

            if module is None:
                continue

            print(f'failed to calculate lora weights for layer {lora_layer_name}')

        setattr(self, "lora_current_names", wanted_names)


def lora_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    setattr(self, "lora_current_names", ())
    setattr(self, "lora_weights_backup", None)


def lora_Linear_forward(self, input):
    lora_apply_weights(self)

    return torch.nn.Linear_forward_before_lora(self, input)


def lora_Linear_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.Linear_load_state_dict_before_lora(self, *args, **kwargs)


def lora_Conv2d_forward(self, input):
    lora_apply_weights(self)

    return torch.nn.Conv2d_forward_before_lora(self, input)


def lora_Conv2d_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.Conv2d_load_state_dict_before_lora(self, *args, **kwargs)


def lora_MultiheadAttention_forward(self, *args, **kwargs):
    lora_apply_weights(self)

    return torch.nn.MultiheadAttention_forward_before_lora(self, *args, **kwargs)


def lora_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.MultiheadAttention_load_state_dict_before_lora(self, *args, **kwargs)


def list_available_loras():
    available_loras.clear()

    os.makedirs(lora_dir, exist_ok=True)

    candidates = \
        glob(os.path.join(lora_dir, '**/*.pt'), recursive=True) + \
        glob(os.path.join(lora_dir, '**/*.safetensors'), recursive=True) + \
        glob(os.path.join(lora_dir, '**/*.ckpt'), recursive=True)

    for filename in sorted(candidates, key=str.lower):
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]

        available_loras[name] = LoraOnDisk(name, filename)




def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
    torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_lora
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora
    torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_lora
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_lora
    torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_lora

if not hasattr(torch.nn, 'Linear_forward_before_lora'):
    torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Linear_load_state_dict_before_lora'):
    torch.nn.Linear_load_state_dict_before_lora = torch.nn.Linear._load_from_state_dict

if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
    torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_lora'):
    torch.nn.Conv2d_load_state_dict_before_lora = torch.nn.Conv2d._load_from_state_dict

if not hasattr(torch.nn, 'MultiheadAttention_forward_before_lora'):
    torch.nn.MultiheadAttention_forward_before_lora = torch.nn.MultiheadAttention.forward

if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_lora'):
    torch.nn.MultiheadAttention_load_state_dict_before_lora = torch.nn.MultiheadAttention._load_from_state_dict

checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k

def inject_lora(sd_model):
  torch.nn.Linear.forward = lora_Linear_forward
  torch.nn.Linear._load_from_state_dict = lora_Linear_load_state_dict
  torch.nn.Conv2d.forward = lora_Conv2d_forward
  torch.nn.Conv2d._load_from_state_dict = lora_Conv2d_load_state_dict
  torch.nn.MultiheadAttention.forward = lora_MultiheadAttention_forward
  torch.nn.MultiheadAttention._load_from_state_dict = lora_MultiheadAttention_load_state_dict

  assign_lora_names_to_compvis_modules(sd_model)

available_loras = {}
loaded_loras = []
import safetensors

def split_lora_from_prompts(prompts):
  re1 = '\<(.*?)\>'
  new_prompt_loras = {}
  new_prompts = {}

  #iterate through prompts keyframes and fill in lora schedules
  for key in prompts.keys():
    subp = prompts[key][0]

    #get a dict of loras:weights from a prompt
    prompt_loras = re.findall(re1, subp)
    prompt_loras_dict = dict([(o.split(':')[1], o.split(':')[-1]) for o in prompt_loras])

    #fill lora dict based on keyframe, lora:weight
    for lora_key in prompt_loras_dict.keys():
      try: new_prompt_loras[lora_key]
      except: new_prompt_loras[lora_key] = {}
      new_prompt_loras[lora_key][key] = float(prompt_loras_dict[lora_key])

    # remove lora keywords from promtps
    new_prompts[key] = [re.sub(re1, '', subp).strip(' ')]

  return new_prompts, new_prompt_loras

def get_loras_weights_for_frame(frame_num, loras_dict):
  loras = list(loras_dict.keys())
  loras_weights = [get_scheduled_arg(frame_num, loras_dict[o]) for o in loras]
  return loras, loras_weights

list_available_loras()
print('Loras detected:\n','\n'.join(list(available_loras.keys())))