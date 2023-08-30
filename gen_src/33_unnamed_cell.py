warp_mode = 'use_image' #@param ['use_latent', 'use_image']
warp_towards_init = 'off' #@param ['stylized', 'off']

if warp_towards_init != 'off':
  if flow_lq:
          raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
        # raft_model = torch.nn.DataParallel(RAFT(args2))
  else: raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()

depth_source = 'init' #@param ['init', 'stylized']