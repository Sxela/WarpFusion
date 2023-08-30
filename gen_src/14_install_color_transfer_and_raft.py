#@title Install Color Transfer and RAFT
##@markdown Run once per session. Doesn't download again if model path exists.
##@markdown Use force download to reload raft models if needed
force_download = False #@param {type:'boolean'}
# import wget 
import zipfile, shutil

if (os.path.exists(f'{root_dir}/raft')) and force_download:
  try:
    shutil.rmtree(f'{root_dir}/raft')
  except:
      print('error deleting existing RAFT model')
if (not (os.path.exists(f'{root_dir}/raft'))) or force_download:
  os.chdir(root_dir)
  gitclone('https://github.com/Sxela/WarpFusion')

try: 
  from python_color_transfer.color_transfer import ColorTransfer, Regrain
except: 
  os.chdir(root_dir)
  gitclone('https://github.com/pengbo-learn/python-color-transfer')

os.chdir(root_dir)
sys.path.append('./python-color-transfer')

if animation_mode == 'Video Input':
  os.chdir(root_dir)
  gitclone('https://github.com/Sxela/flow_tools')
    
  # %cd "{root_dir}/"
  # !git clone https://github.com/princeton-vl/RAFT
  # %cd "{root_dir}/RAFT"
  # if os.path.exists(f'{root_path}/RAFT/models') and force_download:
  #   try:
  #     print('forcing model redownload')
  #     shutil.rmtree(f'{root_path}/RAFT/models')
  #   except:
  #     print('error deleting existing RAFT model')

  # if (not (os.path.exists(f'{root_path}/RAFT/models/raft-things.pth'))) or force_download:

  #   !curl -L https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip -o "{root_dir}/RAFT/models.zip"

  #   with zipfile.ZipFile(f'{root_dir}/RAFT/models.zip', 'r') as zip_ref:
  #       zip_ref.extractall(f'{root_path}/RAFT/')