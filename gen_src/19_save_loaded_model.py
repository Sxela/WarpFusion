#@title Save loaded model 
#@markdown For this cell to work you need to load model in the previous cell.\
#@markdown Saves an already loaded model as an object file, that weights less, loads faster, and requires less CPU RAM.\
#@markdown After saving model as pickle, you can then load it as your usual stable diffusion model in thecell above.\
#@markdown The model will be saved under the same name with .pkl extenstion.
save_model_pickle = False #@param {'type':'boolean'}
save_folder = "/content/drive/MyDrive/models" #@param {'type':'string'}
if save_folder != '' and save_model_pickle:
  os.makedirs(save_folder, exist_ok=True)
  out_path = save_folder+model_path.replace('\\', '/').split('/')[-1].split('.')[0]+'.pkl'
  with open(out_path, 'wb') as f:
    pickle.dump(sd_model, f)
  print('Model successfully saved as: ',out_path)