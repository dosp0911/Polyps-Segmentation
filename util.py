import torch
import os
from pathlib import Path


def load_model(path, model, map_location=None):
  '''
    args:
      path : location to load model
      model : model variable
      map_location : device to load model
    return:
      model loaded weights from saved model
  '''
  load_model = torch.load(path, map_location=map_location)
  model.load_state_dict(load_model['model_state_dict'])
  return model


def save_model(model, optim, save_path, model_name, epoch, loss, save_num=30):
  if not Path(save_path).exists():
      Path(save_path).mkdir()

  models_list = sorted(list(Path(save_path).glob('*.pth')))

  if len(models_list) > save_num:
    Path(models_list[0]).unlink()

  model_f_path = os.path.join(save_path, model_name)
  save_config = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
  if optim is not None:
      save_config['optim_state_dict'] = optim.state_dict()
  torch.save(save_config, model_f_path)
  print(f'model saved \n {model_f_path}')

