import torch
import os

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


def save_model(model, optim, save_path, model_name, epoch, loss):
  model_f_path = os.path.join(save_path, model_name)
  torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'optim_state_dict': optim.state_dict()
    }, model_f_path)
  print(f'model saved \n {model_f_path}')

