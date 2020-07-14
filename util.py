import torch
import os
from pathlib import Path
import logging
import logging.handlers


def get_logger(name='my_logger', level='warning', format='%(asctime)-15s %(message)s', log_file_path=None):
    lvs = {'debug': logging.DEBUG,
           'info': logging.INFO,
           'warning': logging.WARNING,
           'error': logging.ERROR,
           'critical': logging.CRITICAL}

    lv = lvs[level]

    logger = logging.getLogger(name)
    logger.setLevel(lv)
    formatter = logging.Formatter(format)

    if log_file_path is not None:
        file_handler = logging.handlers.RotatingFileHandler(log_file_path, maxBytes=10 * 1024 * 1024)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


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

  models_list = sorted(list(Path(save_path).glob('*.pth')), key=os.path.getctime) # 생성된 날짜 순 삭제

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


if __name__ == '__main__':
    logger = get_logger(log_file_path='test.log')
    h_ = logger.handlers
    from tqdm.auto import tqdm
    for i in tqdm(range(10)):
        logger.debug('debug')
        logger.info('info!')
        logger.warning('warning!')
        logger.error('error!')

    print('')