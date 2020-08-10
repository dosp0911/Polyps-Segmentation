import torch
import os
from pathlib import Path
import logging
import logging.handlers
from pytz import timezone

from datetime import datetime
from dateutil import tz


def get_logger(name='my_logger', level='warning', format='%(asctime)-15s %(message)s', log_file_path=None, is_stream=True):
    lvs = {'debug': logging.DEBUG,
           'info': logging.INFO,
           'warning': logging.WARNING,
           'error': logging.ERROR,
           'critical': logging.CRITICAL}

    lv = lvs[level]

    def timetz(*args):
        return datetime.now(tz_).timetuple()

    tz_ = timezone('Asia/Seoul')  # UTC, Asia/Seoul
    logging.Formatter.converter = timetz

    logger = logging.getLogger(name)
    logger.setLevel(lv)
    formatter = logging.Formatter(format)

    if log_file_path is not None:
        file_handler = logging.handlers.RotatingFileHandler(os.path.join(log_file_path, name), maxBytes=10 * 1024 * 1024)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if is_stream:
        stream_hander = logging.StreamHandler()
        logger.addHandler(stream_hander)

    return logger

def get_current_time_KST():
    to_zone = tz.gettz('Asia/Seoul')
    utc = datetime.now()
    current_time = utc.astimezone(to_zone).strftime('%Y%m%d%H%M')
    return current_time


def load_model(path, model, optim=None, map_location=None):
  '''
    args:
      path : location to load model
      model : model variable
      optim: optimizer
      map_location : device to load model
    return:
      model loaded weights from saved model
  '''
  load_model = torch.load(path, map_location=map_location)
  model.load_state_dict(load_model['model_state_dict'])
  model.to(map_location)
  if optim is not None:
      optim.load_state_dict(load_model['optim_state_dict'])
      print('optimizer is loaded!')
  print('model is loaded!')


def save_model(model, optim, save_path, model_name, epoch, loss, metrics, save_num=5):
  if not Path(save_path).exists():
      Path(save_path).mkdir()

  models_list = sorted(list(Path(save_path).glob('*.pth')), key=os.path.getctime) # 생성된 날짜 순 삭제

  if len(models_list) > save_num:
    os.remove(str(models_list[0]))

  model_f_path = os.path.join(save_path, model_name)
  save_config = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'scores': metrics
    }
  if optim is not None:
      save_config['optim_state_dict'] = optim.state_dict()
  torch.save(save_config, model_f_path)
  print(f'model saved \n {model_f_path}')



