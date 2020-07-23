import torch
import util
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import defaultdict
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import mlflow


def mlflow_init(batch_size, epochs, device, optim):
	optim_params = optim.state_dict()['param_groups'][0].copy()
	optim_params.pop('params')
	mlflow.log_params(optim_params)

	mlflow.log_param('batch_size', batch_size)
	mlflow.log_param('epochs', epochs)
	mlflow.log_param('device', device)


def train(train_dataloader, val_dataloader, model: torch.nn.Module, epochs: int, loss_func, metrics: dict,
          optimizer=None, scheduler=None, device=None, start_epochs: int = 0, print_iter=100, logs_path='logs',
          model_path='models'):

	c_time = datetime.today().strftime('%Y%m%d%H%M')
	if not Path(logs_path).exists():
		Path(logs_path).mkdir()
	logger = util.get_logger(name=f'train_{c_time}.log', level='info', log_file_path=f'{logs_path}/train_{c_time}.log')

	if optimizer is None:
		optimizer = torch.optim.SGD(model.parameters(), momentum=0.99, lr=0.001, weight_decay=0.0005)
	logger.info(f'optimizer{optimizer.state_dict()}')

	if device is None:
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	logger.info(f'device:{device}')

	logger.info('------------------------------- Train start -----------------------------------------')
	with mlflow.start_run():
		mlflow_init(train_dataloader.batch_size, epochs, device, optimizer)
		# mlflow.log_param
		model.train()
		model.to(device)
		for e in tqdm(range(start_epochs, epochs), desc='Total'):
			start = time.time()
			metrics_ = defaultdict(int)
			writer = SummaryWriter(logs_path)
			loss_sum = 0
			for i, (img, mask) in tqdm(enumerate(train_dataloader), desc='Train', total=len(train_dataloader)):
				optimizer.zero_grad()
				img = img.to(device).float()
				mask = mask.to(device).long()
				outputs = model(img)
				loss = loss_func(outputs, mask)
				with torch.no_grad():
					loss_sum += loss
				loss.backward()

				optimizer.step()

				for m_name, m_func in metrics.items():
					metrics_[m_name] += m_func(outputs, mask)

				metrics_str = ''
				mMetrics = {}
				for k, v in metrics_.items():
					metrics_str += f'\t{k}:{v / (i + 1)}'
					mMetrics[k] = v.item() / (i + 1)

				g_step = (e * len(train_dataloader) * train_dataloader.batch_size) + (train_dataloader.batch_size * i)
				logger.info(f'Epoch:{e} Step:{g_step} Train Loss: {loss_sum / (i + 1)} Metrics: {metrics_str}')
				if i % print_iter == 0:
					print(f'Epoch:{e} Train Loss: {loss_sum / (i + 1)} {metrics_str}')

					writer.add_scalar('Train_loss', loss_sum / (i + 1), global_step=g_step)
					writer.add_scalars('Train', mMetrics, global_step=g_step)
					for name, param in model.named_parameters():
						writer.add_histogram(name, param.clone().cpu().detach().numpy(), global_step=g_step)
					mlflow.log_metric('Train Loss', loss_sum.item() / (i + 1))
					mlflow.log_metrics(mMetrics, step=g_step)

			if scheduler is not None:
				scheduler.step()

			logger.info('--------------------------- Validate Start ------------------------------------')
			# Evaluate
			model.eval()
			with torch.no_grad():
				loss_sum_val = 0
				metrics_val = defaultdict(int)
				for j, (img_val, mask_val) in tqdm(enumerate(val_dataloader), desc='Val', total=len(val_dataloader)):
					img_val = img_val.to(device)
					mask_val = mask_val.to(device).long()
					outputs_val = model(img_val)
					loss_val = loss_func(outputs_val, mask_val)
					loss_sum_val += loss_val

					for m_name, m_func in metrics.items():
						metrics_val[m_name] += m_func(outputs_val, mask_val)

					metrics_str = ''
					mMetrics_val = {}
					for k, v in metrics_val.items():
						metrics_str += f'\t{k}_val:{v / (j + 1)}'
						mMetrics_val[f'{k}_val'] = v.item() / (j + 1)

					g_step_val = (e * len(val_dataloader) * val_dataloader.batch_size) + (j * val_dataloader.batch_size)
					logger.info(f'Epoch:{e} Step:{g_step_val} Val Loss: {loss_sum_val / (j + 1)} Metrics: {metrics_str}')
					if j % print_iter == 0:
						print(f'Epoch:{e} Val Loss: {loss_sum_val / (j + 1)} {metrics_str}')
						writer.add_scalar('Val_loss', loss_sum_val / (j + 1),
						                       global_step=g_step_val)
						writer.add_scalars('Val', mMetrics_val, global_step=g_step_val)
						mlflow.log_metric('Val Loss', loss_sum_val.item() / (j + 1))
						mlflow.log_metrics(mMetrics_val, step=g_step_val)

					if j == (len(val_dataloader)-1):
						r_idx = np.random.randint(0, val_dataloader.batch_size)
						pred_masks = (outputs_val > 0.5).float().squeeze(dim=1)
						writer.add_image('Val_ori_img', img_val[r_idx], dataformats='CHW', global_step=g_step_val)
						writer.add_image('Val_ori_mask', mask_val[r_idx].squeeze().cpu().detach().numpy(), dataformats='HW',
						                          global_step=g_step_val)
						writer.add_image('Val_pred_mask', pred_masks[r_idx].cpu().detach().numpy(), dataformats='HW',
						                       global_step=g_step_val)

			writer.close()
			util.save_model(model, optimizer, model_path, f'{model.__class__.__name__}{e}_{int(loss_sum)}.pth',
			                epoch=e, loss=loss_sum)
			end = time.time()
			print('{}th epoch is over. Elasped Time:{} min.'.format(e, (end - start) // 60))
			logger.info('{}th epoch is over. Elasped Time:{} min.'.format(e, (end - start) // 60))

		if scheduler is not None:
			mlflow.log_params(scheduler.state_dict())
			print(f'scheduler:{scheduler.state_dict()}')
			logger.info(f'scheduler:{scheduler.state_dict()}')

		mlflow.log_artifacts(logs_path)
		logger.info(f"Uploading TensorBoard events as a run artifact...")
		print(f"Uploading TensorBoard events as a run artifact...")