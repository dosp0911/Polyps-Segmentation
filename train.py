import torch
import util
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import defaultdict
import time
from pathlib import Path

import numpy as np
import mlflow


def mlflow_init(batch_size, epochs, device, optim, experiment_name):
	optim_params = optim.state_dict()['param_groups'][0].copy()
	optim_params.pop('params')
	mlflow.log_param('optimizer', optim.__class__)
	mlflow.log_params(optim_params)

	mlflow.log_param('batch_size', batch_size)
	mlflow.log_param('epochs', epochs)
	mlflow.log_param('device', device)
	mlflow.set_experiment(experiment_name)


def train(train_dataloader, val_dataloader, model: torch.nn.Module, epochs: int, loss_func, metrics: dict,
        optimizer=None, scheduler=None, device=None, start_epochs: int = 0, print_iter=100, logs_path='logs',
        model_path='models', experiment_name='exp', run_name='run'):

	best_scores = {k: 0 for k, v in metrics.items()}
	is_best = {k: False for k, v in metrics.items()}

	c_time = util.get_current_time_KST()

	if not Path(logs_path).exists():
		Path(logs_path).mkdir()
	logger = util.get_logger(name=f'train_{c_time}.log', level='info', log_file_path=logs_path, is_stream=False)

	if optimizer is None:
		optimizer = torch.optim.SGD(model.parameters(), momentum=0.99, lr=0.001, weight_decay=0.0005)
	print(f'optimizer : {optimizer.__class__}')
	logger.info('optimizer:{} param_groups:{}'.format(optimizer.__class__, optimizer.state_dict()['param_groups']))

	if device is None:
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	logger.info(f'device:{device}')

	logger.info('------------------------------- Train start -----------------------------------------')
	writer = SummaryWriter(logs_path)
	with mlflow.start_run(run_name=run_name):
		mlflow_init(train_dataloader.batch_size, epochs, device, optimizer, experiment_name)
		model.train()
		for e in tqdm(range(start_epochs, epochs), desc='Total'):
			start = time.time()
			metrics_ = defaultdict(int)
			loss_sum = 0
			for i, data in tqdm(enumerate(train_dataloader), desc='Train', total=len(train_dataloader)):
				optimizer.zero_grad()
				imgs, masks, paths = data['image'], data['mask'], data['path']
				imgs = imgs.to(device).float()
				masks = masks.to(device).long()
				outputs = model(imgs)
				loss = loss_func(outputs, masks)
				with torch.no_grad():
					loss_sum += loss.item()
				loss.backward()
				optimizer.step()

				for m_name, m_func in metrics.items():
					metrics_[m_name] += m_func(outputs, masks).item()

				metrics_str = ''
				mMetrics = {}
				for k, v in metrics_.items():
					metrics_str += f'\t{k}_train:{v / (i + 1)}'
					mMetrics[f'{k}_train'] = v / (i + 1)

				g_step = (e * len(train_dataloader) * train_dataloader.batch_size) + (train_dataloader.batch_size * i)
				logger.info(f'Epoch:{e} Step:{g_step} Train Loss: {loss_sum / (i + 1)} Metrics: {metrics_str}')
				if i % print_iter == 0:
					print(f'Epoch:{e} Train Loss: {loss_sum / (i + 1)} {metrics_str}')

					writer.add_scalar('Train_loss', loss_sum / (i + 1), global_step=g_step)
					writer.add_scalars('Train', mMetrics, global_step=g_step)
					for name, param in model.named_parameters():
						writer.add_histogram(name, param.clone().cpu().detach().numpy(), global_step=g_step)
					mlflow.log_metric('Train Loss', loss_sum / (i + 1), step=g_step)
					mlflow.log_metrics(mMetrics, step=g_step)

			if scheduler is not None:
				scheduler.step()

			logger.info('--------------------------- Validate Start ------------------------------------')
			# Evaluate

			model.eval()
			with torch.no_grad():
				loss_sum_val = 0
				metrics_val = defaultdict(int)
				for j, data_val in tqdm(enumerate(val_dataloader), desc='Val', total=len(val_dataloader)):
					imgs_val, masks_val, paths_val = data_val['image'], data_val['mask'], data_val['path']
					imgs_val = imgs_val.to(device)
					masks_val = masks_val.to(device).long()
					outputs_val = model(imgs_val)
					loss_val = loss_func(outputs_val, masks_val)
					loss_sum_val += loss_val.item()

					for m_name, m_func in metrics.items():
						metrics_val[m_name] += m_func(outputs_val, masks_val).item()

					metrics_str = ''
					mMetrics_val = {}
					for k, v in metrics_val.items():
						metrics_str += f'\t{k}_val:{v / (j + 1)}'
						mMetrics_val[f'{k}_val'] = v / (j + 1)

					g_step_val = (e * len(val_dataloader) * val_dataloader.batch_size) + (j * val_dataloader.batch_size)
					logger.info(
						f'Epoch:{e} Step:{g_step_val} Val Loss: {loss_sum_val / (j + 1)} Metrics: {metrics_str}')
					if j % print_iter == 0:
						print(f'Epoch:{e} Val Loss: {loss_sum_val / (j + 1)} {metrics_str}')
						writer.add_scalar('Val_loss', loss_sum_val / (j + 1), global_step=g_step_val)
						writer.add_scalars('Val', mMetrics_val, global_step=g_step_val)
						mlflow.log_metric('Val Loss', loss_sum_val / (j + 1), step=g_step_val)
						mlflow.log_metrics(mMetrics_val, step=g_step_val)

					if j == (len(val_dataloader) - 1):
						r_idx = np.random.randint(0, val_dataloader.batch_size)
						pred_masks = (outputs_val > 0.5).float().squeeze(dim=1)
						writer.add_image('Val_ori_img', imgs_val[r_idx], dataformats='CHW', global_step=g_step_val)
						writer.add_image('Val_ori_mask', masks_val[r_idx].squeeze().cpu().detach().numpy(),
						                 dataformats='HW', global_step=g_step_val)
						writer.add_image('Val_pred_mask', pred_masks[r_idx].cpu().detach().numpy(), dataformats='HW',
						                 global_step=g_step_val)


			for k, v in metrics_val.items():
				avg_ = v / len(val_dataloader)
				if best_scores[k] < avg_:
					best_scores[k] = avg_
					is_best[k] = True
			if any(is_best.values()):
				util.save_model(model, optimizer, model_path, f'{model.__class__.__name__}_E{e}_{int(loss_sum_val)}.pth',
			                epoch=e, loss=loss_sum_val, metrics=metrics_val, save_num=5)
				for k in is_best.keys():
					is_best[k] = False
			end = time.time()
			print('{}th epoch is over. Elasped Time:{} min. loss average:{} Best scores:{} Current scores:{}'.format(e, (end - start) // 60,
			      loss_sum_val/len(val_dataloader), best_scores, avg_))
			logger.info('{}th epoch is over. Elasped Time:{} min. loss average:{} Best scores:{} Current scores:{}'.format(e, (end - start) // 60,
			      loss_sum_val/len(val_dataloader), best_scores, avg_))

		if scheduler is not None:
			mlflow.log_param('lr_scheduler', scheduler.__class__)
			mlflow.log_params(scheduler.state_dict())
			print(f'lr_scheduler:{scheduler.__class__} {scheduler.state_dict()}')
			logger.info(f'lr_scheduler:{scheduler.__class__} {scheduler.state_dict()}')

		writer.close()
		mlflow.log_artifacts(logs_path, artifact_path="events")

		logger.info(f"Uploading TensorBoard events as a run artifact...")
		print(f"Uploading TensorBoard events as a run artifact...")
