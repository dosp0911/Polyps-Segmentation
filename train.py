import torch
import util
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm.autonotebook import tqdm
from collections import defaultdict
import time
from pathlib import Path


def train(train_dataloader, val_dataloader, model: torch.nn.Module, epochs: int, loss_func, metrics: dict,
          optimizer=None, scheduler=None, device=None,  print_iter=100, logs_path='logs', model_path='models'):

	if optimizer is None:
		optimizer = torch.optim.SGD(model.parameters(), momentum=0.99, lr=0.001, weight_decay=0.0005)

	if device is None:
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	model.train()
	model.to(device)
	for e in tqdm(range(epochs)):
		start = time.time()
		metrics_ = defaultdict(int)
		writer = SummaryWriter(logs_path)
		loss_sum = 0
		for i, (img, mask) in tqdm(enumerate(train_dataloader), desc='Train', total=len(train_dataloader)):
			optimizer.zero_grad()

			img = img.to(device).float()
			# mask = mask.to(device).long().unsqueeze(dim=1)
			mask = mask.to(device).long() # colab에서는 이렇게..
			outputs = model(img)
			loss = loss_func(outputs, mask)
			with torch.no_grad():
				loss_sum += loss
			loss.backward()

			optimizer.step()

			for m_name, m_func in metrics.items():
				metrics_[m_name] += m_func(outputs, mask)

			if i % print_iter == 0:
				metrics_str = ''
				for k, v in metrics_.items():
					metrics_str += f'\t{k}:{v/(i+1)}'
				print(f'Epoch:{e} Train Loss: {loss_sum/(i+1)} {metrics_str}')
				writer.add_scalar('Train_loss', loss_sum/(i+1), global_step=e * train_dataloader.batch_size + i)
				writer.add_scalars('Train', metrics_, global_step=e * train_dataloader.batch_size + i)
				for name, param in model.named_parameters():
					writer.add_histogram(name, param.clone().cpu().detach().numpy(), global_step=e * train_dataloader.batch_size + i)

		if scheduler is not None:
			scheduler.step()

		model.eval()
		with torch.no_grad():
			loss_sum_val = 0
			metrics_val = defaultdict(int)
			for j, (img_val, mask_val) in tqdm(enumerate(val_dataloader), desc='Val', total=len(val_dataloader)):
				img_val = img_val.to(device)
				mask_val = mask_val.to(device)
				outputs_val = model(img_val)
				loss_val = loss_func(outputs_val, mask_val)
				loss_sum_val += loss_val


				for m_name, m_func in metrics.items():
					metrics_val[m_name] += m_func(outputs_val, mask_val)

				if j % print_iter == 0:
					metrics_str = ''
					for k, v in metrics_val.items():
						metrics_str += f'\t{k}:{v / (j + 1)}'
					print(f'Epoch:{e} Val Loss: {loss_sum_val / (j + 1)} {metrics_str}')
					writer.add_scalar('Val_loss', loss_sum_val / (j + 1),
					                  global_step=e * val_dataloader.batch_size + j)
					writer.add_scalars('Val', metrics_val, global_step=e * val_dataloader.batch_size + j)

				if j == len(val_dataloader):
					pred_masks = (outputs_val > 0.5).float().squeeze(dim=1)
					writer.add_image('Val_ori_img', img_val[0].detach().numpy(), dataformats='CHW', global_step=(e+1) * val_dataloader.batch_size)
					writer.add_image('Val_ori_mask', mask_val[0].detach().numpy(), dataformats='HW', global_step=(e+1) * val_dataloader.batch_size)
					writer.add_image('Val_pred_mask', pred_masks[0].detach().numpy(), dataformats='HW', global_step=(e+1) * val_dataloader.batch_size)

		if not Path(model_path).exists():
			Path(model_path).mkdir()

		writer.close()
		util.save_model(model, optimizer, model_path, f'Unet_{e}_{loss_sum}', epoch=e, loss=loss_sum)
		end = time.time()
		print('{}th epoch is over. Elasped Time:{} min.'.format(e, (start-end)//60))