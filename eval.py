import torch
from Dataset import KvasirSegDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data.dataloader import DataLoader
from tqdm.autonotebook import tqdm
from util import load_model, get_logger
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def evaluate(model, file_path, device, loss_func, model_path=None, f_name='pred_masks', threshold=0.5):

	if model_path is not None:
		load_model(model_path, model, map_location=device)

	e_dir = Path(file_path) / 'eval'
	if not e_dir.exists():
		e_dir.mkdir()
		print(f'{e_dir} created.')

	logger = get_logger('eval_logger', 'info', log_file_path=e_dir)
	pred_mask_path = e_dir / f_name
	if not pred_mask_path.exists():
		pred_mask_path.mkdir()

	transforms = A.Compose([
		A.Resize(256, 256),
		A.Normalize(),
		ToTensorV2()
	])

	ds = KvasirSegDataset(file_path, transforms)
	d_loader = DataLoader(ds, batch_size=1, pin_memory=True)

	model.to(device)
	model.eval()
	losses = {}
	with torch.no_grad():
		for i, data in tqdm(enumerate(d_loader), desc="Eval", total=len(d_loader)):
			imgs, masks, paths = data['image'], data['mask'], data['path']
			imgs = imgs.to(device).float()
			masks = masks.to(device).float()
			pred_masks = model(imgs)
			loss = loss_func(masks, pred_masks)
			losses[paths[0]] = loss.item()
			pred_masks = (pred_masks.cpu().detach().numpy() > threshold).astype(np.uint8)
			for mask, p in zip(pred_masks, paths):
				f_name = p.split('\\')[-1].split('.')[0] + '.jpg'
				plt.imsave(str(pred_mask_path / f_name), mask.squeeze(), cmap='gray')

	logger.info(sorted(losses.items(), key=lambda x: x[1]))

