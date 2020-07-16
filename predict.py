import torch
from Dataset import TestKvasirSegDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data.dataloader import DataLoader
from tqdm.autonotebook import tqdm
from util import load_model
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def predict(model, file_path, device, model_path=None, f_name='pred_masks',  threshold=0.5):

	if model_path is not None:
		model = load_model(model_path, model, device)
	pred_mask_path = Path(file_path) / f_name
	if not pred_mask_path.exists():
		pred_mask_path.mkdir()
	transforms = A.Compose([
		A.Resize(256, 256),
		A.Normalize(),
		ToTensorV2()
	])

	ds = TestKvasirSegDataset(file_path, transforms)
	d_loader = DataLoader(ds, batch_size=4)

	model.to(device)
	model.eval()

	with torch.no_grad():
		for i, (imgs, paths) in tqdm(enumerate(d_loader), desc="Predict", total=len(d_loader)):
			imgs = imgs.to(device).float()
			pred_masks = model(imgs)
			pred_masks = (pred_masks.cpu().detach().numpy() > threshold).astype(np.uint8)
			for mask, p in zip(pred_masks, paths):
				f_name = p.split('\\')[-1].split('.')[0] + '.png'
				plt.imsave(str(pred_mask_path / f_name), mask.squeeze(), cmap='gray')
