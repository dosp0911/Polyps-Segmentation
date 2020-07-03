from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import torchvision

import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from pathlib import Path


class KvasirSegDataset(Dataset):
	def __init__(self, root: str, transform=None):
		super(KvasirSegDataset, self).__init__()
		self.img_list = sorted([str(p) for p in (Path(root) / 'images').glob('*.jpg')])
		self.mask_list = sorted([str(p) for p in (Path(root) / 'masks').glob('*.jpg')])
		self.transform = transform

	def __getitem__(self, item):
		img = cv2.imread(self.img_list[item])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.img_list[item])
		mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		mask = (mask > 100).astype(np.uint8)
		if isinstance(self.transform, A.Compose):
			aum = self.transform(image=img, mask=mask)
			img, mask = aum['image'], aum['mask']

		return img, mask

	def __len__(self):
		return len(self.img_list)


