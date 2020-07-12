import unittest
from Dataset import KvasirSegDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class MyTestCase(unittest.TestCase):
	def test_something(self):
		root_path = 'Kvasir-SEG'
		batch_size = 8
		transforms = A.Compose([
			A.RandomBrightnessContrast(p=0.5),
			A.HorizontalFlip(p=0.5),
			A.ShiftScaleRotate(p=1),
			A.Resize(256, 256),
			A.Normalize(),
			ToTensorV2()
		])

		dataset = KvasirSegDataset(root_path, transforms)
		train_size = int(len(dataset) * 0.8)
		val_size = len(dataset) - train_size
		train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
		train_dataloader = DataLoader(train_dataset, batch_size)
		val_dataloader = DataLoader(val_dataset, batch_size)

		img, target = dataset[13]
		# mask = (target > 100).astype(np.uint8)
		fig, axes = plt.subplots(1, 2)
		axes[0].imshow(img.permute(1,2,0))
		axes[1].imshow(target.squeeze())
		# axes[2].imshow(mask.squeeze())
		print(np.unique(target))
		plt.show()


if __name__ == '__main__':
	unittest.main()
