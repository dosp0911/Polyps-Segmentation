from Dataset import KvasirSegDataset
from Metrics import iou, DICELoss
from model import Unet
from train import train

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

if __name__ == '__main__':

	root_path = 'Kvasir-SEG'
	batch_size = 2
	epochs = 30
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device['cpu']

	transforms = A.Compose([
		A.Resize(256, 256),
		ToTensorV2()
	])

	dataset = KvasirSegDataset(root_path, transforms)
	train_size = int(len(dataset)*0.8)
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	train_dataloader = DataLoader(train_dataset, batch_size)
	val_dataloader = DataLoader(val_dataset, batch_size)

	in_classes = 3
	out_classes = 1
	model = Unet(in_classes, out_classes)

	criteria = DICELoss()
	metrics = {'iou': iou}

	optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99, weight_decay=0.0005)

	train(train_dataloader, val_dataloader, model, epochs, criteria, metrics, optim, device)

