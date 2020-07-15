from Dataset import KvasirSegDataset
from Metrics import iou, DICELoss
from model import ResUnet
from train import train

from predict import predict
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

if __name__ == '__main__':
	train_path = 'Kvasir-SEG/train'
	val_path = 'Kvasir-SEG/val'

	batch_size = 8
	start_epoch = 0
	epochs = 130
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device['cpu']
	print(device)

	t_transforms = A.Compose([
		# A.RandomBrightnessContrast(p=0.5),
		A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.5),
		A.RandomRotate90(p=0.5),
		A.ShiftScaleRotate(p=0.5),
		A.Resize(256, 256),
		A.Normalize(),
		ToTensorV2()
	])

	v_transforms = A.Compose([
		# A.RandomBrightnessContrast(p=0.5),
		# A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.5),
		# A.RandomRotate90(p=0.5),
		# A.ShiftScaleRotate(p=0.5),
		A.Resize(256, 256),
		A.Normalize(),
		ToTensorV2()
	])

	aa = t_transforms.get_dict_with_id()

	train_dataset = KvasirSegDataset(train_path, t_transforms)
	val_dataset = KvasirSegDataset(val_path, v_transforms)

	train_dataloader = DataLoader(train_dataset, batch_size)
	val_dataloader = DataLoader(val_dataset, batch_size)


	in_classes = 3
	out_classes = 1
	model = ResUnet(in_classes, out_classes)
	criteria = DICELoss()
	metrics = {'iou': iou}

	optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99, weight_decay=0.0005)
	lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.001, max_lr= 0.01)
	lr_s = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100, 0.001)
	train(train_dataloader, val_dataloader, model, epochs, criteria, metrics, optim, scheduler=lr_s, device=device)
	predict(model, 'D:\\Kvasir-SEG', device,
	        'C:\\Users\\DSKIM\\Google 드라이브\\AI\\medical-projects\\Kvasir-Seg\\unet_aug_models\\Unet_129_21.pth')

