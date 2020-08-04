from Dataset import KvasirSegDataset
from Metrics import iou, DICELoss
from model import ResUnetPP
from train import train
from util import load_model
from eval import evaluate
from predict import predict
from torch.utils.data.dataloader import DataLoader
import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

if __name__ == '__main__':
	train_path = 'D:\\Kvasir-SEG\\train'
	val_path = 'D:\\Kvasir-SEG\\val'

	batch_size = 2
	start_epoch = 0
	epochs = 130
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	# device = torch.device('cpu')
	print(device)


	t_transforms = A.Compose([
		# A.RandomBrightnessContrast(p=0.5),
		# A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.5),
		# A.RandomRotate90(p=0.5),
		# A.ShiftScaleRotate(p=0.5),
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


	train_dataset = KvasirSegDataset(train_path, t_transforms)
	val_dataset = KvasirSegDataset(val_path, v_transforms)

	train_dataloader = DataLoader(train_dataset, batch_size)
	val_dataloader = DataLoader(val_dataset, batch_size)


	in_classes = 3
	out_classes = 1
	model = ResUnetPP(in_classes, out_classes)
	criteria = DICELoss()
	metrics = {'iou': iou}

	optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
	# lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.001, max_lr= 0.01)
	# lr_s = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100, 0.001)

	# load_model('ResUnetPP_E99_8.pth', model, optim, map_location=device)
	# train(train_dataloader, val_dataloader, model, epochs, criteria, metrics, optim, scheduler=None, device=device)

	evaluate(model, 'D:\\Kvasir-SEG\\val', device, loss_func=criteria, f_name='ResUnetPP_aug_eval',
	        model_path='ResUnetPP_E47_5.pth')

	# predict(model, 'D:\\Kvasir-SEG\\train', device, f_name='ResUnetPP_aug_pred_masks',
	#       model_path='D:\\Kvasir-SEG\\ResUnetPP_E240_5.pth')

