
import torch
import torch.nn as nn


def Dice(pred, target, dims=(2 ,3), reduction='mean'):
	smooth = 1e-4
	pred = nn.Softmax(dim=1)(pred)
	# pred = nn.Sigmoid()(pred)
	intersection = (pred * target).sum(dim=dims)
	union = pred.sum(dim=dims) + target.sum(dim=dims)

	dice = torch.mean((2 * intersection + smooth) / (union + smooth))

	if reduction.lower() == 'sum':
		dice = torch.sum(dice)
	else:
		dice = torch.mean(dice)

	return dice


def iou(p, g, threshold=0.5):
	p = (p > threshold).float()
	p_ = p.contiguous().view(-1)
	g_ = g.contiguous().view(-1)
	intersection = (p_ * g_).sum()
	union = p_.sum() + g_.sum()
	smooth = 1e-4

	return (intersection + smooth) / (union - intersection + smooth)

def mAP(p, r):
	'''
	  p : precision
	  r : recall
	'''
	return 1

def recall(p, g):
	p_ = p.contiguous().view(-1)
	g_ = g.contiguous().view(-1)
	intersection = (p_ * g_).sum()
	smooth = 1
	p_sum = p_.sum()
	g_sum = g_.sum()

	return (intersection + smooth) / g_sum + smooth

def precision(p, g):
	p_ = p.contiguous().view(-1)
	g_ = g.contiguous().view(-1)
	intersection = (p_ * g_).sum()
	smooth = 1
	p_sum = p_.sum()
	g_sum = g_.sum()

	return (intersection + smooth) / p_sum + smooth

def F_score(p, r):
	return 2 * (p + r) / p * r



class GeneralizedDICELoss(nn.Module):
	'''
	  weigths and pred classes must have same size
	'''
	def __init__(self, weight=None, reduction='mean'):
		super(GeneralizedDICELoss, self).__init__()
		self.weigth = torch.tensor(weight)

	def forward(self, pred, target, dims=(2 ,3), threshhold=0.5, reduction='mean'):
		"""
		  args:
			pred : (N,C,H,W)->dim=(2,3), (N, H, W)->dim=(1,2), (H , W)->dim=None
			target : (N, C, H, W), (C, H, W) one_hot_eocoded
			theshhold : to be True
			reduction : 'mean', 'sum'
			default dim=(2,3), threshhold =0.5, reduction='mean'
		  return :
			1 - dice . mean reduction

		  target must be one_hot_encoded and has same number of channles as pred
		"""

		assert 1 < len(pred.size()) < 5 and 1 < len(target.size()) < 5

		smooth = 1e-4

		intersection = (pred * target).sum(dim=dims)
		union = pred.sum(dim=dims) + target.sum(dim=dims)

		dice = (2 * intersection + smooth) / (union + smooth)

		if reduction.lower() == 'sum':
			dice = torch.sum(self.weigth * dice)
		else:
			dice = torch.mean(self.weigth * dice)

		return 1 - dice


class DICELoss(nn.Module):
	def __init__(self):
		super(DICELoss, self).__init__()

	def forward(self, pred, target, dims=(2 ,3), reduction='mean'):
		"""
		  args:
			pred : (N,C,H,W)->dim=(2,3), (N, H, W)->dim=(1,2), (H , W)->dim=None
			target : (N, C, H, W), (C, H, W) one_hot_eocoded
			theshhold : to be True
			reduction : 'mean', 'sum'
			default dim=(2,3), threshhold =0.5, reduction='mean'
		  return :
			1 - dice . mean reduction

		  target must be one_hot_encoded and has same number of channles as pred
		"""

		assert 1 < len(pred.size()) < 5 and 1 < len(target.size()) < 5

		dice = Dice(pred, target, dims=dims, reduction=reduction)

		return 1 - dice



