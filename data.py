import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_SHAPE = {
	'mnist' : {'in_channels' : 1, 'shp' : (28, 28)},
	'cifar' : {'in_channels' : 3, 'shp' : (32, 32)},
	'svhn' : {'in_channels' : 3, 'shp' : (32, 32)}
}

def mnist_transform():
	return transforms.Normalize((0.1307,), (0.3081,))

def cifar_transform():
	return transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
								std=[0.2470, 0.2435, 0.2616])

def svhn_transform():
	return transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
						 		std=[0.1980, 0.2010, 0.1970])

def mnist_loaders(train_batch_size, args, test_batch_size=None):
		if test_batch_size is None:
				test_batch_size = train_batch_size

		trainLoader = torch.utils.data.DataLoader(
				dset.MNIST(args.data_cache_dir,
						 train=True,
						 download=True,
						 transform=transforms.Compose([
								 transforms.ToTensor()
						 ])),
				batch_size=train_batch_size,
				shuffle=True)

		testLoader = torch.utils.data.DataLoader(
				dset.MNIST(args.data_cache_dir,
						 train=False,
						 transform=transforms.Compose([
								 transforms.ToTensor()
						 ])),
				batch_size=test_batch_size,
				shuffle=False)
		return trainLoader, testLoader


def cifar_loaders(train_batch_size, args, test_batch_size=None, augment=True, crop_pad=4):
		if test_batch_size is None:
				test_batch_size = train_batch_size

		if augment:
				transforms_list = [transforms.RandomHorizontalFlip(),
								   transforms.RandomCrop(32, crop_pad, padding_mode='edge'),
								   transforms.ToTensor()]
		else:
				transforms_list = [transforms.ToTensor()]
		train_dset = dset.CIFAR10(args.data_cache_dir,
								  train=True,
								  download=True,
								  transform=transforms.Compose(transforms_list))
		test_dset = dset.CIFAR10(args.data_cache_dir,
								 train=False,
								 transform=transforms.Compose([
										 transforms.ToTensor()
								 ]))

		trainLoader = torch.utils.data.DataLoader(train_dset, batch_size=train_batch_size,
												  shuffle=True, pin_memory=True)

		testLoader = torch.utils.data.DataLoader(test_dset, batch_size=test_batch_size,
												 shuffle=False, pin_memory=True)

		return trainLoader, testLoader

def svhn_loaders(train_batch_size, args, test_batch_size=None):
		if test_batch_size is None:
				test_batch_size = train_batch_size

		train_loader = torch.utils.data.DataLoader(
						dset.SVHN(
								root=args.data_cache_dir, split='train', download=True,
								transform=transforms.Compose([
										transforms.ToTensor()
								]),
						),
						batch_size=train_batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(
				dset.SVHN(
						root=args.data_cache_dir, split='test', download=True,
						transform=transforms.Compose([
								transforms.ToTensor()
						])),
				batch_size=test_batch_size, shuffle=False)
		return train_loader, test_loader