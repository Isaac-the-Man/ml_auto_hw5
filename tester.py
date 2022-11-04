# SET TEST DIRECTORY HERE
images_path = 'images_path'
labels_path = 'labels_path'







import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision
from prettytable import PrettyTable
from model import CNN

from dataset_new import BuildingDataset
from train_buildings import test

import numpy as np

import matplotlib.pyplot as plt

import os, sys
from tqdm import tqdm # DELETE AFTERWARD

IMAGE_HEIGHT = 3024
IMAGE_WIDTH = 4032

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def count_parameters(model):
	'''
	function to display network parameters from stackoverflow
	source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
	'''
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad: continue
		params = parameter.numel()
		table.add_row([name, params])
		total_params+=params
	print(table)
	print(f"Total Trainable Params: {total_params:,}")
	return total_params


def main():

	# image parameters
	resize_factor = 10
	new_h = int(IMAGE_HEIGHT / resize_factor)
	new_w = int(IMAGE_WIDTH / resize_factor)

	normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
	resize = torchvision.transforms.Resize(size = (new_h, new_w))
	convert = torchvision.transforms.ConvertImageDtype(torch.float)

	test_transforms = torchvision.transforms.Compose([resize, convert, normalize])

	# create dataset
	test_dataset = BuildingDataset(f'{labels_path}/test_labels.csv', images_path, transform_pre=test_transforms)

	# hyperparams
	train_batch_size = 64
	test_batch_size = 64
	n_epochs = 20
	learning_rate = 1e-3
	seed = 100
	input_dim = (3, new_h, new_w)
	out_dim = 11
	momentum = 0.9

	# create dataloader
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

	# create model
	network = CNN(in_dim=input_dim, out_dim=out_dim)
	network = network.to(device)

	# load pretrained model
	network.load_state_dict(torch.load('models/cnn_buildings.pth'))
	network.eval()

	# display parameters
	count_parameters(network)

	# test model with test set
	test(network, test_loader, device)

if __name__ == '__main__':
	main()