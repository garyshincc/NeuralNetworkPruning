import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import copy

import numpy as np

from torchvision import datasets, transforms

torch.set_printoptions(profile="default")

train_loader = torch.utils.data.DataLoader(
				datasets.MNIST('data/', train=True, download=True,
											 transform=transforms.Compose([
													 transforms.ToTensor(),
													 transforms.Normalize((0.1307,), (0.3081,))
											 ])),
				batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data/', train=False, transform=transforms.Compose([
											 transforms.ToTensor(),
											 transforms.Normalize((0.1307,), (0.3081,))
									 ])),
		batch_size=1000, shuffle=True)

linear_model = LinearNetwork()

optimizer = optim.Adam(linear_model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()


##### train the network
device = 'cuda'
linear_model.train()
linear_model.to(device)
for i, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		data = data.view(-1, 1, 28*28)
		
		optimizer.zero_grad()
		output = linear_model(data)
		output = output.squeeze(1)
		output = F.softmax(output)
		
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if i % 100 == 0:
			print(loss)


linear_model.eval()
for i, (data, target) in enumerate(test_loader):
		data, target = data.to(device), target.to(device)
		data = data.view(-1, 1, 28*28)
		
		output = linear_model(data)
		output = output.squeeze(1)
		output = F.softmax(output)
		val, index = output.max(1)
		correct = (index == target)
		print(float(correct.sum()) / float(len(correct)))
		break



##### Initialize the convolutional network
conv_model = ConvNetwork()

optimizer_c = optim.Adam(conv_model.parameters(), lr=0.00025)
criterion_c = nn.CrossEntropyLoss()




##### train the conv network
device = 'cuda'
conv_model.train()
conv_model.to(device)
for i, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer_c.zero_grad()
		output = conv_model(data)
		output = output.squeeze(1)
		output = F.softmax(output)
		
		loss = criterion_c(output, target)
		loss.backward()
		optimizer_c.step()
		if i % 100 == 0:
			print(loss)
				



conv_model.eval()
for i, (data, target) in enumerate(test_loader):
		data, target = data.to(device), target.to(device)
		
		output = conv_model(data)
		output = output.squeeze(1)
		output = F.softmax(output)
		val, index = output.max(1)
		correct = (index == target)
		print(float(correct.sum()) / float(len(correct)))
		break


def get_accuracy(_model):
	_model.eval()
	for i, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			data = data.view(-1, 1, 28*28)
			output = _model(data)
			output = output.squeeze(1)
			output = F.softmax(output)
			val, index = output.max(1)
			correct = (index == target)
			return (float(correct.sum()) / float(len(correct)))




def train_cnn(_model):
	##### train the conv network
	device = 'cuda'
	_model.train()
	_model.to(device)
	_optimizer_c = optim.Adam(_model.parameters(), lr=0.00025)
	_criterion_c = nn.CrossEntropyLoss()
	for i, (data, target) in enumerate(train_loader):
			data, target = data.to(device), target.to(device)
			_optimizer_c.zero_grad()
			output = _model(data)
			output = output.squeeze(1)
			output = F.softmax(output)

			loss = _criterion_c(output, target)
			loss.backward()
			_optimizer_c.step()
	print(f"done training with loss: {loss}")

def get_accuracy_conv(_model):
	_model.eval()
	for i, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			output = _model(data)
			output = output.squeeze(1)
			output = F.softmax(output)
			val, index = output.max(1)
			correct = (index == target)
			return (float(correct.sum()) / float(len(correct)))

def prune_neurons(layer, k):
	for parameter in layer.parameters():
		dims = parameter.shape[0]
		n_to_prune = int(k * dims)
		neuron_sum = parameter.abs().sum(1)
		sorted_ixs = np.argsort(list(neuron_sum))
		print(f"dims: {dims}, pruning: {n_to_prune}")
		for i in range(0, n_to_prune):
			parameter[sorted_ixs[i]] = torch.zeros(parameter[sorted_ixs[i]].shape)

# lets see if the pruning works!
# linear model first
# change dim from 0 or 1 to either clear out a row/colum (neuron/feature)

model2 = copy.deepcopy(linear_model)
k_list = [0, .25, .50, .60, .70, .80, .90, .95, .97, .99]
for k in k_list:
	prune_neurons(model2.fc3, k)
	prune_neurons(model2.fc4, k)
	acc = get_accuracy(model2)
	print(f"acc: {acc}")

conv_model2 = copy.deepcopy(conv_model)
k_list = [0, .25, .50, .60, .70, .80, .90, .95, .97, .99]
for k in k_list:
	prune_neurons(conv_model2.fc1, k)
	prune_neurons(conv_model2.fc2, k)
	acc = get_accuracy_conv(conv_model2)
	print(f"acc: {acc}")

def prune_weights(layer, k):
	for parameter in layer.parameters():
		n_weights = parameter.shape[1]
		n_to_prune = int(k * n_weights)
		weight_sum = parameter.abs().sum(0)
		sorted_ixs = np.argsort(list(weight_sum))
		print(f"n_weights: {n_weights}, pruning: {n_to_prune}")
		for i in range(0, n_to_prune):
			parameter = parameter.t()
			parameter[sorted_ixs[i]] = torch.zeros(parameter[sorted_ixs[i]].shape, requires_grad=True, device="cuda")
			parameter = parameter.t()

# lets see if the pruning works!
# linear model first
# change dim from 0 or 1 to either clear out a row/colum (neuron/feature)

model2 = copy.deepcopy(linear_model)
k_list = [0, .25, .50, .60, .70, .80, .90, .95, .97, .99]
for k in k_list:
	prune_weights(model2.fc3, k)
	prune_weights(model2.fc4, k)
	acc = get_accuracy(model2)
	print(f"acc: {acc}")

conv_model2 = copy.deepcopy(conv_model)
k_list = [0, .25, .50, .60, .70, .80, .90, .95, .97, .99]
for k in k_list:
	prune_weights(conv_model2.fc1, k)
	prune_weights(conv_model2.fc2, k)
	acc = get_accuracy_conv(conv_model2)
	print(f"acc: {acc}")

