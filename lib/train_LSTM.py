from models import LSTM
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR
def train(device, HyperParams, train_loader, train_label_loader, test_loader, test_label_loader):
	# Model parameters
	# if HyperParams.beta is not None:
	# 	input_size = output_size = int(HyperParams.mlp_layer[-1] / 2)
	# else:
	input_size = output_size = int(HyperParams.mlp_layer[-1])
	# input_size = 25  # The number of input features (25 features per time step)
	hidden_size = HyperParams.lstm_hidden_size  # The number of features in the hidden state
	num_layers = HyperParams.lstm_num_layers  # Number of recurrent layers
	# output_size = 25  # The size of the output

	# Initialize model, loss function, and optimizer
	model = LSTM.LSTM(input_size, hidden_size, num_layers, output_size).to(device).double()
	# criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)

	# Training loop
	num_epochs = HyperParams.lstm_epochs
	loop = tqdm(range(num_epochs))
	train_history = dict(train=[])
	test_history = dict(test=[])

	for epoch in loop:
		model.train()
		train_rmse = total_examples = 0
		optimizer.zero_grad()
		for (inputs, labels) in zip(train_loader, train_label_loader):

			inputs = inputs.to(device)
			labels = labels.to(device)

			# Forward pass
			outputs = model(inputs)
			loss = F.mse_loss(outputs, labels, reduction='mean')

			# Backward and optimize
			loss.backward(retain_graph=True)

			train_rmse += loss.item()
			total_examples += 1
		optimizer.step()
		scheduler.step()
		train_rmse = train_rmse / total_examples
		train_history['train'].append(train_rmse)

		test_rmse = total_examples_test = 0
		with torch.no_grad():
			model.eval()
			for (inputs, labels) in zip(test_loader, test_label_loader):
				inputs = inputs.to(device)
				labels = labels.to(device)
				outputs = model(inputs)
				loss = F.mse_loss(outputs, labels, reduction='mean')
				test_rmse += loss.item()
				total_examples_test += 1
			test_rmse = test_rmse / total_examples_test
			test_history['test'].append(test_rmse)

		loop.set_postfix({"Train": train_history['train'][-1], "Test": test_history['test'][-1]})

	np.save(HyperParams.net_dir + '/history_temp_train.npy', train_history)
	np.save(HyperParams.net_dir + '/history_temp_test.npy', test_history)
	torch.save(model, HyperParams.net_dir + '/model_temp.pt')

	return model