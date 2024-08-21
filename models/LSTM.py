import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size):
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		# LSTM layer
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		# Fully connected layer
		# self.fc1 = nn.Linear(hidden_size, hidden_size)
		# self.fc2 = nn.Linear(hidden_size, output_size)
		# self.act = F.elu
		self.fc = nn.Linear(hidden_size, output_size) # original

	def forward(self, x):
		# Initialize hidden and cell states
		h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, dtype=torch.double).to(x.device)
		c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, dtype=torch.double).to(x.device)

		# Forward propagate LSTM
		out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size)

		# Decode the hidden state of the last time step
		# out = self.act(self.fc1(out[:, -1, :]))
		# out = self.fc2(out)
		out = self.fc(out[:, -1, :]) #original
		return out
