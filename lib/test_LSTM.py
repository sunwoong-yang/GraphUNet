import torch
from tqdm import tqdm
import numpy as np
import copy

def get_next_z(model, data, scaler):
	# Inputted data is tensor, not dataloader
	y_list = []
	model.to("cpu")
	with torch.no_grad():
		data_reshaped = data.reshape(-1,data.shape[-1])
		data_reshaped = scaler.transform(data_reshaped)
		data = torch.tensor(data_reshaped.reshape(data.shape))

		if data.dim() != 3:
			data = data.unsqueeze(dim=0)
		y = model(data)

		y_reshaped = y.reshape(-1, y.shape[-1])
		y_reshaped = scaler.inverse_transform(y_reshaped)
		y = torch.tensor(y_reshaped.reshape(y.shape))

		y_list.append(y)

	return torch.cat(y_list, dim=0)

def evaluate_ROM(gca_model, temp_model, init_loader, scaler):
	x_next_list = z_prev_list = []
	gca_model.eval()
	temp_model.eval()
	with torch.no_grad():
		for data in init_loader:
			z_prev, pooling_info = gca_model.encoder(data)
			if gca_model.beta is not None:
				mu, log_var = torch.chunk(z_prev, 2, dim=1)
				z_prev = mu
			z_prev_list.append(z_prev)

		z_prev_list = torch.cat(z_prev_list, dim=0)
		z_next = get_next_z(temp_model, z_prev_list, scaler)

		x_next = gca_model.decoder(z_next, pooling_info) # Use only last data's pooling_info
		x_next_list.append(x_next)

	# Update init_loader
	next_loader = copy.deepcopy(init_loader)
	for idx in range(len(init_loader.dataset)-1):
		next_loader.dataset[idx].x = init_loader.dataset[idx+1].x
	next_loader.dataset[-1].x = x_next

	return next_loader, x_next
