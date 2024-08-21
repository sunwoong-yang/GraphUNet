import numpy as np
import torch
from lib import scaling
def get_Ek(original, rec):
	"""
	Calculate energy percentage reconstructed

	Args:
			original : (NumpyArray) The ground truth

			rec      : (NumpyArray) The reconstruction from decoder

	Returns:

			The energy percentage for construction. Note that it is the Ek/100 !!
	"""

	TKE_real = original[:, 0, :, :] ** 2 + original[:, 1, :, :] ** 2

	u_rec = rec[:, 0, :, :]
	v_rec = rec[:, 1, :, :]

	return 1 - np.sum((original[:, 0, :, :] - u_rec) ** 2 + (original[:, 1, :, :] - v_rec) ** 2) / np.sum(TKE_real)


def get_spatial_modes(model, data_loader, z=1):
	latent_dim = model.HyperParams.mlp_layer[-1]
	# x = [[0 for _ in range(latent_dim)] for _ in range(len(data_loader))]
	x = [0] * latent_dim
	for latent_idx in range(latent_dim):
		x_temp = []
		# z_baseline = torch.zeros((1, latent_dim), dtype=torch.float64)
		# z_sample = torch.zeros((1, latent_dim), dtype=torch.float64)
		# z_sample[:, latent_idx] = z

		for data_idx, data in enumerate(data_loader): # 모든 snapshot 600개에 대한 data추출
			# if HyperParams.train_idx[0] <= data_idx < HyperParams.train_idx[1]:
			# if data_idx == 0:
			# 	data_fixed = data
			if data_idx == 0: # pooling_info wrt first snapshot is used
				with torch.no_grad():
					z_baseline = torch.zeros((len(data), latent_dim), dtype=torch.float64)
					z_sample = torch.zeros((len(data), latent_dim), dtype=torch.float64)
					z_sample[:, latent_idx] = z

					# x_pred_baseline = model.decoder(z_baseline, data_fixed)
					# x_pred = model.decoder(z_sample, data_fixed)
					if model.HyperParams.decoder_type in [3,5]:
						x_pred_baseline = model.decoder(z_baseline, data, device='cpu')
						x_pred = model.decoder(z_sample, data, device='cpu')
					elif model.HyperParams.decoder_type in [6]:
						z_pred, pooling_info = model.encoder(data)
						x_pred_baseline = model.decoder(z_baseline, pooling_info)
						x_pred = model.decoder(z_sample, pooling_info)
					else:
						x_pred_baseline = model.decoder(z_baseline, data)
						x_pred = model.decoder(z_sample, data)
					diff = (x_pred - x_pred_baseline).reshape(-1,len(data))
					x_temp.append(diff)
					# x_temp.append(x_pred - x_pred_baseline) orig
				break
				# x[latent_idx] = x_pred.reshape(-1,1,1)

		x[latent_idx] = torch.stack(x_temp, dim=1)

				# mode = model.decoder(torch.from_numpy(z_sample).to(device)).cpu().numpy()

	return x


def get_Ek(original, pred, scaler_all, HyperParams):
	# original and pred are temporal fields
	# original = scaling.inverse_scaling(original, scaler_all, HyperParams.scaling_type)
	# pred = scaling.inverse_scaling(pred, scaler_all, HyperParams.scaling_type)
	TKE_real = original ** 2
	return 1 - torch.mean(torch.sum((original - pred) ** 2, dim=0) / torch.sum(TKE_real, dim=0)) # snapshot ensemble should be