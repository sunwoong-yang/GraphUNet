import torch
from tqdm import tqdm
import numpy as np


def evaluate(model, loader, HyperParams):
    """
    This function evaluates the performance of a trained Autoencoder (AE) model.
    It encodes the input data using both the model's encoder and a mapping function,
    and decodes the resulting latent representations to obtain predicted solutions.
    The relative error between the two latent representations is also computed.

    Inputs:
    VAR: np.array, ground truth solution
    model: object, trained AE model
    loader: object, data loader for the input data
    params: np.array, model parameters
    HyperParams: class, model architecture and training parameters

    Returns:
    results: np.array, predicted solutions
    latents_map: np.array, latent representations obtained using the mapping function
    latents_gca: np.array, latent representations obtained using the AE encoder
    """
    def reshape_output(x, data):
        return x.view(len(data), -1).t().contiguous().view(-1,len(data))

    x_predicted, z_predicted = [], []
    index = 0
    with torch.no_grad():
        for data in loader:

            x, z = model(data)

            x = reshape_output(x, data)
            z = reshape_output(z, data) # ori
            # z = z.reshape((-1, len(data))) # trial
            x_predicted.append(x)
            z_predicted.append(z)
            index += 1

    return torch.cat(x_predicted, dim=1).unsqueeze(2), torch.cat(z_predicted, dim=1).unsqueeze(2) # origin


def evaluate_auto(model, init_loader):
    x_predicted, z_predicted = [], []
    index = 0
    with torch.no_grad():
        for data in init_loader:

            # if HyperParams.beta is not None:
            #     x, z, _, _ = model.time_stepping(z, data, device='cpu')
            # else:
            # if z_inital is None:
            x, z = model(data)
            z_predicted.append(z) ####
            # else:
            #     x, z_next, z = model.time_stepping(z_inital, data, device='cpu')
            #     z_predicted.append(z_next) ####

            x_predicted.append(x)
            # z_predicted.append(z)

            index += 1

    x_predicted = torch.stack(x_predicted, dim=1)
    z_predicted = torch.stack(z_predicted, dim=1)

    if init_loader.dataset[0].x.shape[1] == 1:
        init_loader.dataset[0].x = x_predicted.reshape(-1, 1)
    else: # move init_loader one step for the next prediction
        temp = init_loader.dataset[0].x[:,1:]
        init_loader.dataset[0].x[:,:-1] = temp.clone()
        init_loader.dataset[0].x[:,[-1]] = x_predicted.reshape(-1, 1)

    return x_predicted, z_predicted, init_loader

def plot_pooled_nodes(model, Net_train_loader, Net_train_loader_next, repeat_idx):
    with torch.no_grad():
        model.eval()

        if Net_train_loader_next is not None:
            for idx in range(len(Net_train_loader)):
                for data, data_next in zip(Net_train_loader[idx], Net_train_loader_next[idx]):
                    data = data
                    _, _ = model.encoder(data, if_plot_mesh=repeat_idx)
                    break
                break

        else:
            for idx in range(len(Net_train_loader)):
                for data in Net_train_loader[idx]:
                    data = data
                    _, _ = model.encoder(data, if_plot_mesh=repeat_idx)
                    break
                break

# def evaluate_fromZ(model, init_loader, z_inital=None):
#     x_predicted, z_predicted = [], []
#     index = 0
#     with torch.no_grad():
#         for data in init_loader:
#
#             # if HyperParams.beta is not None:
#             #     x, z, _, _ = model.time_stepping(z, data, device='cpu')
#             # else:
#             if z_inital is None:
#                 x, z, mu, log_var = model(data, device='cpu', infer=True)
#                 z_predicted.append(z) ####
#             else:
#                 x, z_next, z = model.time_stepping(z_inital, data, device='cpu')
#                 z_predicted.append(z_next) ####
#
#             x_predicted.append(x)
#             # z_predicted.append(z)
#
#             index += 1
#
#     x_predicted = torch.stack(x_predicted, dim=1)
#     z_predicted = torch.stack(z_predicted, dim=1)
#     if z_inital is None:
#         pass
#     else:
#         if init_loader.dataset[0].x.shape[1] == 1:
#             init_loader.dataset[0].x = x_predicted.reshape(-1, 1)
#         else: # move init_loader one step for the next prediction
#             temp = init_loader.dataset[0].x[:,1:]
#             init_loader.dataset[0].x[:,:-1] = temp.clone()
#             init_loader.dataset[0].x[:,[-1]] = x_predicted.reshape(-1, 1)
#
#     return x_predicted, z_predicted, init_loader
