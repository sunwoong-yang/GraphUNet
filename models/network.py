import torch
from torch import nn
from models import UNet
from torch_geometric.loader import DataLoader
from sklearn import preprocessing


class Net(torch.nn.Module):
    """
    A PyTorch neural network class for a VAE model with configurable encoder and decoder modules.

    Parameters
    ----------
    HyperParams : object
        An object containing all hyperparameters for the network including encoder, decoder configurations,
        and others such as activation function, layer dimensions for mapping, etc.

    Attributes
    ----------
    encoder : Module
        An encoder module for encoding input data.
    decoder : Module
        A decoder module for decoding encoded representations.
    act_map : function
        The activation function used in the network.

    Methods
    -------
    forward(data)
        Defines the forward pass of the network.
    reparameterize(mu, log_var)
        Reparameterizes the encoded representation for VAE.
    """

    def __init__(self, HyperParams, device):
        super().__init__()
        self.HyperParams = HyperParams
        self.encoder = UNet.Encoder(HyperParams)
        self.decoder = UNet.Decoder(HyperParams)
        self.to(device).double()

        # self.act_map = HyperParams.act

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std


    # forward without parameters
    def forward(self, data, plot_mesh=None):
        z, pooling_info = self.encoder(data, if_plot_mesh=plot_mesh)
        x = self.decoder(z, pooling_info)

        return x, z

    def time_stepping(self, z, data, device=('cuda' if torch.cuda.is_available() else "cpu")):

        z_prev, pooling_info = self.encoder(data)
        x = self.decoder(z, pooling_info)

        return x, z_prev, z


    def get_latent(self, data):
        z_list = []
        for data_ in data:
            # for data
            z, pooling_info = self.encoder(data_)

            z_list.append(z)

        return torch.cat(z_list, dim=0)

    def make_temp_dataset(self, Net_train_loader, scaler=None):
        Net_train_z = []
        for train_loader in Net_train_loader:
            z = self.get_latent(train_loader)
            # scaler = preprocessing.StandardScaler()
            # z = scaler.fit_transform(z)
            Net_train_z.append(z)
            # Net_scalers.append(scaler)
        Net_train_z_concat = torch.cat(Net_train_z, dim=0)

        if scaler is None:
            scaler = preprocessing.StandardScaler()
            scaler.fit(Net_train_z_concat.detach().numpy())

        for idx in range(len(Net_train_z)):
            Net_train_z[idx] = torch.tensor(scaler.transform(Net_train_z[idx].detach().numpy()))


        train_data_list = []
        train_data_label_list = []
        for z_list in Net_train_z:
            for idx in range(z_list.shape[0] - self.HyperParams.lstm_history):
                train_data_list.append(z_list[idx : idx + self.HyperParams.lstm_history])
                train_data_label_list.append(z_list[idx + self.HyperParams.lstm_history])


        train_loader = DataLoader(train_data_list, batch_size=len(train_data_list), shuffle=False)
        train_label_loader = DataLoader(train_data_label_list, batch_size=len(train_data_label_list), shuffle=False)

        return train_loader, train_label_loader, scaler
