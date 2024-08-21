from lib import scaling
import torch
import torch.nn.functional as F
import numpy as np
import random
class HyperParams:
    """Class that holds the hyperparameters for the autoencoder model.

    Args:
        sparse_method (str): The method to use for sparsity constraint.
        rate (int): Amount of data used in training.
        seed (int): Seed for the random number generator.
        bottleneck_dim (int): The dimension of the bottleneck layer.
        tolerance (float): The tolerance value for stopping the training.
        lambda_map (float): The weight for the map loss.
        learning_rate (float): The learning rate for the optimizer.
        ffn (int): The number of feed-forward layers.
        in_channels (int): The number of input channels.
        hidden_channels (list): The number of hidden channels for each layer.
        act (function): The activation function to use.
        nodes (int): The number of nodes in each hidden layer.
        skip (int): The number of skipped connections.
        layer_vec (list): The structure of the network.
        net_name (str): The name of the network.
        scaler_name (str): The name of the scaler used for preprocessing.
        weight_decay (float): The weight decay for the optimizer.
        max_epochs (int): The maximum number of epochs to run training for.
        miles (list): The miles for learning rate update in scheduler.
        gamma (float): The gamma value for the optimizer.
        num_nodes (int): The number of nodes in the network.
        scaling_type (int): The type of scaling to use for preprocessing.
        net_dir (str): The directory to save the network in.
        cross_validation (bool): Whether to perform cross-validation.
    """

    def __init__(self):
        self.custom_name = 'Case_Folder_Name'
        self.beta = None
        self.folder_name = 'case_folder'
        self.UNet = False
        self.history = 5
        self.future = True
        self.train_list = [4]
        self.Norm = None
        self.testmode = 'auto,100'
        self.if_diff_snap_range = False

        ## Temporal prediction
        self.lstm_history = 30
        self.lstm_epochs = 3000
        self.lstm_hidden_size = 50
        self.lstm_num_layers = 5


        # self.variable = 'VX'
        self.scaling_type = 4 #4 #4 #default 4 #int(argv[2])
        self.scaler_number = 3 #int(argv[3]) default 3 : std scaler
        _, self.scaler_name = scaling.scaler_functions(self.scaler_number)
        self.skip = False #int(argv[4]) default: 1
        self.rate = 50 #16.6 #int(argv[5]) # default: 10
        # self.mlp_layer = [100,75,50] # last element is bottleneck dim
        # self.ffn = 50 #int(argv[6]) default: 200
        # self.nodes = 100 #int(argv[7])
        # self.bottleneck_dim = 50 #int(argv[8])
        # self.in_channels = 4 #int(argv[10])
        self.seed = 10
        self.stop_tolerance = 800
        self.learning_rate = 1e-3
        self.gcn_type = 0 # default: GMM
        self.gcn_k = 2
        self.Enc_HC = [40, 20, 10, 5, 1]
        self.Dec_HC = [1, 3, 5, 3, 1]
        self.act = F.elu
        # self.layer_vec= [2, self.nodes, self.nodes, self.nodes, self.nodes, self.mlp_layer[-1]] #[argv[11], self.nodes, self.nodes, self.nodes, self.nodes, self.bottleneck_dim]
        self.weight_decay = 0.00001
        self.max_epochs = 10000 #argv[12] default 5000
        self.lr_step = 1000
        self.lr_gamma = 0.8
        # self.num_nodes = 0
        # Customizing
        self.in_channels = len(self.Enc_HC)
        self.train_idx = None
        self.pooling = [1000, 500, 250, 100]
        self.zero_unpooling = True
        self.my_pooling = False
        self.augmentation_order = 1
        self.noise_type = 0
        self.noise_size = 0.02
        # self.pooling = [0.5]*(self.in_channels-2) + [250] # None
        self.minibatch = 1
        self.n_repeat = 1
        self.trans = None

        self.option_list = '[' + self.custom_name + ']' + '_beta' + str(self.beta) + '_pool' + str(self.pooling) \
                                + '_norm' + str(self.Norm) + '_seed' + str(self.seed) + '_GCN' + str(self.gcn_type) + '_EncHC' + str(self.Enc_HC) + '_DecHC' + str(self.Dec_HC) + '_hist' + str(self.history) \
                                + '_skip' + str(self.skip) + '_lr' + str(self.learning_rate) + '_sc' + str(self.scaling_type) + '_trainlist' + str(self.train_list) \
                                + '_trainidx' + str(self.train_idx) + '_mbat' + str(self.minibatch) + '_iter' + str(self.max_epochs) + '_LRinfo' + str(self.lr_step) + '-' + str(self.lr_gamma) \
                                + '_ZeroUnp' + str(self.zero_unpooling) + '_MyPooling' + str(self.my_pooling) + '_Aug' + str(self.augmentation_order) + '_NoiseType' + str(self.noise_type) \
                                + '_NoiseSize' + str(self.noise_size) + '_repeat' + str(self.n_repeat) + '_trans' + str(self.trans)

        # self.net_dir = './' + self.folder_name + '/' + self.option_list
        self.net_dir = './' + self.folder_name + '/' + self.custom_name
        self.cross_validation = True
        # self.set_reproducibility()

    def update(self, **kwargs):
        """
        Updates the hyperparameters with the provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments representing the hyperparameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid hyperparameter and will be ignored.")
        if not self.future:
            self.history = 1
        self.update_dir()
        # self.set_reproducibility()

    def update_dir(self):
        self.in_channels = len(self.Enc_HC)
        # self.decoder_hidden_channels = [self.mlp_layer[-1]] + [1] * (len(self.hidden_channels) - 1)
        # self.option_list = (self.custom_name if self.custom_name is not None else '') + '_beta' + str(self.beta) + '_pool' + str(self.pooling) + '_mlp' + str(self.mlp_layer) \
        #                         + '_norm' + str(self.Norm) + '_seed' + str(self.seed) + '_EncHC' + str(self.Enc_HC) + '_DecHC' + str(self.Dec_HC) + '_hist' + str(self.history) \
        #                         + '_skip' + str(self.skip) + '_lr' + str(self.learning_rate) + '_sc' + str(self.scaling_type) + '_train_list' + str(self.train_list) \
        #                         + '_train_idx' + str(self.train_idx) + '_mbat' + str(self.minibatch) + '_iter' + str(self.max_epochs) + '_LRinfo' + str(self.lr_step) + '-' + str(self.lr_gamma)
        self.option_list = '[' + self.custom_name + ']' + '_beta' + str(self.beta) + '_pool' + str(self.pooling) \
                                + '_norm' + str(self.Norm) + '_seed' + str(self.seed) + '_GCN' + str(self.gcn_type) + '_EncHC' + str(self.Enc_HC) + '_DecHC' + str(self.Dec_HC) + '_hist' + str(self.history) \
                                + '_skip' + str(self.skip) + '_lr' + str(self.learning_rate) + '_sc' + str(self.scaling_type) + '_train_list' + str(self.train_list) \
                                + '_train_idx' + str(self.train_idx) + '_mbat' + str(self.minibatch) + '_iter' + str(self.max_epochs) + '_LRinfo' + str(self.lr_step) + '-' + str(self.lr_gamma) \
                                + '_ZeroUnp' + str(self.zero_unpooling) + '_MyPooling' + str(self.my_pooling) + '_Aug' + str(self.augmentation_order) + '_Noise' + str(self.noise_type) \
                                + '_NoiseSize' + str(self.noise_size) + '_repeat' + str(self.n_repeat) + '_trans' + str(self.trans)
        self.net_dir = './' + self.folder_name + '/' + self.custom_name

    def set_reproducibility(self):
        """
        Sets the seed for reproducibility of results.

        Args:
            HyperParams (class): Contains the hyperparameters of the autoencoder
        """

        seed = self.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True