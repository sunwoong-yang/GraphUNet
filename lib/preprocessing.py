import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from lib import scaling, get_data_info
import tensorflow as tf
import copy
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def triangles_to_edges(faces):
      """Computes mesh edges from triangles.
         Note that this triangles_to_edges method was provided as part of the
         code release for the MeshGraphNets paper by DeepMind, available here:
         https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
      """
      # collect edges from triangles
      edges = tf.concat([faces[:, 0:2],
                         faces[:, 1:3],
                         tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
      # those edges are sometimes duplicated (within the mesh) and sometimes
      # single (at the mesh boundary).
      # sort & pack edges as single tf.int64
      receivers = tf.reduce_min(edges, axis=1)
      senders = tf.reduce_max(edges, axis=1)
      packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
      # remove duplicates and unpack
      unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
      senders, receivers = tf.unstack(unique_edges, axis=1)
      # create two-way connectivity
      return (tf.concat([senders, receivers], axis=0),
              tf.concat([receivers, senders], axis=0))
def graphs_dataset(dataset, HyperParams, tr_snap_range = None, print_info=False, truncate_mesh=None):
    """
    graphs_dataset: function to process and scale the input dataset for graph autoencoder model.

    Inputs:
    dataset: an object containing the dataset to be processed.
    HyperParams: an object containing the hyperparameters of the graph autoencoder model.

    Outputs:
    dataset_graph: an object containing the processed and scaled dataset.
    loader: a DataLoader object of the processed and scaled dataset.
    train_loader: a DataLoader object of the training set.
    test_loader: a DataLoader object of the test set.
    val_loader: a DataLoader object of the validation set.
    scaler_all: a scaler object to scale the entire dataset.
    scaler_test: a scaler object to scale the test set.
    xyz: an list containig array of the x, y and z-coordinate of the nodes.
    var: an array of the node features.
    VAR_all: an array of the scaled node features of the entire dataset.
    VAR_test: an array of the scaled node features of the test set.
    train_snapshots: a list of indices of the training set.
    test_snapshots: a list of indices of the test set.
    """


    N = HyperParams.history
    xx = torch.tensor(dataset["mesh_pos"][0,:,0]) # shape=(1896)
    yy = torch.tensor(dataset["mesh_pos"][0,:,1]) # shape=(1896)


    if truncate_mesh is not None:
        # sliced_parts = np.concatenate((np.array(range(0, 29)), np.array(range(57, 500))))
        sliced_parts = np.where(xx > truncate_mesh)[0]
        nodes_mask = np.zeros(len(xx), dtype=bool)
        nodes_mask[sliced_parts] = True
        xyz = [xx[~nodes_mask], yy[~nodes_mask]]
    else:
        xyz = [xx, yy]


    if dataset["mesh_pos"].shape[2] == 3:
       zz = dataset["mesh_pos"][0,:,2]
       xyz.append(zz)
    # vel_x = torch.tensor(dataset['velocity'][:,:,0])  # shape=(600, 1896)
    edges = triangles_to_edges(torch.tensor(np.array(dataset['cells'][0])))
    if truncate_mesh is not None:
        vel_x = torch.tensor(dataset['velocity'][:, ~nodes_mask, 0])  # shape=(600, 1896)
        edge_mask = ~np.isin(edges[0].numpy(), sliced_parts) & ~np.isin(edges[1].numpy(), sliced_parts)
        edges_0 = torch.tensor(edges[0].numpy())[edge_mask]
        edges_1 = torch.tensor(edges[1].numpy())[edge_mask]

        new_indices_map = torch.zeros(len(xx), dtype=torch.long) -1 # Allocate a tensor for the new indices
        counter = 0
        for i in range(len(xx)):
            if not nodes_mask[i]:  # 지워지는 node가 아닐경우
                new_indices_map[i] = counter
                counter += 1

        edges_0 = new_indices_map[edges_0]
        edges_1 = new_indices_map[edges_1]
        valid_edges = (edges_0 >= 0) & (edges_1 >= 0)
        edges_0 = edges_0[valid_edges]
        edges_1 = edges_1[valid_edges]
        edge_index = torch.cat((edges_0.unsqueeze(0), edges_1.unsqueeze(0)), dim=0).type(torch.long)
    else:
        vel_x = torch.tensor(dataset['velocity'][:, :, 0])  # shape=(600, 1896)
        edge_index = torch.cat((torch.tensor(edges[0].numpy()).unsqueeze(0),
                                torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

    edge_index = edge_index.transpose(0, 1)  # shape=(10908, 2)

    if print_info:
        # target = [torch.tensor([0.6, 0.15])]
        # get_data_info.get_closest_point(target, xx, yy, vel_x)
        target = [torch.tensor([0., 0.205])]
        get_data_info.get_closest_point(target, xx, yy, vel_x)
        # target = [torch.tensor([0.6, 0.25])]
        # get_data_info.get_closest_point(target, xx, yy, vel_x)
    # get_closest_point(target, xx, yy, torch.tensor(dataset['velocity'][:,:,1]))
    """
    In case of "04_navier_stokes_vx.py"
    -> # of nodes = 2719
    -> # of graph (=snapshot =parameter combinations) = 341
    -> xx.shape = yy.shape = var.shape = (2719, 341)
    -> len(xyz) = [xx, yy]
    -> dataset.E.shape = (7786, 2)
    ---> 7786 denotes there are 7786 edges in the graph, and element in dataset.E indicates a set of two integers,
         which means the edge is connected with ith and jth points.    
    """

    # PROCESSING DATASET
    num_nodes = xyz[0].shape[0] # vel_x.shape[1]
    num_graphs = vel_x.shape[0] # number of snapshot (time steps)

    print("Number of nodes processed: ", num_nodes)
    print("Number of edges processed: ", edge_index.shape)
    print("Number of snapshots processed: ", num_graphs)
    total_sims = int(num_graphs)
    rate = HyperParams.rate/100
    train_sims = int(rate * total_sims)
    test_sims = total_sims - train_sims
    main_loop = np.arange(total_sims).tolist()
    # Since temporal prediction, first 'train_sims' snapshots are used for training, and the remaining for testing
    # np.random.shuffle(main_loop)

    ##Split train/test with the predefined ratio (HyperParams.rate)
    if isinstance(HyperParams.train_idx, list) and len(HyperParams.train_idx) >= 2:
        if tr_snap_range is not None:
            start, end = tr_snap_range[0], tr_snap_range[1]
        else:
            start, end = HyperParams.train_idx[0], HyperParams.train_idx[1]
        train_snapshots = main_loop[start: end]
        # train_snapshots_next = main_loop[start+N: end]
        test_snapshots = main_loop[end: total_sims]
        init_snapshots = main_loop[end: end + N]
        init_ROM_snapshots = main_loop[end: end + HyperParams.lstm_history]
        # start, end = HyperParams.train_idx[0], HyperParams.train_idx[1]
        # train_snapshots = main_loop[start : end]
        # # train_snapshots_next = main_loop[start+N: end]
        # test_snapshots = main_loop[end : total_sims]
        # init_snapshots = main_loop[end : end+N]
        # init_ROM_snapshots = main_loop[end : end + HyperParams.lstm_history]
    elif HyperParams.train_idx is not None and not (isinstance(HyperParams.train_idx, list) and len(HyperParams.train_idx) == 2):
        raise ValueError("********** Invalid HyperParams.train_idx **********\n" 
                         "It must be a list with exactly two elements.")
    else:
        train_snapshots = main_loop[0:train_sims]
        test_snapshots = main_loop[train_sims:total_sims]
        init_snapshots = main_loop[train_sims:train_sims+N]
    train_snapshots.sort()
    # train_snapshots_next.sort()
    test_snapshots.sort()

    ## FEATURE SCALING
    # vel_x_test = dataset['velocity'][test_snapshots,:,0]
    vel_x_test = vel_x[test_snapshots]
    vel_x_train = vel_x[train_snapshots]

    scaling_type = HyperParams.scaling_type
    # scaler_all, VAR_all_temp = scaling.tensor_scaling(vel_x, scaling_type, HyperParams.scaler_number) # Original version
    scaler_all, _ = scaling.tensor_scaling(vel_x_train, scaling_type, HyperParams.scaler_number) # Get scaler only from train data
    VAR_all = torch.tensor(scaler_all.transform(vel_x)).t().unsqueeze(-1)
    scaler_test, VAR_test = scaling.tensor_scaling(vel_x_test, scaling_type, HyperParams.scaler_number)


    graphs = []
    edge_index = torch.t(edge_index) - 0 # edge_index.shape=(2, 10908)

    ########## (erase) don't need for loop per graph since this is temporal prediction
    for graph in range(num_graphs):
        if dataset["mesh_pos"].shape[2] == 2:
            pos = torch.cat((xx.unsqueeze(1), yy.unsqueeze(1)), 1)
        elif dataset["mesh_pos"].shape[2] == 3:
            pos = torch.cat((xx.unsqueeze(1), yy.unsqueeze(1), zz.unsqueeze(1)), 1)
        # print(pos.shape, torch.min(pos[:,0]), torch.max(pos[:,0]), torch.min(edge_index[0, :]), torch.max(edge_index[0, :]))
        if truncate_mesh is not None:
            pos = torch.cat((xyz[0].unsqueeze(1), xyz[1].unsqueeze(1)), 1)

        ei = torch.index_select(pos, 0, edge_index[0, :]) # pos: torch.Size([1896, 2]) & edge_index torch.Size([2, 10908])
        ej = torch.index_select(pos, 0, edge_index[1, :])

        edge_diff = ej - ei
        if dataset["mesh_pos"].shape[2] == 2:
            edge_attr = torch.sqrt(torch.pow(edge_diff[:, 0], 2) + torch.pow(edge_diff[:, 1], 2))
        elif dataset["mesh_pos"].shape[2] == 3:
            edge_attr = torch.sqrt(torch.pow(edge_diff[:, 0], 2) + torch.pow(edge_diff[:, 1], 2) + torch.pow(edge_diff[:, 2], 2))

        node_features = VAR_all[:, graph, :]
        dataset_graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        graphs.append(dataset_graph)



    HyperParams.num_nodes = dataset_graph.num_nodes

    train_dataset = [graphs[i] for i in train_snapshots]
    test_dataset = [graphs[i] for i in test_snapshots]
    init_dataset = [graphs[i] for i in init_snapshots]
    init_ROM_dataset = [graphs[i] for i in init_ROM_snapshots]

    def make_prev_next(train_dataset, N, future=True):
        train_dataset_prev = []
        train_dataset_next = []
        for idx in range(len(train_dataset) - N + 0):
            data_prev = copy.deepcopy(train_dataset[idx])
            data_next = copy.deepcopy(train_dataset[idx])
            data_stack = [train_dataset[i].x for i in range(idx, idx + N)]
            data_prev.x = torch.concat(data_stack, dim=1)
            if future:
                data_next.x = train_dataset[idx + N].x
            elif not future and N ==1:
                data_next.x = train_dataset[idx + N - 1].x
            else:
                raise ValueError("Though future prediction (HyperParams.future) is False, HyperParams.history is not equal to 1")
            train_dataset_prev.append(data_prev)
            train_dataset_next.append(data_next)

        return train_dataset_prev, train_dataset_next

    train_dataset_prev, train_dataset_next = make_prev_next(train_dataset, N, future=HyperParams.future)
    test_dataset_prev, test_dataset_next = make_prev_next(test_dataset, N, future=HyperParams.future)

    init_dataset_ = copy.deepcopy(init_dataset)
    init_dataset_prev = [init_dataset[i].x for i in range(N)]
    init_dataset_prev = torch.concat(init_dataset_prev, dim=1)
    init_dataset_[0].x = init_dataset_prev
    init_dataset_ = [init_dataset_[0]]
    # print('inita',len(init_dataset),init_dataset[0].shape)

    print("Length of train dataset: ", len(train_dataset))
    print("Length of test dataset: ", len(test_dataset))

    loader = DataLoader(graphs, batch_size=1)
    # batch_size를 여러개 가져가면 pooling에서 문제발생해서 그냥 1로 수정
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    if HyperParams.minibatch is None: # whole dataset is regarded as single mini-batch
        # train_loader = DataLoader(train_dataset, batch_size=train_sims, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        train_loader_prev = DataLoader(train_dataset_prev, batch_size=len(train_dataset_prev), shuffle=False)
        train_loader_next = DataLoader(train_dataset_next, batch_size=len(train_dataset_next), shuffle=False)
        test_loader_prev = DataLoader(test_dataset_prev, batch_size=len(test_dataset_prev), shuffle=False)
        test_loader_next = DataLoader(test_dataset_next, batch_size=len(test_dataset_next), shuffle=False)
        # init_loader_ROM =  DataLoader(test_dataset_next, batch_size=1, shuffle=False) # new
    else:
        train_loader = DataLoader(train_dataset, batch_size=HyperParams.minibatch, shuffle=False)
        train_loader_prev = DataLoader(train_dataset_prev, batch_size=HyperParams.minibatch, shuffle=False)
        train_loader_next = DataLoader(train_dataset_next, batch_size=HyperParams.minibatch, shuffle=False)
        test_loader_prev = DataLoader(test_dataset_prev, batch_size=HyperParams.minibatch, shuffle=False)
        test_loader_next = DataLoader(test_dataset_next, batch_size=HyperParams.minibatch, shuffle=False)
        # init_loader_ROM = DataLoader(test_dataset_next, batch_size=1, shuffle=False) # new

    # train_loader_single = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_sims, shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    init_loader  = DataLoader(init_dataset_, batch_size=1, shuffle=False)
    init_loader_ROM = DataLoader(init_ROM_dataset, batch_size=1, shuffle=False)
    # init_loader_ROM = DataLoader(init_ROM_dataset, batch_size=len(init_ROM_dataset), shuffle=False)
    train_snapshots = train_snapshots[N:]
    test_snapshots = test_snapshots[N:]
    return loader, train_loader, test_loader, \
            val_loader, scaler_all, scaler_test, xyz, VAR_all, vel_x_train, \
                train_snapshots, test_snapshots, train_loader_prev, train_loader_next, test_loader_prev, test_loader_next, init_loader, init_loader_ROM

def extract_snapshot(data_loader, batch_idx=0, averaging=None):
    counter = -1
    x = 0
    for batch in data_loader:
        counter += 1
        if averaging is None:
            if counter == batch_idx:
                return batch
        elif averaging[0] <= counter < averaging[1]:
            x += batch.x

    x /= averaging[1] - averaging[0]
    batch.x = x
    return batch

def get_data_from_idx(data, data_idx_str):
    train_data = []
    if data_idx_str == '':
        train_list=[]
    else:
        train_list = [int(item) for item in data_idx_str.split(',')]
        for train_idx in train_list:
            train_data.append(data['dataset_'+f'{train_idx}'])

    return train_data, train_list

def get_train_dataset_from_list(data_list, idx_list, HyperParams, truncate_mesh=None):
    Net_graph_loader = []
    # Net_train_trajectories = []
    # Net_test_trajectories = []
    Net_xyz = []
    Net_train_loader = []
    Net_test_loader = []
    Net_train_loader_prev = []
    Net_train_loader_next = []
    Net_test_loader_prev = []
    Net_test_loader_next = []
    Net_scaler = []
    for single_data, tr_idx in zip(data_list, idx_list):
        if HyperParams.if_diff_snap_range and (tr_idx in [8, 15, 16]): # in slow shedding cases, training range becomes 150~350
            tr_snap_range = [150,350]
        else:
            tr_snap_range = None

        graph_loader, train_loader, test_loader, \
            val_loader, scaler_all, scaler_test, xyz, VAR_all, vel_x_train, \
            train_trajectories, test_trajectories, train_loader_prev, train_loader_next, \
            test_loader_prev, test_loader_next, init_loader, _ = graphs_dataset(single_data, HyperParams, tr_snap_range=tr_snap_range, truncate_mesh=truncate_mesh)
        Net_graph_loader.append(graph_loader)
        Net_xyz.append(xyz)
        Net_train_loader.append(train_loader)
        Net_test_loader.append(test_loader)
        Net_train_loader_prev.append(train_loader_prev)
        Net_train_loader_next.append(train_loader_next)
        Net_test_loader_prev.append(test_loader_prev)
        Net_test_loader_next.append(test_loader_next)
        Net_scaler.append(scaler_all)
    return Net_graph_loader, Net_xyz, Net_train_loader, Net_test_loader, Net_train_loader_prev, Net_train_loader_next, \
        Net_test_loader_prev, Net_test_loader_next, Net_scaler

def get_scaler4mixed_data(vel_x_trains, HyperParams):
    scaling_type = HyperParams.scaling_type
    vel_x_train = torch.stack(vel_x_trains, dim=0)
    # scaler_all, VAR_all_temp = scaling.tensor_scaling(vel_x, scaling_type, HyperParams.scaler_number) # Original version
    mixed_scaler, _ = scaling.tensor_scaling(vel_x_train, scaling_type,
                                           HyperParams.scaler_number)  # Get scaler only from train data

    return mixed_scaler
    # VAR_all = torch.tensor(mixed_scaler.transform(vel_x)).t().unsqueeze(-1)
    # scaler_test, VAR_test = scaling.tensor_scaling(vel_x_test, scaling_type, HyperParams.scaler_number)

def get_test_dataset_from_list(test_data_list, idx_list, HyperParams, truncate_mesh=None):
    test_graph_loaders = []
    test_scaler_alls = []
    test_xyzs = []
    test_VAR_alls = []
    # init_ROM_loaders = []
    for y, tr_idx in zip(test_data_list, idx_list):
        if HyperParams.if_diff_snap_range and (tr_idx in [8, 15, 16]): # in slow shedding cases, training range becomes 150~350
            tr_snap_range = [150,350]
        else:
            tr_snap_range = None

        test_graph_loader, _, test_loader, \
            test_val_loader, test_scaler_all, _, test_xyz, test_VAR_all, _, \
            train_snapshots, test_snapshots, train_loader_prev, train_loader_next, test_loader_prev, test_loader_next, \
            init_loader, init_loader_ROM = graphs_dataset(y, HyperParams, tr_snap_range=tr_snap_range, truncate_mesh=truncate_mesh)

        if 'ROM' in HyperParams.testmode:
            test_graph_loaders.append(init_loader_ROM)
        elif 'auto' in HyperParams.testmode:
            test_graph_loaders.append(init_loader)
        elif 'train' in HyperParams.testmode or 'z' in HyperParams.testmode:
            test_graph_loaders.append(train_loader_prev)  # new
        elif 'test' in HyperParams.testmode:
            test_graph_loaders.append(test_loader_prev)  # new
        else:
            raise ValueError("Improper selection of -testmode")
        test_scaler_alls.append(test_scaler_all)  # origin
        test_xyzs.append(test_xyz)
        test_VAR_alls.append(test_VAR_all)
        # init_ROM_loaders.append(init_loader_ROM)
    if 'ROM' in HyperParams.testmode:
        snapshots = test_snapshots
    elif 'auto' in HyperParams.testmode:
        snapshots = test_snapshots
    elif 'train' in HyperParams.testmode or 'z' in HyperParams.testmode:
        snapshots = train_snapshots
    elif 'test' in HyperParams.testmode:
        snapshots = test_snapshots

    return test_graph_loaders, test_scaler_alls, test_xyzs, test_VAR_alls, snapshots #, init_ROM_loaders

# def get_closest_point(targets, xx, yy, vel_x):
#     pts_idx_list = []
#     for target in targets:
#         points = torch.stack((xx, yy), dim=1)
#         distances = torch.norm(points - target, dim=1)
#         min_dist_index = torch.argmin(distances)
#         pts_idx_list.append(min_dist_index)
#         closest_point = points[min_dist_index]
#         # snapshot = 0
#         # print("Total Node # / target node idx:", vel_x.shape[1], min_dist_index)
#         # print("Closest point:", closest_point)
#         # print("Distance to target:", distances[min_dist_index])
#         # print("x-velocity:", vel_x[snapshot, min_dist_index])
#     return