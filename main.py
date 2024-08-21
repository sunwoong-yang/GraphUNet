import torch
from lib import train_UNet, train_LSTM, test_UNet, test_LSTM, preprocessing, plot, hyperparameter, get_mode, visualize_latent, parser, get_data_info
from models import network
import h5py


args = parser.parsing()
device = ('cuda' if torch.cuda.is_available() else "cpu")
torch.set_num_threads(12)

if __name__ == "__main__":

    # Load dataset
    datafile = "./data/Deepmind_valid.h5"
    data = h5py.File(datafile, 'r')

    train_data_list, trainlist = preprocessing.get_data_from_idx(data, args.trainlist)
    test_data_list, testlist = preprocessing.get_data_from_idx(data, args.testlist)
    print("#"*10 + "  Train dataset info  " + "#"*10)
    for single_data in train_data_list:
        for item in single_data.keys():
            print('{} : {}'.format(item, single_data[item].shape))

    if_train = False
    if_test = False
    args.train = args.train.lower() in ['t', 'true']
    if trainlist != [] and testlist == [] and args.train:
        if_train = True
    elif trainlist != [] and testlist != [] and args.train:
        if_train = True
        if_test = True
    elif testlist != [] and not args.train:
        if_test = True
    else:
        raise ValueError("Select properly whether to train or test using -te or -tr or -train")

    HyperParams = hyperparameter.HyperParams()
    HyperParams.update(

        custom_name = args.name,  # None
        # beta = args.beta,
        # train_idx = [150, 300],
        train_idx = args.train_idx,
        train_list = trainlist,
        history = args.history,
        future = args.future,
        lstm_history = args.lstm_history,
        Enc_HC = args.Enc_HC, # original
        # Enc_HC=[40, 20, 10, 5, 1], # original
        # Enc_HC = [40, 20, 10, 5, 3, 1],
        # Dec_HC = [1, 3, 5, 3, 1], # original
        Dec_HC = args.Dec_HC,
        # Dec_HC=[1, 1, 1, 1, 1, 1],
        pooling = args.pooling,
        # pooling = [1000, 500, 250, 100], # original
        # pooling=[1000, 750, 500, 250, 100],
        # mlp_layer = args.mlp_layer, # original
        # mlp_layer = [75, 50],
        # minibatch = 25,
        minibatch = args.minibatch,
        max_epochs = args.epoch,  # 3000
        stop_tolerance = 50000,
        learning_rate = args.lr,  # 1e-3
        lr_step = 1000,
        lr_gamma = 0.8,
        scaling_type = args.scaler,
        # skip = args.skip.lower() in ['t', 'true'],
        UNet = args.unet,
        Norm = args.norm,
        testmode = args.testmode,
        gcn_type = args.gcn_type,
        gcn_k = args.gcn_k,
        zero_unpooling = args.zero_unpooling,
        my_pooling = args.my_pooling,
        augmentation_order = args.augmentation_order,
        noise_type = args.noise_type,
        noise_size = args.noise_size,
        n_repeat = args.n_repeat,
        if_diff_snap_range = args.snap,
        trans = args.trans,
        # testmode = [str(item) for item in args.testmode.split(',')],

    )
    # HyperParams.set_reproducibility()
    plot.plot_mesh(train_data_list, trainlist, HyperParams, name_index="_tr", truncate_mesh=HyperParams.trans)
    # plot.plot_mesh(test_data_list, testlist, HyperParams, name_index="_te", truncate_mesh=None)

    for repeat in range(HyperParams.n_repeat):
        HyperParams.set_reproducibility()
        Net_graph_loader, Net_train_xyz, Net_train_loader, Net_test_loader, Net_train_loader_prev, Net_train_loader_next, Net_test_loader_prev, Net_test_loader_next, \
            Net_scaler = preprocessing.get_train_dataset_from_list(train_data_list, trainlist, HyperParams, truncate_mesh=HyperParams.trans)

        model = network.Net(HyperParams, device)
        # model = model.to(device).double()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        optimizer = torch.optim.Adam(model.parameters(), lr=HyperParams.learning_rate,
                                     weight_decay=HyperParams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=HyperParams.lr_step, gamma=HyperParams.lr_gamma)

        if if_train:
            print('Training network')
            # training_gca.train(model, optimizer, device, scheduler, train_loader_prev, test_loader, train_trajectories,
            train_UNet.train(model, optimizer, device, scheduler, HyperParams, Net_train_loader_prev, Net_test_loader_prev, \
                               Net_train_loader_next=Net_train_loader_next, Net_test_loader_next=Net_test_loader_next, repeat_idx=str(repeat))

            if not HyperParams.future: # AE-LSTM (ROM)
                model.to("cpu")

                # temp_train_loader, temp_train_label_loader, temp_scaler = model.make_temp_dataset(Net_train_loader)
                # temp_test_loader, temp_test_label_loader, _ = model.make_temp_dataset(Net_test_loader, scaler=temp_scaler)
                temp_train_loader, temp_train_label_loader, temp_scaler = model.make_temp_dataset(Net_train_loader_prev)
                temp_test_loader, temp_test_label_loader, _ = model.make_temp_dataset(Net_test_loader_prev, scaler=temp_scaler)

                LSTM = train_LSTM.train(device, HyperParams, temp_train_loader, temp_train_label_loader, temp_test_loader,
                                        temp_test_label_loader)

        if if_test:


            print(f'Loading network (auto-reg type: {HyperParams.testmode})')
            model.load_state_dict(torch.load(HyperParams.net_dir + f'/model{str(repeat)}.pt', map_location=torch.device('cpu')))
            model.to("cpu")

            print(f'Plotting pooled nodes...')
            test_UNet.plot_pooled_nodes(model, Net_train_loader_prev, Net_train_loader_next, repeat_idx=repeat)

            test_graph_loaders, test_scaler_alls, test_xyzs, test_VAR_alls, test_snapshots = \
                preprocessing.get_test_dataset_from_list(test_data_list, testlist, HyperParams, truncate_mesh=None)

            test_x_predicteds = []
            test_z_predicteds = []
            # prove_three_pts_list = []
            if 'auto' in HyperParams.testmode:
                for idx in range(len(test_graph_loaders)):
                    init_loader = test_graph_loaders[idx]
                    test_x_predicteds_temp = []
                    test_z_predicteds_temp = []

                    for i in range(int(HyperParams.testmode[1])):
                        test_x_predicted, test_z_predicted, init_loader = test_UNet.evaluate_auto(model, init_loader)

                        test_x_predicted = test_x_predicted.reshape(-1, 1)
                        test_z_predicted = test_z_predicted.reshape(-1, 1)

                        test_x_predicteds_temp.append(test_x_predicted)
                        test_z_predicteds_temp.append(test_z_predicted)

                    test_x_predicteds.append(torch.stack(test_x_predicteds_temp, dim=1))
                    test_z_predicteds.append(torch.stack(test_z_predicteds_temp, dim=1))

            elif 'train' in HyperParams.testmode or 'test' in HyperParams.testmode:
                for idx, test_graph_loader in enumerate(test_graph_loaders):
                    test_x_predicteds_temp = []
                    test_z_predicteds_temp = []
                    # data = test_graph_loaders[idx]
                    test_x_predicted, test_z_predicted = test_UNet.evaluate(model, test_graph_loader, HyperParams) # Origin

                    # visualize_latent.visualize_z(HyperParams, test_z_predicted, idx, draw_trajectory=True, tSNE=True, pca=True)
                    visualize_latent.visualize_z(HyperParams, test_z_predicted, idx, draw_trajectory=False, tSNE=True, pca=True)

                    test_x_predicteds.append(test_x_predicted)
                    test_z_predicteds.append(test_z_predicted)

            elif 'ROM' in HyperParams.testmode:
                temp_train_loader, temp_train_label_loader, temp_scaler = model.make_temp_dataset(Net_train_loader)
                LSTM = torch.load(HyperParams.net_dir + '/model_temp.pt', map_location=torch.device('cpu'))
                LSTM.to("cpu")
                for idx in range(len(test_graph_loaders)):
                    test_x_predicteds_temp = []
                    init_loader = test_graph_loaders[idx]
                    for i in range(100):
                        init_loader, x_next = test_LSTM.evaluate_ROM(model, LSTM, init_loader, temp_scaler)
                        test_x_predicteds_temp.append(x_next)
                    test_x_predicteds.append(torch.stack(test_x_predicteds_temp, dim=1))

            elif 'z' in HyperParams.testmode:
                for idx, test_graph_loader in zip(testlist, test_graph_loaders):
                    test_x_predicted, test_z_predicted = test_UNet.evaluate(model, test_graph_loader, HyperParams)  # Origin
                    visualize_latent.visualize_z(HyperParams, test_z_predicted, idx, draw_trajectory=True, tSNE=True, pca=True)
            else:
                raise ValueError("HyperParams.testmode(-mode) is improper: select auto/train/test/ROM")


            if not HyperParams.testmode == "z":

                plot.postprocess_all(HyperParams, testlist, test_data_list, test_x_predicteds, test_scaler_alls, test_VAR_alls,
                                # test_xyzs, test_snapshots, repeat_idx=str(repeat), if_prove_pts=True, draw_mesh=False)
                                     test_xyzs, test_snapshots, repeat_idx=str(repeat), if_prove_pts=True, draw_mesh=True)
    plot.compute_average(HyperParams)
