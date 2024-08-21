import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import os
import pandas as pd
import time

# def train(model, optimizer, device, scheduler, params, train_loader, test_loader, train_trajectories, test_trajectories, HyperParams):
def train(model, optimizer, device, scheduler, HyperParams, Net_train_loader, Net_test_loader, Net_train_loader_next=None, Net_test_loader_next=None, repeat_idx=''):

    if not os.path.exists(HyperParams.net_dir):
        os.makedirs(HyperParams.net_dir)
    # options_text = "\n".join(HyperParams.option_list)
    options_file_path = os.path.join(HyperParams.net_dir, "Case_options.txt")
    # options_text = HyperParams.option_list.split("_")
    with open(options_file_path, "w") as file:
        for attr, value in vars(HyperParams).items():
            file.write(f"{attr} = {value}\n")
        # for element in options_text:
        #     file.write(element + '\n')

    train_history = dict(epoch=[], train_total_loss=[], train_mse=[])
    test_history = dict(epoch=[], test_total_loss=[], test_mse=[])
    min_train_loss = min_test_loss = np.Inf
    no_improve_epoch = 0

    model.train()
    loop = tqdm(range(HyperParams.max_epochs))
    start_time = time.time()
    for epoch in loop:

        total_examples = sum_loss = sum_mse = sum_kl = 0

        if Net_train_loader_next is not None:
            optimizer.zero_grad()
            for idx in range(len(Net_train_loader)):
                for data, data_next in zip(Net_train_loader[idx], Net_train_loader_next[idx]):
                    # optimizer.zero_grad()
                    # data.x = torch.tensor(data.x, dtype=torch.float64)
                    data = data.to(device)
                    data_next = data_next.to(device)
                    # out, z, z_estimation = model(data, params[train_trajectories, :])

                    if HyperParams.noise_type == 1:
                        with torch.random.fork_rng():
                            torch.manual_seed(int(repeat_idx))
                            data.x += torch.normal(0, std=HyperParams.noise_size, size=data.x.shape).to(device)
                    elif HyperParams.noise_type == 2:
                        with torch.random.fork_rng():
                            torch.manual_seed(int(repeat_idx))
                            noise_inp = torch.normal(0, std=HyperParams.noise_size, size=data.x.shape).to(device)
                            noise_out = torch.normal(0, std=HyperParams.noise_size, size=data_next.x.shape).to(device)
                        data.x += noise_inp
                        data_next.x += noise_out

                    out, z = model(data)
                    mse = F.mse_loss(out, data_next.x, reduction='mean')
                    if idx == 0:
                        loss_train = 1 * mse # for idx4 weighting
                    else:
                        loss_train = mse  # for idx8 weighting

                    loss_train.backward()
                    # optimizer.step()
                    sum_loss += loss_train
                    sum_mse += mse
                    # total_examples += 1
                    total_examples += len(data)
                    # optimizer.step() # gpu memory때문에 connectivity aug를 못해서
            optimizer.step()
        else:
            optimizer.zero_grad()
            for idx in range(len(Net_train_loader)):
                for data in Net_train_loader[idx]:
                    # optimizer.zero_grad()
                    # data.x = torch.tensor(data.x, dtype=torch.float64)
                    data = data.to(device)
                    # out, z, z_estimation = model(data, params[train_trajectories, :])

                    if HyperParams.noise_type == 1:
                        with torch.random.fork_rng():
                            torch.manual_seed(int(repeat_idx))
                            data.x += torch.normal(0, std=HyperParams.noise_size, size=data.x.shape).to(device)
                    elif HyperParams.noise_type == 2:
                        with torch.random.fork_rng():
                            torch.manual_seed(int(repeat_idx))
                            noise = torch.normal(0, std=HyperParams.noise_size, size=data.x.shape).to(device)
                        data.x += noise
                        data_next.x += noise

                    out, z = model(data)
                    mse = F.mse_loss(out, data_next.x, reduction='mean')
                    loss_train = mse

                    loss_train.backward()
                    # optimizer.step()

                    sum_loss += loss_train
                    sum_mse += mse
                    # total_examples += 1
                    total_examples += len(data)
                    # optimizer.step() # gpu memory때문에 connectivity aug를 못해서
            optimizer.step()
        scheduler.step()

        train_loss = sum_loss.detach().cpu().numpy() / total_examples
        train_mse = sum_mse.detach().cpu().numpy() / total_examples
        # train_kl = sum_kl.detach().cpu().numpy() / total_examples

        train_history['epoch'].append(epoch)
        train_history['train_total_loss'].append(train_loss)
        train_history['train_mse'].append(train_mse)
        # train_history['train_KL'].append(train_kl)


        with torch.no_grad():
            model.eval()
            total_examples = sum_loss = sum_mse = 0

            if Net_test_loader_next is not None:
                for idx in range(len(Net_test_loader)):
                    for data, data_next in zip(Net_test_loader[idx], Net_test_loader_next[idx]):
                        data = data.to(device)
                        data_next = data_next.to(device)
                        # out, z, z_estimation = model(data, params[test_trajectories, :])

                        out, z = model(data)
                        mse_loss = F.mse_loss(out, data_next.x, reduction='mean')
                        loss_test = mse_loss

                        sum_loss += loss_test
                        sum_mse += mse_loss
                        # total_examples += 1
                        total_examples += len(data)
            else:
                for idx in range(len(Net_test_loader)):
                    for data in Net_test_loader[idx]:
                        data = data.to(device)
                        # out, z, z_estimation = model(data, params[test_trajectories, :])

                        out, z = model(data)
                        mse_loss = F.mse_loss(out, data.x, reduction='mean')
                        loss_test = mse_loss

                        sum_loss += loss_test
                        sum_mse += mse_loss
                        # total_examples += 1
                        total_examples += len(data)

            test_loss = sum_loss.detach().cpu().numpy() / total_examples
            test_mse = sum_mse.detach().cpu().numpy() / total_examples
            # test_kl = sum_kl.detach().cpu().numpy() / total_examples
            test_history['epoch'].append(epoch)
            test_history['test_total_loss'].append(test_loss)
            test_history['test_mse'].append(test_mse)
            # test_history['test_KL'].append(test_kl)
        # print("Epoch[{}/{}, train_mse loss:{}, test_mse loss:{}".format(epoch + 1, HyperParams.max_epochs, train_history['train'][-1], test_history['test'][-1]))
        loop.set_postfix({"Tr MSE": train_history['train_mse'][-1], "Te MSE": test_history['test_mse'][-1]})


        if train_loss < min_train_loss:
            min_train_loss = train_loss
            min_test_loss = test_loss
            no_improve_epoch = 0
            # torch.save(model.state_dict(), HyperParams.net_dir + f'/model{repeat_idx}.pt')
            model_best = model
            saved_epoch = epoch
            training_time_saved = time.time() - start_time
        else:
            no_improve_epoch += 1

        if no_improve_epoch >= HyperParams.stop_tolerance:
            print(f'Early stop at epoch {epoch}!')
            # torch.save(model.state_dict(), HyperParams.net_dir + '/model.pt')
            break

        # torch.save(model.state_dict(), HyperParams.net_dir + '/model.pt')
    # print('aaaa',len(train_history['train_mse']))
    # for i in range(len(train_history['train_mse'])):
    #     train_history['saved_epoch'].append(saved_epoch)

    # with torch.no_grad():
    #     model_best.eval()
    #
    #     if Net_train_loader_next is not None:
    #         for idx in range(len(Net_train_loader)):
    #             for data, data_next in zip(Net_train_loader[idx], Net_train_loader_next[idx]):
    #                 data = data.to(device)
    #                 _, _ = model_best.encoder(data, if_plot_mesh=repeat_idx)
    #                 break
    #             break
    #
    #     else:
    #         for idx in range(len(Net_train_loader)):
    #             for data in Net_train_loader[idx]:
    #                 data = data.to(device)
    #                 _, _ = model_best.encoder(data, if_plot_mesh=repeat_idx)
    #                 break
    #             break

    torch.save(model_best.state_dict(), HyperParams.net_dir + f'/model{repeat_idx}.pt')
    np.save(HyperParams.net_dir+f'/history_train{repeat_idx}.npy', train_history)
    np.save(HyperParams.net_dir+f'/history_test{repeat_idx}.npy', test_history)

    training_time = time.time() - start_time
    with open(HyperParams.net_dir+f'/summary{repeat_idx}.txt', 'w') as file:
        file.write(f"Training time (total) [s]: {training_time}\n")
        file.write(f"Training time (until saved) [s]: {training_time_saved}\n")
        file.write(f"Saved epoch: {saved_epoch}\n")
        file.write(f"Train loss when saved: {min_train_loss}\n")
        file.write(f"Test loss when saved: {min_test_loss}\n")

    df_train = pd.DataFrame(train_history)
    df_test = pd.DataFrame(test_history)
    df_train.to_excel(HyperParams.net_dir+f'/train_history{repeat_idx}_epoch{saved_epoch}.xlsx', index=False)
    df_test.to_excel(HyperParams.net_dir+f'/test_history{repeat_idx}.xlsx', index=False)
    # with pd.ExcelWriter(f'{HyperParams.net_dir}/loss_history.xlsx', engine='xlsxwriter') as writer:
    #     df_train.to_excel(writer, sheet_name='Train History', index=False)
    #     df_test.to_excel(writer, sheet_name='Test History', index=False)
    print("\nEpoch for the saved network: ", saved_epoch)
    # model.load_state_dict(torch.load(HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'.pt', map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load(HyperParams.net_dir + '/model.pt', map_location=torch.device('cpu')))

