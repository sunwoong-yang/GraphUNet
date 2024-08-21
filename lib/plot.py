import numpy as np
import torch
import matplotlib.pyplot as plt
from lib import scaling, get_data_info
from collections import defaultdict
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from PIL import Image
import tensorflow as tf
import pandas as pd
import csv
import seaborn as sns

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


def plot_loss(HyperParams):
    """
    Plots the history of losses during the training of the autoencoder.

    Attributes:
    HyperParams (namedtuple): An object containing the parameters of the autoencoder.
    """

    history = np.load(HyperParams.net_dir+'history'+HyperParams.net_run+'.npy', allow_pickle=True).item()
    history_test = np.load(HyperParams.net_dir+'history_test'+HyperParams.net_run+'.npy', allow_pickle=True).item()
    ax = plt.figure().gca()
    ax.semilogy(history['l1'])
    ax.semilogy(history['l2'])
    ax.semilogy(history_test['l1'], '--')
    ax.semilogy(history_test['l2'], '--')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Loss over training epochs')
    plt.legend(['Autoencoder (train)', 'Map (train)', 'Autoencoder (test)', 'Map (test)'])
    plt.savefig(HyperParams.net_dir+'history_losses'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)


def plot_latent(HyperParams, latents, latents_estimation):
    """
    Plot the original and estimated latent spaces
    
    Parameters:
    HyperParams (obj): object containing the Autoencoder parameters 
    latents (tensor): tensor of original latent spaces
    latents_estimation (tensor): tensor of estimated latent spaces
    """

    plt.figure()
    for i1 in range(HyperParams.bottleneck_dim):
        plt.plot(latents[:,i1].detach(), '--')
        plt.plot(latents_estimation[:,i1].detach(),'-')
    plt.title('Evolution in the latent space')
    plt.ylabel('$u_N(\mu)$')
    plt.xlabel('Snaphots')
    plt.legend(['Autoencoder', 'Map'])
    plt.savefig(HyperParams.net_dir+'latents'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)
    green_diamond = dict(markerfacecolor='g', marker='D')
    _, ax = plt.subplots()
    ax.boxplot(latents_estimation.detach().numpy(), flierprops=green_diamond)
    plt.title('Variance in the latent space')
    plt.ylabel('$u_N(\mu)$')
    plt.xlabel('Bottleneck')
    plt.savefig(HyperParams.net_dir+'box_plot_latents'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)
    

def plot_error(res, VAR_all, scaler_all, HyperParams, mu_space, params, train_trajectories, vars, p1=0, p2=-1):
    """
    This function plots the relative error between the predicted and actual results.

    Parameters:
    res (ndarray): The predicted results
    VAR_all (ndarray): The actual results
    scaler_all (object): The scaler object used for scaling the results
    HyperParams (object): The HyperParams object holding the necessary hyperparameters
    mu1_range (ndarray): Range of the first input variable
    mu2_range (ndarray): Range of the second input variable
    params (ndarray): The input variables
    train_trajectories (ndarray): The indices of the training data
    vars (str): The name of the variable being plotted
    """

    u_hf = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
    u_app = scaling.inverse_scaling(res, scaler_all, HyperParams.scaling_type)
    error = np.linalg.norm(u_app - u_hf, axis=0) / np.linalg.norm(u_hf, axis=0)
    mu1_range = mu_space[p1]
    mu2_range = mu_space[p2]
    n_params = params.shape[1]
    tr_pt_1 = params[train_trajectories, p1]
    tr_pt_2 = params[train_trajectories, p2]
    if n_params > 2:
        rows, ind = np.unique(params[:, [p1, p2]], axis=0, return_inverse=True)
        indices_dict = defaultdict(list)
        [indices_dict[tuple(rows[i])].append(idx) for idx, i in enumerate(ind)]
        error = np.array([np.mean(error[indices]) for indices in indices_dict.values()])
        tr_pt = [i for i in indices_dict if any(idx in train_trajectories for idx in indices_dict[i])]
        tr_pt_1 = [t[0] for t in tr_pt]
        tr_pt_2 = [t[1] for t in tr_pt]
    X1, X2 = np.meshgrid(mu1_range, mu2_range, indexing='ij')
    output = np.reshape(error, (len(mu1_range), len(mu2_range)))
    fig = plt.figure('Relative Error '+vars)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, output, cmap=colormaps['coolwarm'], color='blue')
    ax.contour(X1, X2, output, zdir='z', offset=output.min(), cmap=colormaps['coolwarm'])
    ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]),
           ylim=tuple([mu2_range[0], mu2_range[-1]]),
           xlabel=f'$\mu_{str((p1%n_params)+1)}$',
           ylabel=f'$\mu_{str((p2%n_params)+1)}$',
           zlabel='$\\epsilon_{GCA}(\\mathbf{\mu})$')
    ax.plot(tr_pt_1, tr_pt_2, output.min()*np.ones(len(tr_pt_1)), '*r')
    ax.set_title('Relative Error '+vars)
    ax.zaxis.offsetText.set_visible(False)
    exponent_axis = np.floor(np.log10(max(ax.get_zticks()))).astype(int)
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
    ax.text2D(0.9, 0.82, "$\\times 10^{"+str(exponent_axis)+"}$", transform=ax.transAxes, fontsize="x-large")
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_'+vars+HyperParams.net_run+'.png', transparent=True, dpi=500)


def plot_error_2d(res, VAR_all, scaler_all, HyperParams, mu_space, params, train_trajectories, vars, p1=0, p2=-1):
    """
    This function plots the relative error between the predicted and actual results in 2D

    Parameters:
    res (ndarray): The predicted results
    VAR_all (ndarray): The actual results
    scaler_all (object): The scaler object used for scaling the results
    HyperParams (object): The HyperParams object holding the necessary hyperparameters
    mu1_range (ndarray): Range of the first input variable
    mu2_range (ndarray): Range of the second input variable
    params (ndarray): The input variables
    train_trajectories (ndarray): The indices of the training data
    vars (str): The name of the variable being plotted
    """

    u_hf = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
    u_app = scaling.inverse_scaling(res, scaler_all, HyperParams.scaling_type)
    error = np.linalg.norm(u_app - u_hf, axis=0) / np.linalg.norm(u_hf, axis=0)
    mu1_range = mu_space[p1]
    mu2_range = mu_space[p2]
    n_params = params.shape[1]
    tr_pt_1 = params[train_trajectories, p1]
    tr_pt_2 = params[train_trajectories, p2]
    X1, X2 = np.meshgrid(mu1_range, mu2_range, indexing='ij')
    output = np.reshape(error, (len(mu1_range), len(mu2_range)))
    fig = plt.figure('Relative Error 2D '+vars)
    ax = fig.add_subplot()
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    colors = output.flatten()
    sc = plt.scatter(X1.flatten(), X2.flatten(), s=(2e1*colors/output.max())**2, c=colors, cmap=colormaps['coolwarm'])
    plt.colorbar(sc, format=fmt)
    ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]),
           ylim=tuple([mu2_range[0], mu2_range[-1]]),
           xlabel=f'$\mu_{str((p1%n_params)+1)}$',
           ylabel=f'$\mu_{str((p2%n_params)+1)}$')
    ax.plot(tr_pt_1, tr_pt_2,'*r')
    ax.set_title('Relative Error 2D '+vars)
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_2d_'+vars+HyperParams.net_run+'.png', transparent=True, dpi=500)
    plt.show()


def plot_fields(SNAP, results, scaler_all, HyperParams, dataset, xyz, name_index='', draw_mesh=False):
    """
    Plots the field solution for a given snapshot.

    The function takes in the following inputs:

    SNAP: integer value indicating the snapshot to be plotted.
    results: array of shape (num_samples, num_features), representing the network's output.
    scaler_all: instance of the scaler used to scale the data.
    HyperParams: instance of the Autoencoder parameters class containing information about the network architecture and training.
    dataset: array of shape (num_samples, 3), representing the triangulation of the spatial domain.
    xyz: list of arrays of shape (num_samples, num_features), containing the x, y and z-coordinates of the domain.
    params: array of shape (num_features,), containing the parameters associated with each snapshot.
    The function generates a plot of the field solution and saves it to disk using the filepath specified in HyperParams.net_dir.
    """

    fig = plt.figure()
    predicted_qoi = scaling.inverse_scaling(results, scaler_all, HyperParams.scaling_type)
    xx = xyz[0]
    yy = xyz[1]
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if dataset["mesh_pos"].shape[2] == 2:
        triang = np.asarray(dataset['cells'][0])
        # triang = np.asarray(dataset.T - 1)
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])
        levels = np.linspace(0.0, 2.5, 101)
        cs = ax.tricontourf(xx, yy, triang, predicted_qoi[SNAP], 100, cmap=colormaps['coolwarm'], levels=levels, extend='both')
        # cs = ax.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, z_net, 100, cmap=colormaps['jet'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, format=fmt)
    elif dataset["mesh_pos"].shape[2] == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left", 
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        p = ax.scatter(xx, yy, zz, c=predicted_qoi[SNAP], cmap=colormaps['jet'], linewidth=0.5)
        # p = ax.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=z_net, cmap=colormaps['jet'], linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax, format=fmt)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)

    if draw_mesh:

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

        edges = triangles_to_edges(torch.tensor(np.array(dataset['cells'][0])))

        xx = torch.tensor(dataset["mesh_pos"][0, :, 0])
        yy = torch.tensor(dataset["mesh_pos"][0, :, 1])
        edge_index = torch.cat((torch.tensor(edges[0].numpy()).unsqueeze(0),
                                torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

        # fig, ax = plt.subplots()
        # ax.scatter(single_data['mesh_pos'][0, :, 0], single_data['mesh_pos'][0, :, 1], s=0.1, c='k')
        # ax.scatter(xx, yy, s=0.1, c='k')
        for i in range(edge_index.shape[1]):
            ax.plot((dataset['mesh_pos'][0, edge_index[0, i], 0], dataset['mesh_pos'][0, edge_index[1, i], 0]),
                    (dataset['mesh_pos'][0, edge_index[0, i], 1], dataset['mesh_pos'][0, edge_index[1, i], 1]),
                    color='k', lw=0.1, alpha=0.5)

    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')  
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    # ax.set_title('Solution field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    if not os.path.exists(HyperParams.net_dir+'/figures'):
        os.makedirs(HyperParams.net_dir+'/figures')
    plt.savefig(HyperParams.net_dir + '/figures/' + name_index + '_.png', bbox_inches='tight', dpi=500)
    plt.close()

    return predicted_qoi[SNAP]

def plot_error_fields(SNAP_pred, SNAP_real, results, VAR_all, scaler_all, HyperParams, dataset, xyz, name_index=''):
    """
    This function plots a contour map of the error field of a given solution of a scalar field.
    The error is computed as the absolute difference between the true solution and the predicted solution,
    normalized by the 2-norm of the true solution.

    Inputs:
    SNAP: int, snapshot of the solution to be plotted
    results: np.array, predicted solution
    VAR_all: np.array, true solution
    scaler_all: np.array, scaling information used in the prediction
    HyperParams: class, model architecture and training parameters
    dataset: np.array, mesh information
    xyz: list of arrays of shape (num_samples, num_features), containing the x, y and z-coordinates of the domain.
    params: np.array, model parameters
    """

    fig = plt.figure()
    gt_qoi = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
    predicted_qoi = scaling.inverse_scaling(results, scaler_all, HyperParams.scaling_type)
    # z = Z[:, SNAP]
    # z_net = Z_net[:, SNAP]
    # error = abs(z - z_net)/np.linalg.norm(z, 2)
    xx = xyz[0]
    yy = xyz[1]
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if dataset["mesh_pos"].shape[2] == 2:
        triang = np.asarray(dataset['cells'][0])
        # triang = np.asarray(dataset.T - 1)
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])
        levels = np.linspace(-0.3, 0.3, 101)
        error = predicted_qoi[SNAP_pred] - gt_qoi[SNAP_real]
        cs = ax.tricontourf(xx, yy, triang, error, 100, cmap=colormaps['coolwarm'], levels=levels, extend='both')
        # cs = ax.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, error, 100, cmap=colormaps['coolwarm'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, format=fmt)
    elif dataset["mesh_pos"].shape[2] == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left", 
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        error = predicted_qoi[SNAP_pred] - gt_qoi[SNAP_real]
        p = ax.scatter(xx, yy, zz, c=error, cmap=colormaps['coolwarm'], linewidth=0.5)
        # p = ax.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=error, cmap=colormaps['coolwarm'], linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax, format=fmt)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')  
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    # ax.set_title('Error field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    if not os.path.exists(HyperParams.net_dir + '/figures'):
        os.makedirs(HyperParams.net_dir + '/figures')

    error = torch.mean(torch.pow(predicted_qoi[SNAP_pred] - gt_qoi[SNAP_real], 2))

    plt.savefig(HyperParams.net_dir + '/figures/' + name_index + f'_MSE{error}'+'.png', bbox_inches='tight', dpi=500)
    plt.close()
    return error

def plot_gt_fields(SNAP, VAR_all, scaler_all, HyperParams, dataset, xyz, name_index='', noise_size=None):
    """
    This function plots a contour map of the error field of a given solution of a scalar field.
    The error is computed as the absolute difference between the true solution and the predicted solution,
    normalized by the 2-norm of the true solution.

    Inputs:
    SNAP: int, snapshot of the solution to be plotted
    results: np.array, predicted solution
    VAR_all: np.array, true solution
    scaler_all: np.array, scaling information used in the prediction
    HyperParams: class, model architecture and training parameters
    dataset: np.array, mesh information
    xyz: list of arrays of shape (num_samples, num_features), containing the x, y and z-coordinates of the domain.
    params: np.array, model parameters
    """

    fig = plt.figure()
    gt_qoi = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
    if noise_size is not None:
        gt_qoi += torch.normal(0, std=noise_size, size=gt_qoi.shape)
    xx = xyz[0]
    yy = xyz[1]
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if dataset["mesh_pos"].shape[2] == 2:
        triang = np.asarray(dataset['cells'][0])
        # triang = np.asarray(dataset.T - 1)
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])
        levels = np.linspace(0.0, 2.5, 101)
        cs = ax.tricontourf(xx, yy, triang, gt_qoi[SNAP], 100, cmap=colormaps['coolwarm'], levels=levels, extend='both')
        # cs = ax.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, error, 100, cmap=colormaps['coolwarm'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, format=fmt)
    elif dataset["mesh_pos"].shape[2] == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left",
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        p = ax.scatter(xx, yy, zz, c= gt_qoi[SNAP], cmap=colormaps['coolwarm'], linewidth=0.5)
        # p = ax.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=error, cmap=colormaps['coolwarm'], linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax, format=fmt)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    # ax.set_title('Error field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    if not os.path.exists(HyperParams.net_dir + '/figures'):
        os.makedirs(HyperParams.net_dir + '/figures')
    plt.savefig(HyperParams.net_dir + '/figures/' + name_index + '_.png', bbox_inches='tight', dpi=500)
    plt.close()

    return gt_qoi[SNAP]

def make_gif(dir, name_index, save_index=''):
    dir_fig =  dir + '/figures'
    file_paths = [file for file in os.listdir(dir_fig) if file.startswith(name_index) and file.endswith('.png')]
    # file_paths = sorted(file_paths, key=lambda x: int(x.replace(name_index + '_snap', "").replace(".png", "")))
    file_paths = sorted(file_paths, key=lambda x: int(x.split('_snap')[1].split('_')[0]))
    file_paths = [os.path.join(dir_fig, file) for file in file_paths]

    if not file_paths:
        raise ValueError("No PNG files found in the specified directory.")

    # images = [Image.open(file) for file in file_paths]
    # base_image = images[0]
    # # base_image.convert('P', palette=Image.ADAPTIVE, colors=256).save(dir + f'/test.png', quality=95, subsampling=0)
    # converted_images = [base_image.convert('P', palette=Image.ADAPTIVE, colors=256)]
    # # converted_images = [base_image]
    # for img in images[1:]:
    #     converted_images.append(
    #         img.convert('P', palette=Image.ADAPTIVE, colors=256).resize(base_image.size, Image.ANTIALIAS))
    #         # img.resize(base_image.size, Image.ANTIALIAS))
    #
    # converted_images[0].save(dir + f'/[Gif]{file_name}.gif', save_all=True, append_images=images[1:], optimize=False, duration=60,
    #                loop=0, quality=95)

    images = [Image.open(file).convert('P', palette=Image.ADAPTIVE, colors=256) for file in file_paths]
    # Convert images to RGB mode and save as GIF
    dir_vid = dir + '/videos'
    if not os.path.exists(dir_vid):
        os.makedirs(dir_vid)
    images[0].save(dir_vid + f'/[Gif]{name_index}{save_index}.gif', save_all=True, append_images=images[1:], optimize=False, duration=60, loop=0)

def plot_mesh(data_list, data_idx_list, HyperParams, name_index='', truncate_mesh=None):

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

    for single_data, data_idx in zip(data_list, data_idx_list):
        # 어차피 time step마다 mesh변화 없으므로 ts=0 상황에서 edge 꺼내옴
        edges = triangles_to_edges(torch.tensor(np.array(single_data['cells'][0])))
        if truncate_mesh is not None:
            # sliced_parts = np.concatenate((np.array(range(0, 29)), np.array(range(57, 500))))
            xx = torch.tensor(single_data["mesh_pos"][0, :, 0])
            yy = torch.tensor(single_data["mesh_pos"][0, :, 1])
            sliced_parts = np.where(xx > truncate_mesh)[0]
            nodes_mask = np.zeros(len(xx), dtype=bool)
            nodes_mask[sliced_parts] = True
            mask = ~np.isin(edges[0].numpy(), sliced_parts) & ~np.isin(edges[1].numpy(), sliced_parts)
            edges_0 = torch.tensor(edges[0].numpy())[mask]
            edges_1 = torch.tensor(edges[1].numpy())[mask]

            edge_index = torch.cat((edges_0.unsqueeze(0), edges_1.unsqueeze(0)), dim=0).type(torch.long)
            xx = xx[~nodes_mask]
            yy = yy[~nodes_mask]
        else:
            xx = torch.tensor(single_data["mesh_pos"][0, :, 0])
            yy = torch.tensor(single_data["mesh_pos"][0, :, 1])
            edge_index = torch.cat((torch.tensor(edges[0].numpy()).unsqueeze(0),
                                    torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

        fig, ax = plt.subplots()
        # ax.scatter(single_data['mesh_pos'][0, :, 0], single_data['mesh_pos'][0, :, 1], s=0.1, c='k')
        ax.scatter(xx, yy, s=0.1, c='k')
        for i in range(edge_index.shape[1]):
            ax.plot((single_data['mesh_pos'][0, edge_index[0, i], 0], single_data['mesh_pos'][0, edge_index[1, i], 0]),
                     (single_data['mesh_pos'][0, edge_index[0, i], 1], single_data['mesh_pos'][0, edge_index[1, i], 1]),
                     color='k', lw=0.5)
        plt.tight_layout()
        plt.xlim(0, 1.6)
        plt.ylim(0, 0.4)
        ax.set_aspect('equal', 'box')
        # ax.set_title('Error field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
        if not os.path.exists(HyperParams.net_dir + '/figures'):
            os.makedirs(HyperParams.net_dir + '/figures')
        plt.savefig(HyperParams.net_dir + '/figures/' + f'Mesh_idx{data_idx}' + name_index + '.png', bbox_inches='tight', dpi=500)
        plt.close()

def postprocess_all(HyperParams, testlist, test_data_list, test_x_predicteds, test_scaler_alls, test_VAR_alls, test_xyzs, test_snapshots, repeat_idx, if_prove_pts=False, draw_mesh=False):
    err_all_idx = []

    indices = ['']
    mean_errors = [f"Mean Error in test mode '{HyperParams.testmode}'"]
    std_errors = [f"STD Error in test mode '{HyperParams.testmode}'"]

    pred_prove_pts_list, gt_prove_pts_list = [], []
    for idx in range(len(test_data_list)):
        err_list = []
        pred_prove_pts, gt_prove_pts = [], []
        for SNAP in range(test_x_predicteds[idx].shape[1]):
            pred_prove_three_pts_element = plot_fields(SNAP, test_x_predicteds[idx], test_scaler_alls[idx], HyperParams, test_data_list[idx], \
                        test_xyzs[idx], name_index=f'Pred{repeat_idx}_test_idx{testlist[idx]}_snap{SNAP + test_snapshots[0]}', draw_mesh=draw_mesh)
            err_list.append(
                plot_error_fields(SNAP, SNAP + test_snapshots[0], test_x_predicteds[idx], test_VAR_alls[idx],
                                       test_scaler_alls[idx], \
                                       HyperParams, test_data_list[idx], test_xyzs[idx],
                                       name_index=f'Err{repeat_idx}_test_idx{testlist[idx]}_snap{SNAP + test_snapshots[0]}'))
            gt_prove_three_pts_element = plot_gt_fields(SNAP + test_snapshots[0], test_VAR_alls[idx], test_scaler_alls[idx], \
                                HyperParams, test_data_list[idx], test_xyzs[idx],
                                name_index=f'GT{repeat_idx}_test_idx{testlist[idx]}_snap{SNAP + test_snapshots[0]}')
            if if_prove_pts:
                if testlist[idx] in [15,16]:
                    # prove_idx_list = get_data_info.get_prove_pts_idx(test_data_list[idx], x=0.8)
                    prove_idx_list = get_data_info.get_prove_pts_idx(test_data_list[idx], x=0.6)
                else:
                    prove_idx_list = get_data_info.get_prove_pts_idx(test_data_list[idx], x=0.6)
                pred_prove_pts.append([pred_prove_three_pts_element[i] for i in prove_idx_list])
                gt_prove_pts.append([gt_prove_three_pts_element[i] for i in prove_idx_list])

            if (SNAP == 0) and (HyperParams.noise_type != 0):
                plot_gt_fields(SNAP + test_snapshots[0], test_VAR_alls[idx], test_scaler_alls[idx], \
                               HyperParams, test_data_list[idx], test_xyzs[idx],
                               name_index=f'Noised_GT{repeat_idx}_test_idx{testlist[idx]}_snap{SNAP + test_snapshots[0]}', noise_size=HyperParams.noise_size)
        if if_prove_pts:
            pred_prove_pts = torch.tensor(pred_prove_pts).reshape(-1, len(prove_idx_list))
            pred_prove_pts_list.append(pred_prove_pts)
            gt_prove_pts = torch.tensor(gt_prove_pts).reshape(-1, len(prove_idx_list))
            gt_prove_pts_list.append(gt_prove_pts)

        err_list = torch.tensor(err_list)
        err_all_idx.append(err_list)
        err4idx = torch.mean(err_list).item()
        err4idx_std = torch.std(err_list).item()
        idx4err = testlist[idx]

        indices.append(f'Idx {idx4err}')
        mean_errors.append(err4idx)
        std_errors.append(err4idx_std)

    plot_three_prove_pts(gt_prove_pts_list, pred_prove_pts_list, HyperParams, testlist, repeat_idx)

    with open(HyperParams.net_dir + f'/{HyperParams.custom_name}_Err_summary{repeat_idx}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(indices)
        csvwriter.writerow(mean_errors)
        csvwriter.writerow(std_errors)


    snap_range = range(test_snapshots[0], test_x_predicteds[idx].shape[1] + test_snapshots[0])
    df = pd.DataFrame({"Snap": snap_range})
    for i, tensor in zip(testlist, err_all_idx):
        # Convert tensor to numpy array and ensure it's flattened in case it's not 1D
        column_data = tensor.numpy().flatten()
        df[f'idx_{i}'] = column_data[:len(snap_range)]

    excel_path = HyperParams.net_dir + f'/MSE_pred{repeat_idx}.xlsx'  # Define your desired output path
    df.to_excel(excel_path, index=False, engine='openpyxl')

    for idx in range(len(test_data_list)):
        if 'auto' in HyperParams.testmode:
            auto_reg_info = f'auto{HyperParams.testmode[1]}'
        else:
            auto_reg_info = ''
        make_gif(dir=HyperParams.net_dir, name_index=f"Pred{repeat_idx}_test_idx{testlist[idx]}_", save_index=auto_reg_info)
        make_gif(dir=HyperParams.net_dir, name_index=f"Err{repeat_idx}_test_idx{testlist[idx]}_", save_index=auto_reg_info)
        make_gif(dir=HyperParams.net_dir, name_index=f"GT{repeat_idx}_test_idx{testlist[idx]}_", save_index=auto_reg_info)

def compute_average(HyperParams):
    means, stds = [], []
    for repeat_idx in range(HyperParams.n_repeat):
        file_path = HyperParams.net_dir + f'/{HyperParams.custom_name}_Err_summary{repeat_idx}.csv'
        df = pd.read_csv(file_path)
        if repeat_idx == 0:  # Only capture column names once
            headers = df.columns.tolist()
        mean = df.iloc[0, 1:]
        std = df.iloc[1, 1:]
        means.append(mean)
        stds.append(std)

    means_avg = np.mean(np.array(means), axis=0).tolist()
    stds_avg = np.mean(np.array(stds), axis=0).tolist()

    means_avg.insert(0, 'Mean')
    stds_avg.insert(0, 'STD')
    with open(HyperParams.net_dir + f'/[Avg]{HyperParams.custom_name}_Err_summary.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerow(means_avg)
        csvwriter.writerow(stds_avg)
        csvwriter.writerow(["Mean of each model"])
        for idx, mean_ in enumerate(means):
            mean_ = mean_.tolist()
            csvwriter.writerow([f'Model{idx}'] + mean_)
        csvwriter.writerow(["STD of each model"])
        for idx, std_ in enumerate(stds):
            std_ = std_.tolist()
            csvwriter.writerow([f'Model{idx}'] + std_)

    import re

    # Define the regex pattern to extract the training time until saved
    pattern = r"Training time \(until saved\) \[s\]: ([\d.]+)"

    # List of file names
    file_names = [HyperParams.net_dir+f"/summary{i}.txt" for i in range(HyperParams.n_repeat)]

    # Initialize variables to store the cumulative time and individual times
    cumulative_time = 0
    individual_times = []

    # Loop through the file names and extract the times
    for file_name in file_names:
        with open(file_name, 'r') as file:
            contents = file.read()
            # Use regex to find the specific line containing the time
            match = re.search(pattern, contents)
            if match:
                # Convert the matched string to a float and add to the list
                training_time = float(match.group(1))
                individual_times.append(training_time)
                cumulative_time += training_time

    # Calculate the average time
    average_time = cumulative_time / len(individual_times)

    # Write the new summary file
    with open(HyperParams.net_dir+'/[Avg]time_summary.txt', 'w') as file:
        file.write(f"Average Training time (until saved) [s]: {average_time}\n")
        for idx, time in enumerate(individual_times):
            file.write(f"Model{idx} time (until saved) [s]: {time}\n")


def plot_three_prove_pts(gt_prove_pts_list, pred_prove_pts_list, HyperParams, testlist, repeat_idx):
    palette = sns.color_palette("pastel")
    for idx_data, gt_prove_pts, pred_prove_pts in zip(testlist, gt_prove_pts_list, pred_prove_pts_list):
        fig, ax = plt.subplots()
        # colors = ['r','g','b']
        # ax.scatter(single_data['mesh_pos'][0, :, 0], single_data['mesh_pos'][0, :, 1], s=0.1, c='k')
        for prove_idx in range(gt_prove_pts.shape[1]):
            ax.plot(range(gt_prove_pts.shape[0]), gt_prove_pts[:,prove_idx], color=palette[prove_idx], lw=2, alpha=1., label=f'GT (point{prove_idx+1})', zorder=0)
        for prove_idx in range(pred_prove_pts.shape[1]):
            # ax.plot(range(pred_prove_pts.shape[0]), pred_prove_pts[:, prove_idx], color=palette[prove_idx], lw=1.5, ls='--', label=f'Prediction (point{prove_idx+1})')
            ax.scatter(range(pred_prove_pts.shape[0]), pred_prove_pts[:, prove_idx], color='k', edgecolor=palette[prove_idx], s=14, label=f'Prediction (point{prove_idx+1})', zorder=1)
        plt.tight_layout()
        plt.xlim(0, 200)
        # plt.ylim(-0.5, 2.5)
        # ax.set_aspect('equal', 'box')
        # ax.set_title('Error field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
        plt.legend(fontsize=17.5, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1+0.1*gt_prove_pts.shape[1]), columnspacing=0.8)
        plt.xlabel('Rollout step after trained snapshots', fontsize=20)
        plt.ylabel('x-velocity [m/s]', fontsize=20)
        if not os.path.exists(HyperParams.net_dir + '/figures'):
            os.makedirs(HyperParams.net_dir + '/figures')
        plt.savefig(HyperParams.net_dir + '/figures/' + f'Three_prove_idx{idx_data}_model{repeat_idx}' + '.png', bbox_inches='tight',
                    dpi=500)
        plt.close()