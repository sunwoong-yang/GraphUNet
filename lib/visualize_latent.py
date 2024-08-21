import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os

def visualize_z(HyperParams, test_z_predicted, idx, draw_trajectory=False, tSNE=False, pca=False):
	save_loc = HyperParams.net_dir + '/latent'
	if not os.path.exists(save_loc):
		os.makedirs(save_loc)

	if draw_trajectory:
		print("Trajectory plotting starts")
		comb_indices = list(combinations(range(len(test_z_predicted)), 2))
		# idx0, idx1 = 0, 1
		for comb in comb_indices:
			segment_size = 30
			offset = 0

			dx = [x2 - x1 for x1, x2 in zip(test_z_predicted[comb[0]][:-1], test_z_predicted[comb[0]][1:])]
			dy = [y2 - y1 for y1, y2 in zip(test_z_predicted[comb[1]][:-1], test_z_predicted[comb[1]][1:])]

			offset = 0
			indices = np.arange(test_z_predicted[comb[0]].shape[0])
			plt.scatter(test_z_predicted[comb[0]], test_z_predicted[comb[1]], c=indices, cmap='jet')

			for i in range(len(dx)):
				plt.quiver(test_z_predicted[comb[0]][i], test_z_predicted[comb[1]][i] + offset, dx[i], dy[i], angles='xy', \
				           scale_units='xy', scale=1, color='k', lw=.1)
				# if i % 30 == 0:
				# 	plt.axhline(y=(test_z_predicted[comb[1]][i] + offset).detach().numpy())
				# offset += 0.1

			# spearman_correlation, p_value = spearmanr(test_z_predicted[comb[0]], test_z_predicted[comb[1]])
			# print(f"Spearman btw {comb[0]} and {comb[1]}: {spearman_correlation:.3f}")
			# print(f"P-value: {p_value:.3f}")

			# for start_index in range(0, len(test_z_predicted[comb[0]]), segment_size):
			# 	end_index = start_index + segment_size
			# 	plt.plot(test_z_predicted[comb[0]][start_index:end_index],
			# 	         test_z_predicted[comb[1]][start_index:end_index] + offset)
			# 	offset += 1
			# plt.plot(test_z_predicted[comb[0]][:30], test_z_predicted[comb[1]][:30])
			# plt.plot(test_z_predicted[idx0], test_z_predicted[idx1])
			plt.savefig(save_loc + f'/test_idx{idx}_z{comb[0]}andz{comb[1]}.png')
			plt.close()
		print("Trajectory plotting ends")

	if tSNE or pca:
		print("t-SNE/PCA starts")
		def draw_reduced_z(x_reduced, DR_method="tSNE"):
			dx = [x2 - x1 for x1, x2 in zip(x_reduced[:, 0][:-1], x_reduced[:, 0][1:])]
			dy = [y2 - y1 for y1, y2 in zip(x_reduced[:, 1][:-1], x_reduced[:, 1][1:])]

			offset = 0
			for i in range(len(dx)):
				# plt.quiver(x_reduced[:, 0][i], x_reduced[:, 1][i] + offset, dx[i], dy[i], angles='xy',
				#            scale_units='xy', scale=1, color='r')
				if i < 30:
					plt.quiver(x_reduced[:, 0][i], x_reduced[:, 1][i] + offset, dx[i], dy[i], angles='xy',
					           scale_units='xy', scale=1, color='r')
				elif 30 <= i < 60:
					plt.quiver(x_reduced[:, 0][i], x_reduced[:, 1][i] + offset, dx[i], dy[i], angles='xy',
					           scale_units='xy', scale=1, color='k')
				elif 60 <= i < 90:
					plt.quiver(x_reduced[:, 0][i], x_reduced[:, 1][i] + offset, dx[i], dy[i], angles='xy',
					           scale_units='xy', scale=1, color='b')
				elif 90 <= i < 120:
					plt.quiver(x_reduced[:, 0][i], x_reduced[:, 1][i] + offset, dx[i], dy[i], angles='xy',
					           scale_units='xy', scale=1, color='g')
				else:
					plt.quiver(x_reduced[:, 0][i], x_reduced[:, 1][i] + offset, dx[i], dy[i], angles='xy',
					           scale_units='xy', scale=1, color='c')

			plt.savefig(save_loc + f'/{DR_method}_arrow_test_idx{idx}.png')
			plt.close()

			# plt.scatter(x_reduced[:, 0], x_reduced[:, 1])

			indices = np.arange(x_reduced.shape[0])
			plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=indices, cmap='jet')
			plt.plot(x_reduced[:, 0], x_reduced[:, 1], lw=1, color='k')
			plt.colorbar(label='Index')

			# num_groups = 5
			# points_per_group = 30
			#
			# # Define a list of colors for the groups
			# colors = ['red', 'blue', 'green', 'purple', 'orange']
			#
			# for i in range(num_groups):
			# 	start_index = i * points_per_group
			# 	end_index = start_index + points_per_group
			# 	plt.scatter(x_reduced[start_index:end_index, 0], x_reduced[start_index:end_index, 1], color=colors[i])

			plt.savefig(save_loc + f'/{DR_method}_scatter_test_idx{idx}.png')
			plt.close()

			indices = np.arange(x_reduced.shape[0])
			plt.scatter(x_reduced[:30, 0], x_reduced[:30, 1], c=indices[:30], cmap='jet')
			plt.plot(x_reduced[:30, 0], x_reduced[:30, 1], lw=1, color='k')
			plt.colorbar(label='Index')
			plt.savefig(save_loc + f'/test1.png')
			plt.close()

			indices = np.arange(x_reduced.shape[0])
			plt.scatter(x_reduced[30:60, 0], x_reduced[30:60, 1], c=indices[30:60], cmap='jet')
			plt.plot(x_reduced[30:60, 0], x_reduced[30:60, 1], lw=1, color='k')
			plt.colorbar(label='Index')
			plt.savefig(save_loc + f'/test2.png')
			plt.close()

		# print('tt', test_z_predicted.shape) 25 150 1
		x_full = np.array(test_z_predicted.squeeze().t())

		if tSNE:
			tsne = TSNE(n_components=2, random_state=0)
			x_reduced = tsne.fit_transform(x_full)
			draw_reduced_z(x_reduced, DR_method="tSNE")
		if pca:
			pca = PCA(n_components=2)
			x_reduced = pca.fit_transform(x_full)
			draw_reduced_z(x_reduced, DR_method="PCA")

		print("t-SNE/PCA ends")
