from pydmd import DMD
import numpy as np
from sklearn.decomposition import PCA

# Assuming X is your data array of shape (M, N)
# M = number of nodes per snapshot, N = number of snapshots
X = np.random.rand(M, N)  # Replace with your actual data

def DMD(X):
	dmd = DMD(svd_rank=2)
	dmd.fit(X)

	# Access the DMD modes and eigenvalues
	modes = dmd.modes
	eigenvalues = dmd.eigs

	# To reconstruct the data using the obtained modes
	reconstructed_data = dmd.reconstructed_data

	return dmd


def PCA(X):
	pca = PCA(n_components=None)

	# Fit PCA on your data
	pca.fit(X)

	# Transform the data to the new 2D space
	X_reduced = pca.transform(X)

	return pca