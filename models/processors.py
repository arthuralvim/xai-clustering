import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_and_normalization(
    data, apply_normalization=True, apply_l2_normalization=True, pca=256
):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    if apply_normalization:
        scaling = StandardScaler()
        scaled_data = scaling.fit_transform(data)
    else:
        scaled_data = data

    principal = PCA(n_components=pca)
    p_data = principal.fit_transform(scaled_data)

    if apply_l2_normalization:
        # L2 normalization
        row_sums = np.linalg.norm(p_data, axis=1)
        p_data = p_data / row_sums[:, np.newaxis]

    return p_data
