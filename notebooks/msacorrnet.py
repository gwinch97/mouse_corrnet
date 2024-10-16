import numpy as np
from copy import deepcopy
import networkx as nx
from tqdm import trange, tqdm
import scipy as sp
from scipy.sparse.csgraph import laplacian
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lobpcg
import topcorr
import os
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from sklearn.linear_model import LinearRegression

def check_stationarity(X):
    """
    Computes the relative slopes of time series data to assess stationarity by fitting
    a linear regression model to each time series and normalizing the slope by the data range.

    Parameters:
    -----------
    X : numpy.ndarray
        A 2D array where each column represents a time series. The rows correspond to
        time points, and the columns to different variables or series.

    Returns:
    --------
    relative_slopes : numpy.ndarray
        A 1D array where each element represents the relative slope of the corresponding
        time series in `X`. The relative slope is calculated as the ratio of the linear
        regression slope to the range (max - min) of the data. Values closer to zero indicate
        a more stationary time series.
    """
    relative_slopes = np.zeros((X.shape[1]))
    for i in range(X.shape[1]):
        model = LinearRegression().fit(np.arange(len(X[:,i])).reshape(-1, 1), X[:,i])
        data_range = X[:,i].max() - X[:,i].min()
        relative_slopes[i] = model.coef_[0] / data_range

    return relative_slopes

def align_data(data_dict, samplef, t):
    """
    Aligns microservice data collected from the ChaosStarBench deployment to a common time stamp
    for consistent comparison across services.

    Parameters:
    -----------
    data_dict : dict
        A dictionary where the keys are microservice names and the values are
        2D numpy arrays. For arrau the first column represents timestamps and the second column corresponds to
        measurements (CPU usage, memory usage, etc.).

    samplef : float
        The sampling frequency, which defines the interval between aligned time points.

    t : float
        The total duration over which the data should be aligned, starting from 0 up to t.

    Returns:
    --------
    aligned_dict : dict
        A dictionary containing the aligned data for each service. Each entry is a 2D array
        with two columns: the first column contains the aligned timestamps, and the second
        column contains the corresponding measurements, aligned to the common time grid.
        Missing data points are filled with zeros.
    """
    align_t = np.arange(0, t, samplef)
    aligned_dict = {}

    # Check for missing data
    for s, data in data_dict.items():
        if data.size == 0 or data.shape[0] == 3600:
            continue

        norm_to_start = data[:, 0] - data[0, 0]  # TODO: replace data[0,0] with ensemble start when implemented
        align_mapping = np.zeros(align_t.size)
        for ts in align_t:
            exists_within_tolerance = \
            np.where((norm_to_start >= ts - samplef / 2) & (norm_to_start <= ts + samplef / 2))[0]
            if len(exists_within_tolerance) > 0:
                align_mapping[int(ts / samplef)] = int(exists_within_tolerance[0])

        mask = align_mapping > 0
        mask[0] = True
        aligned_data = np.column_stack((align_t, np.zeros(align_t.size)))
        aligned_data[mask, 1] = data[align_mapping[mask].astype(int), 1]
        aligned_dict[s] = deepcopy(aligned_data)

    return aligned_dict


def win_shape(w, w_shape):
    """
    Generates a window shape based on the specified type and size.

    Parameters:
    -----------
    w : int
        The size of the window, this must match the number of samples in your array.

    w_shape : str
        The type of window shape. It can take one of the following values:
        - "square": A square window where each element is equally weighted.
        - "tapered": A tapered (exponential decay) window with weights that decrease
          exponentially across the window.

    Returns:
    --------
    numpy.ndarray
        A 2D array of shape (w, 1) representing the window with the desired shape. The
        array will have w rows and 1 column.

        - For "square", all elements are equal to 1/w (uniform weights).
        - For "tapered", the elements follow an exponential decay pattern.
    """
    if w_shape == "square":
        return np.ones((w, 1)) / w

    elif w_shape == "tapered":
        theta = np.round(w / 3)
        w0 = (1 - np.exp(-1 / theta)) / (1 - np.exp(-w / theta))
        return ((w0 * np.exp((np.array(range(-w + 1, 1)) / theta))).T).reshape(w, 1)


def weightedcorrs(X, w):
    """
    Computes the weighted correlation matrix for a dataset X with given weights w.

    Parameters:
    -----------
    X : numpy.ndarray
        A 2D array of shape (m_samples, n_features), where each row represents a data point and each column represents
        a feature.

    w : numpy.ndarray
        A 1D array of shape (dt, 1) representing the weights for each observation.

    Returns:
    --------
    numpy.ndarray
        A 2D array of shape (N, N) representing the weighted correlation matrix. Each element
        in the matrix represents the correlation between the corresponding variables in X
        after accounting for the given weights.
    """
    dt, N = np.shape(X)
    temp = X - np.tile(np.dot(w.T, X), (dt, 1))
    temp = np.dot(temp.T, (temp * np.tile(w, (1, N))))
    temp = 0.5 * (temp + temp.T)
    R = np.diag(temp)
    R = R.reshape(len(R), 1)
    R = temp / np.sqrt(np.dot(R, R.T))
    return R


def rolling_window(X, w_size, w_shape, step, thr=0, type_thr=False, corr="pearsons"):
    """
    Computes correlation matrices over sliding windows of data with various correlation methods and applies thresholding.

    Parameters:
    ----------
    X : np.ndarray
        Input data array of shape (n_samples, n_features), where each row represents a data point and each column
        represents a feature.
    w_size : int
        Size of the sliding window, i.e., the number of data points to include in each window.
    w_shape : tuple
        Shape of the window for the weighted correlation calculation (used when corr="pearsons").
    step : int
        Step size to move the sliding window along the data.
    thr : float, optional, default=0
        Threshold value to apply to the correlation matrix. Values below the threshold are set to zero.
    type_thr : bool or str, optional, default=False
        Type of thresholding to apply:
        - False: Zeroes out values within the range (-thr, thr).
        - "neg": Zeroes out all values greater than -thr.
        - "pos": Zeroes out all values less than thr.
        - "proportional": Applies proportional thresholding such that the top thr% strongest edges are kept.
        - "pmfg": Applies the Planar Maximally Filtered Graph (PMFG) with optional thresholding.
        - "tmfg": Applies the Triangulated Maximally Filtered Graph (TMFG) with absolute filtering.
    corr : str, optional, default="pearsons"
        Correlation method to use:
        - "pearsons": Pearson's correlation coefficient with optional weighting.
        - "spearmans": Spearman's rank correlation coefficient.
        - "dcca": Detrended Cross-Correlation Analysis (DCCA).

    Returns:
    -------
    r_dict : dict
        A dictionary where each key is the starting index of the window and the value is the corresponding correlation matrix.
    """
    data_len = (np.shape(X)[0] // step) * step
    r_dict = {}
    for w in trange(0, int(data_len - (w_size)), step):
        if corr == "pearsons":
            r_dict[w] = weightedcorrs(X[w:w + w_size, :], win_shape(w_size, w_shape))

        elif corr == "spearmans":
            corr_matrix = np.zeros((X.shape[1], X.shape[1]))
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    rho, _ = spearmanr(X[w:w + w_size, i], X[w:w + w_size, j])
                    corr_matrix[i, j] = rho
                    corr_matrix[j, i] = rho
            np.fill_diagonal(corr_matrix, 1)
            r_dict[w] = corr_matrix

        elif corr == "dcca":
            r_dict[w] = topcorr.dcca(X[w:w + w_size, :], 25)

        r_dict[w][np.where(np.isnan(r_dict[w]) == True)] = 0
        np.fill_diagonal(r_dict[w], 1)

        ## Thresholding
        if type_thr == False:
            r_dict[w][np.where((r_dict[w] < thr) & (r_dict[w] > -thr))] = 0
        elif type_thr == "neg":
            r_dict[w][np.where(r_dict[w] > -thr)] = 0
        elif type_thr == "pos":
            r_dict[w][np.where(r_dict[w] < thr)] = 0
        elif type_thr == "proportional":
            r_dict[w] = proportional_thr(r_dict[w], thr)
        elif type_thr == "pmfg":
            if thr != 0:
                r_dict[w][np.where(r_dict[w] > -thr)] = 0
            r_dict[w] = nx.to_numpy_array(topcorr.pmfg(r_dict[w]))
        elif type_thr == "tmfg":
            r_dict[w] = nx.to_numpy_array(topcorr.tmfg(r_dict[w], absolute=True))
    return r_dict


def eigenspectrum(L):
    """
    Computes the eigenvalues for a given Laplacian matrix and sorts them in descending order.

    Parameters:
    -----------
    L : numpy array
        A laplacian matrix.

    Returns:
    -----------
    eigenvalues : numpy array
        The eigenvalues sorted in descending order.
    """
    eigvals = np.real(np.linalg.eig(L)[0])
    return -np.sort(-eigvals)


def all_spectra_lobpcg(A_dict, N, norm=True):
    """
    Computes the eigenspectra (smallest eigenvalues) of the Laplacian matrices corresponding
    to a set of adjacency matrices from graphs, using the Locally Optimal Block Preconditioned
    Conjugate Gradient (LOBPCG) method.

    Parameters:
    -----------
    A_dict : dict
        A dictionary where the keys are identifiers for each graph and the values are adjacency matrices.
        These adjacency matrices represent the connections between nodes in the graph.
    N : int
        A positive integer representing the number of eigenvalues to compute, from lowest to highest.

    norm : bool, optional (default=True)
        A boolean indicating whether to compute the normalized Laplacian matrix.

    Returns:
    --------
    eigenspectrums : numpy.ndarray
        A 2D array of shape (N, len(dict_keys)), where each column corresponds to the first
        N smallest eigenvalues (the eigenspectrum) of the Laplacian matrix for each adjacency
        matrix in A_dict.

    Notes:
    ------
    """
    dict_keys = list(A_dict.keys())
    eigenspectrums = np.zeros((N, len(dict_keys)))
    i = 0
    for key in dict_keys:
        L = laplacian(A_dict[key], norm)
        L_sparse = csr_matrix(L)
        X = np.random.normal(size=(L_sparse.shape[0], N))
        eigenspectrums[:, i], _ = lobpcg(L_sparse, X, largest=False)
        i += 1
    return eigenspectrums


def all_spectra(A_dict, norm=True):
    """
    Computes the full eigenspectra of Laplacian matrices derived from a set of adjacency
    matrices for graphs.

    Parameters:
    -----------
    A_dict : dict
        A dictionary where the keys are identifiers for each graph and the values are adjacency matrices.
        These adjacency matrices represent the connections between nodes in the graph.

    norm : bool, optional (default=True)
        A boolean indicating whether to compute the normalized Laplacian matrix.

    Returns:
    --------
    eigenspectrums : numpy.ndarray
        A 2D array where each column contains the full set of eigenvalues (the eigenspectrum)
        for the Laplacian matrix corresponding to each adjacency matrix in A_dict. The number
        of rows is equal to the size of the adjacency matrix, and the number of columns is
        equal to the number of graphs.
    """
    dict_keys = list(A_dict.keys())
    eigenspectrums = np.zeros((np.shape(A_dict[dict_keys[0]])[0], len(dict_keys)))
    i = 0
    for key in tqdm(dict_keys):
        L = laplacian(A_dict[key], norm)
        eigenspectrums[:, i] = eigenspectrum(L)
        i += 1
    return eigenspectrums


def all_spectra_parallel(A_dict, norm=True):
    """
    Computes the full eigenspectra of Laplacian matrices derived from a set of adjacency
    matrices for graphs in parallel.

    Parameters:
    -----------
    A_dict : dict
        A dictionary where the keys are identifiers for each graph and the values are adjacency matrices.
        These adjacency matrices represent the connections between nodes in the graph.

    norm : bool, optional (default=True)
        A boolean indicating whether to compute the normalized Laplacian matrix.

    Returns:
    --------
    eigenspectrums : numpy.ndarray
        A 2D array where each column contains the full set of eigenvalues (the eigenspectrum)
        for the Laplacian matrix corresponding to each adjacency matrix in A_dict. The number
        of rows is equal to the size of the adjacency matrix, and the number of columns is
        equal to the number of graphs.
    """
    dict_keys = list(A_dict.keys())
    n = np.shape(A_dict[dict_keys[0]])[0]
    eigenspectrums = np.zeros((n, len(dict_keys)))

    def compute_spectrum(key):
        L = laplacian(A_dict[key], norm)
        return eigenspectrum(L)

    with tqdm_joblib(tqdm(total=len(dict_keys))) as progress_bar:
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(compute_spectrum)(key) for key in dict_keys
        )

    for i, spectrum in enumerate(results):
        eigenspectrums[:, i] = spectrum

    return eigenspectrums

def all_spectra_signed(A_dict, norm=True):
    """
    Computes the composite eigenspectra of the positive and negative Laplacian matrices derived from a set of adjacency
    matrices for graphs in parallel.

    Parameters:
    -----------
    A_dict : dict
        A dictionary where the keys are identifiers for each graph and the values are adjacency matrices.
        These adjacency matrices represent the connections between nodes in the graph.

    norm : bool, optional (default=True)
        A boolean indicating whether to compute the normalized Laplacian matrix.

    Returns:
    --------
    eigenspectrums : numpy.ndarray
        A 2D array where each column contains the full set of eigenvalues (the eigenspectrum)
        for the Laplacian matrix corresponding to each adjacency matrix in A_dict. The number
        of rows is equal to the size of the adjacency matrix, and the number of columns is
        equal to the number of graphs.
    """
    dict_keys = list(A_dict.keys())
    eigenspectrums = np.zeros((np.shape(A_dict[dict_keys[0]])[0], len(dict_keys)))
    i=0
    for r in A_dict:
        R = A_dict[r]
        A_pos = np.maximum(0, R)
        A_neg = -np.minimum(0, R)

        D_pos = np.diag(np.sum(A_pos, axis=1))
        D_neg = np.diag(np.sum(A_neg, axis=1))

        D_pos_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_pos)))
        D_neg_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_neg)))

        D_pos_inv_sqrt[np.isinf(D_pos_inv_sqrt)] = 0
        D_neg_inv_sqrt[np.isinf(D_neg_inv_sqrt)] = 0

        L_pos = np.eye(R.shape[0]) - np.dot(np.dot(D_pos_inv_sqrt, A_pos), D_pos_inv_sqrt)
        L_neg = np.eye(R.shape[0]) - np.dot(np.dot(D_neg_inv_sqrt, A_neg), D_neg_inv_sqrt)

        L_signed = L_pos + L_neg

        eigenvalues, eigenvectors = np.linalg.eigh(L_signed)

        eigenspectrums[:,i] = eigenvalues

        i+=1

    return eigenspectrums


def snapshot_dist(eigenspectrums, norm=True):
    """
    Computes the pairwise Euclidean distance between the eigenspectra of different snapshots
    (graphs).

    Parameters:
    -----------
    eigenspectrums : numpy.ndarray
        A 2D array where each column represents the eigenspectrum (eigenvalues) of the Laplacian
        for a specific snapshot or graph. The number of rows corresponds to the number of
        eigenvalues, and the number of columns corresponds to the number of snapshots.

    norm : bool, optional (default=True)
        A boolean indicating whether to normalize the distances between eigenspectra.

    Returns:
    --------
    dist : numpy.ndarray
        A 2D array of shape (N, N), where N is the number of snapshots (graphs). Each element
        dist[i, j] contains the Euclidean distance between the eigenspectrum of snapshot i and
        snapshot j.
    """
    N = np.shape(eigenspectrums)[1]
    dist = np.zeros((N, N))
    for i in trange(N):
        for j in range(N):
            dist[i, j] = np.sqrt(np.sum(np.power((eigenspectrums[:, i] - eigenspectrums[:, j]), 2)))
            if norm == True:
                if max(max(eigenspectrums[:, i]), max(eigenspectrums[:, j])) > 1e-10:
                    dist[i, j] = dist[i, j] / np.sqrt(
                        max((np.sum(np.power(eigenspectrums[:, i], 2))), (np.sum(np.power(eigenspectrums[:, j], 2)))))
                else:
                    dist[i, j] = 0

    return dist


def landmark_snapshot_dist(eigenspectrums, lm_inds, norm=True):
    """
    IGNORE THIS FUNCTION
    """
    N = np.shape(eigenspectrums)[1]
    dist = np.zeros((N, len(lm_inds)))
    for i in trange(N):
        for j in range(len(lm_inds)):
            k = lm_inds[j]
            dist[i, j] = np.sqrt(np.sum(np.power((eigenspectrums[:, i] - eigenspectrums[:, k]), 2)))
            if norm == True:
                if max(max(eigenspectrums[:, i]), max(eigenspectrums[:, k])) > 1e-10:
                    dist[i, j] = dist[i, j] / np.sqrt(
                        max((np.sum(np.power(eigenspectrums[:, i], 2))), (np.sum(np.power(eigenspectrums[:, k], 2)))))
                else:
                    dist[i, j] = 0
    return dist


def LMDS(D, lands, dim):
    """
    Performs Landmark Multidimensional Scaling (LMDS) on a distance matrix to reduce
    the dimensionality of data, using a subset of points (landmarks) to approximate
    the lower-dimensional coordinates of the full dataset.

    Parameters:
    -----------
    D : numpy.ndarray
        A 2D distance matrix of shape (N, L), where N is the number of eigenspecta and L is the number of landmarks.
        Each element D[i, j] represents the distance between eigenspectra i and landmark eigenspectra j.

    lands : list or array-like
        A list or array of indices specifying which points are used as landmarks. The
        landmark points are used to approximate the embedding for all data points.

    dim : int
        The number of dimensions to project the data into. This represents the target
        dimensionality of the reduced dataset.

    Returns:
    --------
    numpy.ndarray
        A 2D array of shape (N, dim) containing the coordinates of the data points in the
        reduced-dimensional space. Each row corresponds to a point, and each column to
        a coordinate in the new dimensionality.
    """
    Dl = D[:, lands]
    n = len(Dl)

    # Centering matrix
    H = - np.ones((n, n)) / n
    np.fill_diagonal(H, 1 - 1 / n)
    # YY^T
    H = -H.dot(Dl ** 2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(H)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    if dim:
        arr = evals
        w = arr.argsort()[-dim:][::-1]
        if np.any(evals[w] < 0):
            print('Error: Not enough positive eigenvalues for the selected dim.')
            return []
    if w.size == 0:
        print('Error: matrix is negative definite.')
        return []

    V = evecs[:, w]
    N = D.shape[1]
    Lh = V.dot(np.diag(1. / np.sqrt(evals[w]))).T
    Dm = D - np.tile(np.mean(Dl, axis=1), (N, 1)).T
    X = -Lh.dot(Dm) / 2.
    X -= np.tile(np.mean(X, axis=1), (N, 1)).T

    _, evecs = sp.linalg.eigh(X.dot(X.T))

    return (evecs[:, ::-1].T.dot(X)).T


def r_thr(r, thr, signed="pos"):
    """
    Thresholds a matrix `r` by applying a cutoff value, `thr`, and modifies the matrix based on
    the specified sign.
    Args:
        r : numpy.ndarray
            A 2D array representing a correlation matrix.

        thr : float
            The threshold value. Elements of 'r' below this threshold will be zeroed dependeing on the 'signed'
            parameter.

        signed : str or bool
            A parameter that controls the thresholding behavior:
            - "pos": Only positive values are considered, values below `thr` are set to 0.
            - "neg": Only negative values are considered, values above `-thr` are set to 0.
            - False: Both positive and negative values are considered. Values with absolute
              magnitudes smaller than `thr` are set to 0.


    Returns:
    --------
    numpy.ndarray
        The modified matrix `r`, with values thresholded based on the given conditions.

    """
    if signed == False:
        r[np.where((r < thr) & (r > -thr))] = 0
    elif signed == "neg":
        r[np.where(r > -thr)] = 0
    elif signed == "pos":
        r[np.where(r < thr)] = 0
    return r


def proportional_thr(r, p):
    """
    Applies a proportional thresholding to the input matrix `r`, retaining the top `p` proportion
    of the strongest values (in terms of magnitude) while setting the rest to zero.

    Parameters:
    -----------
    r : numpy.ndarray
        A 2D square matrix, typically representing correlation coefficients or other pairwise
        relationships.

    p : float
        The proportion of the strongest values to retain in the matrix. Should be a value between
        0 and 1, where 1 retains all values.

    Returns:
    --------
    thr_r : numpy.ndarray
        A 2D matrix of the same shape as `r`, where the top `p` proportion of values (by magnitude)
        are retained, and the rest are set to zero.
    """
    thr_r = np.zeros((r.shape))
    r[np.where((-0.05 < r) & (r < 0.05))] = 0
    ut = np.triu(r, k=1)
    n = ((len(r) ** 2) / 2) - len(r)
    n = int(n * p)
    if len(np.where((0 < ut) | (ut < 0))[0]) < n:
        return r
    else:
        elem = [[x, y] for x, y in zip(np.where((0 < ut) | (ut < 0))[0], np.where((0 < ut) | (ut < 0))[1])]
        vals = ut[np.where((0 < ut) | (ut < 0))]
        vals = abs(vals)
        ind = np.argpartition(vals, -n)[-n:]
        for i in ind:
            thr_r[elem[i][0], elem[i][1]] = r[elem[i][0], elem[i][1]]
            thr_r[elem[i][1], elem[i][0]] = r[elem[i][0], elem[i][1]]
        np.fill_diagonal(thr_r, 1)
        return thr_r


def pmfg(corr_mat):
    """
    Constructs a Planar Maximally Filtered Graph (PMFG) from a given correlation matrix.

    Parameters:
    -----------
    corr_mat : numpy.ndarray
        A 2D square matrix representing pairwise correlations between variables. Typically,
        the matrix is symmetric, with elements in the range [-1, 1] representing correlations.

    Returns:
    --------
    numpy.ndarray
        A 2D adjacency matrix representation of the Planar Maximally Filtered Graph (PMFG),
        where edges represent the strongest correlations that maintain the planarity of the graph.
    """
    corr_mat[np.where(np.isnan(corr_mat) == True)] = 0
    n = corr_mat.shape[0]
    edges = [(corr_mat[i, j], i, j) for i in range(n) for j in range(i + 1, n) if corr_mat[i, j] != 0]
    # Sort edges based on weight in descending order
    edges.sort(reverse=True, key=lambda x: x[0])

    G = nx.Graph()
    G.add_nodes_from(list(range(n)))

    for _, i, j in edges:
        G.add_edge(i, j)
        if not nx.check_planarity(G)[0]:  # Check if the graph remains planar
            G.remove_edge(i, j)  # Remove the edge if adding it violates planarity

    return nx.to_numpy_array(G)


def sub_sample_zeros(X):
    """
    IGNORE THIS FUNCTION
    """
    zeros = list(np.where(X == 0)[0][1:])
    ind = 0
    while ind != len(zeros):
        intr = []
        curr = zeros[ind]
        intr.append(zeros[ind])
        ind += 1
        if ind != len(zeros):
            while zeros[ind] == curr:
                ind += 1
                curr = zeros[ind]
                intr.append(zeros[ind])
        if intr[-1] + 1 < X.shape[0]:
            st_val = X[intr[0] - 1]
            nd_val = X[intr[-1] + 1]
            n_val = len(intr)
            step = (nd_val - st_val) / (n_val + 1)
            for i in range(intr[0], intr[-1] + 1):
                X[i] = X[i - 1] + step
    return X


def overlap_ts(X, win, step):
    """
    IGNORE THIS FUNCTION
    """
    chunks = []
    y = []
    win_st = 0
    win_nd = win_st+win
    win_ind = 0
    while win_nd <= len(X):
        chunks.append(X[win_st:win_nd,:])
        y.extend([win_ind]*win)
        win_st+=step
        win_nd+=step
        win_ind+=1
    Xt = np.concatenate(chunks)
    return Xt, y


def export_mats_to_bins(r_dict, dir):
    """
    IGNORE THIS FUNCTION
    """
    os.mkdir(dir)

    for name, matrix in r_dict.items():
        flatten_matrix = matrix.flatten()

        with open(f'{dir}/{dir}-{name}.bin', 'wb') as f:
            flatten_matrix.tofile(f)


def import_par_tmfg_res(dir, n_files, n_nodes):
    """
    IGNORE THIS FUNCTION
    """
    r_dict = {}
    for n in range(n_files):
        filename = dir+f"-{n}-exact-P-1"
        mat = np.zeros((n_nodes,n_nodes))
        with open(filename, "r") as f:
            for line in f:
                i, j, value = line.split()
                i, j = int(i), int(j)
                value = float(value)
                mat[i, j] = value

        r_dict[n] = deepcopy(mat)

    return r_dict