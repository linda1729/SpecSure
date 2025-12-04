import numpy as np
import scipy.io as sio

def load_mat_features(mat_path):
    mat = sio.loadmat(mat_path)
    arr = None
    for v in mat.values():
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            arr = v
            break
    if arr is None:
        raise ValueError('mat文件未找到有效数组变量')
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 3:
        return arr.reshape(-1, arr.shape[2])
    else:
        return arr.reshape(-1, arr.shape[-1])
