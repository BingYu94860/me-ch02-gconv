import scipy.sparse as sp
import tensorflow as tf
import numpy as np
import math

# from math_graph_knn import get_adj, get_dW
# from math_graph_knn import get_adj, get_H_run_KNeighbors
# from math_graph_knn import get_adj, get_H_run_RadiusNeighbors
#==========#==========#==========#==========#==========#==========#==========#

def get_adj(edges: list, num_nodes: int):
    e_rows, e_cols = np.array(edges, dtype=np.int).transpose()
    values = np.ones(shape=(len(e_rows), ), dtype=np.float32)
    adj = sp.coo_matrix((values, (e_rows, e_cols)),
                        shape=[num_nodes, num_nodes])
    # triu adj --> adj
    adj.setdiag(0)
    bigger = adj.T > adj
    adj = adj - adj.multiply(bigger) + adj.T.multiply(bigger)
    return adj


def np_adj_to_dW(adj):
    dW = sp.coo_matrix(adj)
    dW = dW.toarray().astype('float32')
    dW[dW == 0] = float('inf')
    np.fill_diagonal(dW, 0)
    return dW


def tf_dist_W(dW, num_loop=None, batch_size=None, verbose=False):
    num_nodes = dW.shape[0]
    # 計算估計 max loop
    max_loop = int(np.ceil(np.log2(num_nodes)))
    if num_loop is not None:
        max_loop = min(num_loop, max_loop)
    # 設定 預設 batch_size
    if batch_size is None:
        if num_nodes >= 1000:
            batch_size = 10
        else:
            batch_size = num_nodes
    if verbose:
        print(f'max_loop={max_loop}, batch_size={batch_size}')
    # 迭代 max_loop 次
    for i_loop in range(max_loop):
        dW_odd = dW
        A = tf.expand_dims(dW, 2)  # [N,N,1]
        B = tf.expand_dims(dW, 0)  # [1,N,N]
        # 以 batch 方式計算 dW = tf.reduce_min(A+B, 1)
        ys = []
        for ia in range(0, num_nodes, batch_size):
            # 依序抽取 batch_size 個
            ib = ia + batch_size
            if ib >= num_nodes:
                ib = num_nodes
            iw = A[ia:ib]  # [batch_size,N,1]
            # 計算 batch_size 個
            ys.append(tf.reduce_min(iw + B, 1))
        dW = tf.concat(ys, 0)
        # 檢查如果 dW 和 dW_odd 則 提前停止
        if tf.math.reduce_all(tf.equal(dW, dW_odd)):
            break
        if verbose:
            print(f'run {i_loop}/{max_loop}')
            print(f'{dW}')
    return dW

#==========#==========#==========#==========#==========#==========#==========#


def get_dW(adj, num_loop=None, batch_size=1, verbose=False):
    dW = np_adj_to_dW(adj)  # [N,N]
    dW = tf.convert_to_tensor(dW)
    dW = tf_dist_W(dW, num_loop, batch_size, verbose=verbose)
    return dW.numpy()


#==========#==========#==========#==========#==========#==========#==========#

def tf_KNN_for_dW(dW, K=3):
    num_nodes = dW.shape[0]
    rr = tf.tile(tf.expand_dims(tf.range(num_nodes), -1), [1, K])
    cc = tf.argsort(dW, -1, direction='ASCENDING')[:, :K]
    indices = tf.stack([tf.reshape(rr, -1), tf.reshape(cc, -1)], -1)
    indices = tf.cast(indices, 'int64')
    values = tf.gather_nd(dW, indices)
    y = tf.SparseTensor(indices, values, dense_shape=[num_nodes, num_nodes])
    y = tf.sparse.to_dense(tf.sparse.reorder(y), default_value=math.inf)
    return y  # [my-node, k-node]


def get_H_run_KNeighbors(adj, n_neighbors=5, num_loop=None, batch_size=1, verbose=False):
    dW = np_adj_to_dW(adj)  # [N,N]
    dW = tf.convert_to_tensor(dW)
    dW = tf_dist_W(dW, num_loop, batch_size, verbose=verbose)
    if verbose:
        print(f'dW = \n{dW}')
    dW = tf_KNN_for_dW(dW, K=n_neighbors)
    if verbose:
        print(f'KNN={n_neighbors} => [my-node, k-node]\n{dW}')
    H = tf.linalg.matrix_transpose(dW)
    H = tf.cast(tf.logical_not(tf.math.is_inf(H)), 'float32')
    H = sp.coo_matrix(H.numpy())
    if verbose:
        print(f'H=\n{H.toarray()}')
    return H


#==========#==========#==========#==========#==========#==========#==========#

def get_H_run_RadiusNeighbors(adj, radius=None, num_loop=None, batch_size=None, verbose=False):
    dW = np_adj_to_dW(adj)  # [N,N]
    dW = tf.convert_to_tensor(dW)
    dW = tf_dist_W(dW, num_loop, batch_size, verbose=verbose)
    if verbose:
        print(f'dW = \n{dW}')
    if radius is None:
        H = tf.cast(tf.logical_not(tf.math.is_inf(dW)), 'float32')
    else:
        H = tf.cast(tf.less_equal(dW, radius), 'float32')
    H = H.numpy()
    if verbose:
        print(f'H=\n{H}')
    return H


#==========#==========#==========#==========#==========#==========#==========#
