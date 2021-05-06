import numpy as np
import scipy.sparse as sp


def get_edges(sparse_matrix, is_triu=True):
    coo = sp.coo_matrix(sparse_matrix)
    if is_triu:
        coo = sp.triu(coo, 1)
    return np.vstack((coo.row, coo.col)).transpose()  # .tolist()
#==========#==========#==========#==========#==========#==========#==========#
# 產生鄰接矩陣adj


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


#==========#==========#==========#==========#==========#==========#==========#
# 從 adj 計算出 D^(-0.5) @ (I+adj) @D^(-0.5)

def get_sp_DAD(adj, mode=1):
    N = adj.shape[-1]
    if mode == 1:  # 拉普拉斯平滑化
        A = sp.eye(N) + sp.coo_matrix(adj)
    elif mode == 2:  # 拉普拉斯銳利化
        A = 2 * sp.eye(N) - sp.coo_matrix(adj)
    else:
        A = sp.coo_matrix(adj)
    # 幾何平均
    D = np.power(np.sum(np.abs(A), -1), -0.5)
    # D[np.isinf(D)] = 0.
    DAD = A.multiply(D).T.multiply(D)
    return DAD.astype('float32')

#==========#==========#==========#==========#==========#==========#==========#
# 從 adj 計算出 正規化拉普拉斯 L_norm


def get_sp_L_norm_from_adj(adj):
    N = adj.shape[-1]
    a = sp.coo_matrix(adj)
    D = np.power(np.sum(np.abs(a), -1), -0.5)
    DaD = get_sp_DAD(adj, mode=1)
    L_norm = sp.eye(N) - DaD
    return L_norm


#==========#==========#==========#==========#==========#==========#==========#
# 將 L_norm 做chebyshev輸入的正規化

def get_sp_L_chebyshev_norm_from_adj(adj, lambda_max=2.0):
    N = adj.shape[-1]
    a = sp.coo_matrix(adj)
    D = np.power(np.sum(np.abs(a), -1), -0.5)
    DaD = a.multiply(D).T.multiply(D)
    # chebyshev norm
    if lambda_max == 2:
        L_chebyshev_norm = -DaD
    else:
        L_norm = sp.eye(N) - DaD
        L_chebyshev_norm = (2.0 / lambda_max) * L_norm - sp.eye(N)
    return L_chebyshev_norm.astype('float32')
