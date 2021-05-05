import numpy as np
import scipy.sparse as sp

#==========#==========#==========#==========#==========#==========#==========#
# 產生連接矩陣H


def get_np_H(hyperedges: list, num_nodes: int):
    arr = np.arange(num_nodes)
    H_T = []
    for edge in hyperedges:
        arr1 = np.tile(arr, (len(edge), 1))
        arr2 = np.tile(edge, (num_nodes, 1)).transpose()
        v_edge = (arr1 == arr2).astype(np.int).sum(0)
        H_T.append(v_edge)
    H = np.array(H_T).transpose()
    return H.astype(np.float32)


def get_sp_H(hyperedges: list, num_nodes: int):
    num_hyperedges = len(hyperedges)
    e_rows = np.array([node for edge in hyperedges for node in edge])

    num_nodes_list = [len(edge) for edge in hyperedges]
    e_cols = np.repeat(range(len(num_nodes_list)), num_nodes_list)

    values = np.ones(shape=(len(e_rows)))

    H = sp.coo_matrix((values, (e_rows, e_cols)),
                      shape=[num_nodes, num_hyperedges])
    return H.astype(np.float32)

#==========#==========#==========#==========#==========#==========#==========#
# 從 H 計算出 鄰接矩陣adj


def get_sp_adj_from_H(H, W=None):
    num_nodes, num_hyperedges = H.shape
    if W is None:
        W = np.ones(shape=[num_hyperedges])
    W = np.reshape(W, [1, num_hyperedges])
    Dv = np.sum(H.multiply(W), -1)  # (num_nodes, 1)
    Dv = sp.diags(np.squeeze(np.asarray(Dv)))
    # H @ W @ H.T
    HWH = H.multiply(W) @ H.T
    adj = HWH - Dv
    return adj

#==========#==========#==========#==========#==========#==========#==========#
# 從 H 計算出 Dv^(-0.5)@H @ W@De^(-1) @ H@Dv^(-0.5)


def get_sp_DvH_WDe_HDv(H, W=None):
    num_nodes, num_hyperedges = H.shape
    if W is None:
        W = np.ones(shape=[num_hyperedges])
    W = np.reshape(W, [1, num_hyperedges])
    De = np.sum(H, -2)  # (1, num_hyperedges)
    Dv = np.sum(H.multiply(W), -1)  # (num_nodes, 1)

    # WDe = W @ De^(-1)
    WDe = np.multiply(W, np.power(De, -1))
    # DvH = Dv^(-0.5) @ H
    DvH = H.multiply(np.power(Dv, -0.5))
    # DvH @ WDe @ HDv
    DvH_WDe_HDv = DvH.multiply(WDe) @ DvH.T
    return DvH_WDe_HDv

#==========#==========#==========#==========#==========#==========#==========#
# 從 H 計算出 正規化拉普拉斯 L_norm


def get_sp_L_norm_from_H(H, W=None):
    num_nodes, num_hyperedges = H.shape
    DvH_WDe_HDv = get_sp_DvH_WDe_HDv(H, W)
    L_norm = sp.eye(num_nodes) - DvH_WDe_HDv
    return L_norm


#==========#==========#==========#==========#==========#==========#==========#
# 將 L_norm 做chebyshev輸入的正規化

def get_sp_L_chebyshev_norm_from_H(H, W=None, lambda_max=2.0):
    num_nodes, num_hyperedges = H.shape
    DvH_WDe_HDv = get_sp_DvH_WDe_HDv(H, W)
    # chebyshev norm
    if lambda_max == 2:
        L_chebyshev_norm = -DvH_WDe_HDv
    else:
        L_norm = sp.eye(num_nodes) - DvH_WDe_HDv
        L_chebyshev_norm = (2.0 / lambda_max) * L_norm - sp.eye(N)
    return L_chebyshev_norm
