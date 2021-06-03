import numpy as np
import scipy.sparse as sp

# 取出無向圖的邊


def get_edges(sparse_matrix, is_triu=True):
    coo = sp.coo_matrix(sparse_matrix)
    if is_triu:
        coo = sp.triu(coo, 1)
    return np.vstack((coo.row, coo.col)).transpose()  # .tolist()

# 從邊組成無項圖的鄰接矩陣


def get_adj(edges: list, num_nodes: int):
    e_rows, e_cols = np.array(edges, dtype=np.int).transpose()
    values = np.ones(shape=[len(e_rows), ], dtype=np.float32)
    adj = sp.coo_matrix((values, (e_rows, e_cols)),
                        shape=[num_nodes, num_nodes])
    # triu adj --> adj
    adj.setdiag(0)
    bigger = adj.T > adj
    adj = adj - adj.multiply(bigger) + adj.T.multiply(bigger)
    return adj

#==========#==========#==========#==========#==========#==========#==========#


def get_line_adj(rows, cols, select='4C'):
    s = select  # {'0','90','45','135','22.5','67.5','112.5','157.5','4C','8C'}
    er, ec = rows - 1, cols - 1

    def fn(r, c, cs=cols):
        return r * cs + c  # 計算 node id

    def fe(r1, c1, r2, c2):
        return (fn(r1, c1), fn(r2, c2))  # 得到 edge
    # 斜邊: feq: 0~360 {feq1: 0~90 ; feq2: 90~180}

    def feq1(ir, ic, dr, dc):
        return fe(ir, ic + abs(dc), ir + abs(dr), ic)

    def feq2(ir, ic, dr, dc):
        return fe(ir, ic, ir + abs(dr), ic + abs(dc))

    def feq(ir, ic, dr, dc):
        return feq1(ir, ic, dr, dc) if dr * dc > 0 else feq2(
            ir, ic, dr, dc)
    # 斜線

    def qline(dr, dc, rs=rows, cs=cols):
        return [
            feq(ir, ic, dr, dc) for ir in range(rs - abs(dr))
            for ic in range(cs - abs(dc))
        ]
    # 實線 d=1; 虛線d=2

    def vline(ir, sc, d, ec=ec):
        return [
            feq(ir, ic, 0, 1) for ic in range(sc, ec, d)
        ]

    def hline(ic, sr, d, er=er):
        return [
            feq(ir, ic, 1, 0) for ir in range(sr, er, d)
        ]
    # c_vline: 垂直排列的水平線 ; r_hline: 水平排列的垂直線

    def c_vline(ic, dr=0, rs=rows):
        return [
            feq(ir, ic, dr, 1) for ir in range(rs - abs(dr))
        ]

    def r_hline(ir, dc=0, cs=cols):
        return [
            feq(ir, ic, 1, dc) for ic in range(cs - abs(dc))
        ]
    edges = []
    if s == '22.5':
        edges += qline(1, 2)  # 22.5度
        edges += c_vline(ec - 1, 1 if cols % 2 == 0 else 0)  # 垂直排列的0/45線
    if s == '67.5':
        edges += qline(2, 1)  # 67.5度
        edges += r_hline(er - 1, 1 if rows % 2 == 0 else 0)  # 水平排列的90/45線
    if s == '112.5':
        edges += qline(-2, 1)  # 112.5度
        edges += r_hline(er - 1, -1 if rows % 2 == 0 else 0)  # 水平排列的90/135線
    if s == '157.5':
        edges += qline(-1, 2)  # 157.5
        edges += c_vline(ec - 1, -1 if cols % 2 == 0 else 0)  # 垂直排列的0/135線
    if s == '0' or s == '4C' or s == '8C':
        edges += qline(0, 1)  # 0度 (水平)
    if s == '90' or s == '4C' or s == '8C':
        edges += qline(1, 0)  # 90度 (垂直)
    if s == '45' or s == '8C':
        edges += qline(1, 1)  # 45度 (右傾斜)
    if s == '135' or s == '8C':
        edges += qline(-1, 1)  # 135度 (左傾斜)
    if s == '45' or s == '135' or s == '157.5':
        edges += vline(0, 0, 2)  # 上虛線0-1
    if s == '90' or s == '22.5':
        edges += vline(0, 1, 2)  # 上虛線1-2
    if s == '90' or (s == '45' or s == '135') and rows % 2 == 0 or s == '22.5':
        edges += vline(er, 0, 2)  # 下虛線0-1
    if (s == '45' or s == '135') and rows % 2 == 1 or s == '157.5':
        edges += vline(er, 1, 2)  # 下虛線1-2
    if s == '112.5':
        edges += hline(0, 0, 2)  # 左虛線0-1
    if s == '0' or s == '45' or s == '135' or s == '67.5':
        edges += hline(0, 1, 2)  # 左虛線1-2
    if s == '0' or (s == '45' or s == '135') and cols % 2 == 1 or s == '67.5':
        edges += hline(ec, 0, 2)  # 右虛線0-1
    if (s == '45' or s == '135') and cols % 2 == 0 or s == '112.5':
        edges += hline(ec, 1, 2)  # 右虛線1-2
    if s == '22.5' or s == '157.5':
        edges += c_vline(0)  # 垂直排列的0線
    if s == '67.5' or s == '112.5':
        edges += r_hline(0)  # 水平排列的90線
    return get_adj(edges, num_nodes=rows * cols)


def get_dict_fn_line_adj():
    keys = [
        '0', '90', '45', '135', '22.5', '67.5', '112.5', '157.5', '4C', '8C'
    ]
    dict_fn_adj = {}
    for key in keys:
        dict_fn_adj[key] = lambda r, c, key=key: get_line_adj(r, c, key)
    return dict_fn_adj

#==========#==========#==========#==========#==========#==========#==========#

# 產生 p4座標 的鄰接矩陣,大小為[2^N,2^N]


def get_p4_graph(N=3, links=[(0, 1), (0, 2), (0, 3)]):
    data = [0]
    s_add = 1
    for i in range(N):
        s = pow(4, i) - 1
        nodes = [s + k * s_add for k in range(4)]
        #==========#==========#
        data = sp.coo_matrix(data)
        data = sp.block_diag((data, data, data, data)).tolil()
        for iu, iv in links:
            u, v = nodes[iu], nodes[iv]
            data[u, v] = 1
            data[v, u] = 1
        #==========#==========#
        s_add += 2 * pow(4, i)
    return sp.coo_matrix(data)


# id座標 (給予座標(ic,ir)位置做轉換)
# (按順序跑保留的[rows,cols]完,在跑要裁切的[max_num,max_num])
def get_id(ir, ic, rows, cols, max_num):
    assert ir < max_num and ic < max_num
    if ir < rows and ic < cols:
        return ir * cols + ic
    elif ir >= rows:
        return ir * max_num + ic
    else:
        return rows * cols + ir * (max_num - cols) + (ic - cols)


# p4座標 (給予座標(ic,ir)位置做轉換)
def get_p4(ir, ic, N):
    assert ir < pow(2, N) and ic < pow(2, N)
    str_format = '{0:0' + str(N) + 'b}'
    sr = str_format.format(ir)
    sc = str_format.format(ic)
    return sum([
        int(r + c, 2) * pow(4, N - 1 - i)
        for i, (r, c) in enumerate(zip(sr, sc))
    ])


def ceil_log2(x):  # x <= 2^y
    return int(np.ceil(np.log2(x)))


# p4座標 傳換到 id座標
def get_dict_p4_to_id(rows, cols):
    # rows, cols <= 2^N = max_num
    N = max(ceil_log2(rows), ceil_log2(cols))
    max_num = pow(2, N)
    dict_p4_to_id = {}
    # 給予 座標(ic,ir)位置 算出 id座標 和 p4座標
    for ir in range(max_num):
        for ic in range(max_num):
            id = get_id(ir, ic, rows, cols, max_num)
            p4 = get_p4(ir, ic, N)
            dict_p4_to_id[p4] = id
    return dict_p4_to_id


# 產生 任意大小的 p4 graph
def get_p4_adj(rows, cols, links=[(0, 1), (0, 2), (0, 3)]):
    N = max(ceil_log2(rows), ceil_log2(cols))
    num_nodes = rows * cols
    p4_adj = get_p4_graph(N, links)  # 產生p4座標 的鄰接矩陣,大小為[2^N,2^N]
    dict_p4_to_id = get_dict_p4_to_id(rows, cols)  # 產生"p4座標 傳換到id座標" 的字典
    p4_edges = get_edges(p4_adj)  # 得到p4座標的adj的edges
    # 轉換出 id座標的adj的edges
    edges = []
    for u, v in p4_edges:
        iu = dict_p4_to_id[u]
        iv = dict_p4_to_id[v]
        if iu < num_nodes and iv < num_nodes:
            edges.append((iu, iv))
    adj = get_adj(edges, num_nodes)
    return adj


def get_dict_fn_p4_adj():
    # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    e01, e02, e03 = (0, 1), (0, 2), (0, 3)
    e12, e13, e23 = (1, 2), (1, 3), (2, 3)
    dict_links = {
        'a': [e01, e02, e03],  # 爪 1
        'b': [e01, e12, e13],  # 爪 2
        'c': [e03, e13, e23],  # 爪 4
        'd': [e02, e12, e23],  # 爪 3
        'e': [e02, e03, e13],  # N
        'f': [e01, e12, e23],  # Z
        'g': [e02, e12, e13],  # |/|
        'h': [e01, e03, e23],  # Z\
        'i': [e01, e03, e12],  # 又 1
        'j': [e13, e03, e12],  # 又 2
        'k': [e23, e03, e12],  # 又 3
        'l': [e02, e03, e12],  # 又 4
        'm': [e01, e02, e13],  # ㄇ 1
        'n': [e01, e13, e23],  # ㄇ 2
        'o': [e02, e23, e13],  # ㄇ 3
        'p': [e01, e02, e23]   # ㄇ 4
    }
    dict_fn_adj = {}
    for key in dict_links.keys():
        dict_fn_adj[key] = lambda r, c, key=key: get_p4_adj(
            r, c, dict_links[key])
    return dict_fn_adj

#==========#==========#==========#==========#==========#==========#==========#

