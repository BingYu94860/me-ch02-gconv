# me-ch02-gconv

###### 環境：

1. 使用 Tensorflow 2.x
2. 在 Anaconda 的 Jupyter Notebook 上執行
3. 亦可以在 Colab上運行

```python
#### 掛接 Google 雲端硬碟 ####
from google.colab import drive
drive.mount('/content/drive')

####　切換工作資料夾的目錄 ####
# root 需要更改成 自己的 資料夾路徑，並資料夾內放 所有 *.py 與 *.ipynb 的檔案。
root = "/content/drive/MyDrive/Colab Notebooks/me-ch02-gconv"
import os
os.chdir(root)
```



# Part1 圖卷積的數學運算

執行「[Demo] Math for Graph.ipynb」的 Jupyter Notebook的檔案，需要將「math_graph.py」放在同一資料夾下。



# Part2 超圖卷積的數學運算 

執行「[Demo] Math for Hyper Graph.ipynb」的 Jupyter Notebook的檔案，需要將「math_hypergraph.py」放在同一資料夾下。



# Part3 圖卷積的Keras Layer

執行「[Demo] Layer for GraphConvs.ipynb」的 Jupyter Notebook的檔案，需要將「layer_gconv.py、math_graph.py」放在同一資料夾下。

備註:  

1. 使用密集張量運算 (以後再更新，稀疏張量運算的部分)

# Part4 超圖卷積的Keras Layer

執行「[Demo] Layer for HyperGraphConvs.ipynb」的 Jupyter Notebook的檔案，需要將「layer_hgconv.py、math_hypergraph.py、math_graph.py」放在同一資料夾下。

備註:  

1. 使用密集張量運算 (以後再更新，稀疏張量運算的部分)
2.  HyperGraphConvs() 架構類似於 GraphConvs()



