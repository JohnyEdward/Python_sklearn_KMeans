#-*- coding: utf-8 -*-
import  pandas as pd
import numpy as np
data0 = pd.read_excel('..\Python_sklearn_KMeans\old2.xls')
data0.dropna(axis=1)

# 以省份分组，初步查看数据结构
data1 = data0.groupby('province')
print data1.describe()
data2 = data0.as_matrix()

# 选择数据集
data_len = int(len(data0) * 0.8)
data_train = data0.iloc[:,1:]
print np.isnan(data_train).any()
# 训练分类器
from sklearn.cluster import KMeans
n_clusters_= 3
data_cluster = KMeans(n_clusters = n_clusters_)
data_cluster.fit(data_train)

data_train['jllable']  = data_cluster.labels_
data_count_type = data_train.groupby('jllable').apply(np.size)

print '各个类别的数量',data_count_type
print  '统计',data_train.groupby('jllable').mean()
print '聚类中心',data_cluster.cluster_centers_

new_data = data_train[:]
print
new_data.to_csv('..\Python_sklearn_KMeans\old_new.csv')

# 将用于聚类的数据的特征的维度降至2维，并输出降维后的数据，形成一个dataframe名字new_pca
# from  sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# new_pca = pd.DataFrame(pca.fit_transform(new_data))
# 不适合降维，舍弃
"new_pca, data_train 都是datafram "
import matplotlib.pyplot as plt
p = data_train[data_train['jllable'] == 0]

plt.plot(p.iloc[1:6], 'r.')
p = data_train[data_train['jllable'] == 1]
plt.plot(p.iloc[1:6], 'go')
d = data_train[data_train['jllable'] == 2]
plt.plot(p.iloc[1:6], 'b*')

plt.show()