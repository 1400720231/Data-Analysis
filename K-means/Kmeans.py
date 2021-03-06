import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import *
"""
随机初始点
聚类数量k最大为7 因为只设置了7种颜色，多了颜色会重复
想要更多聚类，颜色满足即可



"""


class KMeans():
    # n_clusters是聚类数，data表示数据集，矩阵格式
    def __init__(self, n_clusters,data):
        self.k = n_clusters
        self.data = data
        self.new_assigns = []
        self.colors = ['b','c','m','g','y','w']
        self.markers =['o','s','*','h','H','+']
    def fit(self):
        n_samples, _ = self.data.shape
        # initialize cluster centers
        # 随机生成初始中心
        self.centers =np.array(random.sample(list(self.data), self.k))
        # 准备用来保存聚类后的中心
        self.initial_centers = np.copy(self.centers)
        old_assigns = None
        n_iters = 0
        while True:
            # 每次分类后的结果列表，像这样[0,1,2,1,2,1,2,1,4]
            self.new_assigns  = [self.classify(datapoint) for datapoint in self.data]
            # 判断此次分类结果与上一次是否相同，相同的话就停止迭代，结束函数
            if self.new_assigns  == old_assigns:
                print("Training finished after {0} iterations!".format(n_iters))
                return 

            old_assigns = self.new_assigns 
            n_iters += 1
            for id_ in range(self.k):
                # 对所有值为id的进行下标检索，返回索引值，为array
                points_idx = np.where(np.array(self.new_assigns ) == id_)  
                datapoints = self.data[points_idx]  # 直接在最原始的data中获取所有对应的下标索引的值为第id类
                self.centers[id_] = datapoints.mean(axis=0) # 取第id类的平均值，为此类的中心，并且覆盖原来的中心
    def distEclud(self,vecA, vecB):
        """
        arrary()把数组转化成矩阵，方便加减
        power(a,2) 对a中的每个数字都平方
        sum求和
        sqrt开方
        """
        return sqrt(sum(power(array(vecA)- array(vecB),2))) # 求两个向量之间的距离

    def summary(self):
        a = self.distEclud(self.centers[0],self.centers[1])
        b = self.distEclud(self.centers[0],self.centers[2])
        c = self.distEclud(self.centers[1],self.centers[2])
        # f1
        res =a+b+c
        #f2
        d =[]
        e = []
        f = []
        for i in range(len(self.new_assigns)):
            if self.new_assigns[i] ==0:
                d.append(self.distEclud(self.centers[0],self.data[i]))
            elif self.new_assigns[i]==1:
                e.append(self.distEclud(self.centers[1],self.data[i]))
            else:
                f.append(self.distEclud(self.centers[2],self.data[i]))
        return res,sum(d)+sum(f)+sum(e)
    # 某个点与centers中所有点的距离开方的列表
    def l2_distance(self, datapoint):
        dists = np.sqrt(np.sum((self.centers - datapoint)**2, axis=1))
        return dists

    def classify(self, datapoint):
        dists = self.l2_distance(datapoint)
        return np.argmin(dists) # 取距离中的最小值的索引下标值，即对某个点分类成功

    def plot_clusters(self):
        plt.figure(figsize=(12,10))
        plt.title("Initial centers in red with triangle, final centers in red with point")
        for i in range(len(self.new_assigns)):
            plt.scatter(self.data[:, 0][i], self.data[:, 1][i], marker=self.markers[self.new_assigns[i]], c=self.colors[self.new_assigns[i]])
        plt.scatter(self.centers[:, 0], self.centers[:,1], c='r')
        plt.scatter(self.initial_centers[:, 0], self.initial_centers[:,1], c='k',marker='v')
        # # x y轴增加标签信息
        plt.xlabel('Sepal.Length')
        plt.ylabel('Sepal.width')
        plt.show()




