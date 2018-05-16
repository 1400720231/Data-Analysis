from numpy import *
import numpy as np
import random
 # 加载数据
def loadDataSet(fileName): # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = [] # 文件的最后一个字段是类别标签
    with open(fileName,'r') as f:
        for line in f.readlines():
            curLine = line.strip('\n').split(',')
            fltLine = list(map(float, curLine))   # 将每个元素转成float类型
            dataMat.append(fltLine)
    return np.array(dataMat)# 此时返回的是一个列表，其中每个嵌套数组的元素都是float型

# 计算欧几里得距离
def distEclud(vecA, vecB):
    """
    arrary()把数组转化成矩阵，方便加减
    power(a,2) 对a中的每个数字都平方
    sum求和
    sqrt开方
    """
    return sqrt(sum(power(array(vecA)- array(vecB),2))) # 求两个向量之间的距离



# 计算F1 F2
def summary(centers,data,new_assign):
    a = distEclud(centers[0],centers[1])
    b = distEclud(centers[0],centers[2])
    c = distEclud(centers[1],centers[2])
    # f1
    res =a+b+c
    #f2
    d =[]
    e = []
    f = []
    for i in range(len(new_assign)):
        if new_assign[i] ==0:
            d.append(distEclud(centers[0],data[i]))
        elif new_assign[i]==1:
            e.append(distEclud(centers[1],data[i]))
        else:
            f.append(distEclud(centers[2],data[i]))
    return res,sum(d)+sum(f)+sum(e)

def combine(data,k):
    # 排列组合的每个组合的所有点之间的距离总和的值
    sums = []
    from itertools import combinations
    # 排列结果的组合列表 [(1,2,3),(0,3,4),(3,1,5)....]
    combines = [c for c in  combinations(range(len(data)), k)]
    for i in combines:
        # 排列结果的组合列表 [(1,2),(0,3),(3,1)....]
        b = [(i[c[0]],i[c[1]]) for c in  combinations(range(len(i)), 2)]

        # 求出一个组合的所有点之间的组合的距离，并求和
        values = sum([distEclud(list(data)[l[0]],list(data)[l[1]]) for l in b])
        sums.append(values)
    sums2 = sorted(sums,reverse=True)  # 由大到小排序
    index = sums.index(sums2[0])
    centers = []
    for i in combines[index]:
        centers.append(list(data)[i])
    return np.array(centers)


"""
# 画图代码
# 获取改进后算法初始聚类中心点
centers = combine(data,3)
x =[]
y = []
for i in data:
    x.append(i[0])
    y.append(i[1])
#散点图
plt.scatter(x,y)   
# s=150表示点的大小
plt.scatter(centers[0][0],centers[0][1],s=150,marker='*',c='r')
plt.scatter(centers[1][0],centers[1][1],s=150,marker='*',c='r')
plt.scatter(centers[2][0],centers[2][1],s=150,marker='*',c='r')
plt.xlabel('Sepal.Length')
plt.ylabel('Sepal.Width')
plt.show()

-----------------------------------------------
#f1f2画图：

# 读取csv文件数据格式化
f =open('f12.csv','r')
data = f.readlines()
x =[] # f1
y=[] # f2
for i in data:
    x.append(float(i.split(',')[0]))
for i in data:
    y.append(float(i.split(',')[1]))




#画f1

# 画随机f1
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12],x)
# 画不随机f1
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12],[10.127641561315702,10.1276415613
15702,10.127641561315702,10.127641561315702,10.127641561315702,10.12764
1561315702,10.127641561315702,10.127641561315702,10.127641561315702,10.127641561315702,10.127641561315702,10.127641561315702])
# 画点上去
plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12],x,s=80)
n [108]: plt.plot([1,2,3,4,5,6,7,8,9,10,11,12],x,linewidth=2.5)
Out[108]: [<matplotlib.lines.Line2D at 0x7f12ef178438>]
In [109]: plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12],x,s=100)
Out[109]: <matplotlib.collections.PathCollection at 0x7f12ef163048>
In [110]: plt.plot([1,2,3,4,5,6,7,8,9,10,11,12],[10.127641561315702,10.127641561
     ...: 315702,10.127641561315702,10.127641561315702,10.127641561315702,10.127
     ...: 641561315702,10.127641561315702,10.127641561315702,10.127641561315702,
     ...: 10.127641561315702,10.127641561315702,10.127641561315702],linewidth=2.
     ...: 5)
Out[110]: [<matplotlib.lines.Line2D at 0x7f12ef163d68>]
In [111]: plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12],[10.127641561315702,10.127641
     ...: 561315702,10.127641561315702,10.127641561315702,10.127641561315702,10.
     ...: 127641561315702,10.127641561315702,10.127641561315702,10.1276415613157
     ...: 02,10.127641561315702,10.127641561315702,10.1276415613157



# 画f2

# 画随机f2
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12],y,linewidth=2.5)
Out[118]: [<matplotlib.lines.Line2D at 0x7f12eea58a90>]
#画点
In [119]: plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12],y,s=100)
Out[119]: <matplotlib.collections.PathCollection at 0x7f12ee9dd828>
#画不随机f2
In [120]: plt.plot([1,2,3,4,5,6,7,8,9,10,11,12],[ 97.22486903387323, 97.22486903
     ...: 387323, 97.22486903387323, 97.22486903387323, 97.22486903387323, 97.22
     ...: 486903387323, 97.22486903387323, 97.22486903387323, 97.22486903387323,
     ...:  97.22486903387323, 97.22486903387323, 97.22486903387323],linewidth=2.
     ...: 5)
Out[120]: [<matplotlib.lines.Line2D at 0x7f12ee9ddf60>]
# 画点
In [121]: plt.scatter([1,2,3,4,5,6,7,8,9,10,11,12],[ 97.22486903387323, 97.22486
     ...: 903387323, 97.22486903387323, 97.22486903387323, 97.22486903387323, 97
     ...: .22486903387323, 97.22486903387323, 97.22486903387323, 97.224869033873
     ...: 23, 97.22486903387323, 97.22486903387323, 97.22486903387323],s=100)

"""