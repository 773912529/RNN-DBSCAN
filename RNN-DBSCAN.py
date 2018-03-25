import numpy as np
import pandas as pd
import Dataset
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import deque
from numba import jit
import datetime

class RnnDbcsan(object):
    is_cluster = 'UNCLASSIFIED'
    count_time = 0
    '''
    updata:获取数据集并添加辅助索引:is_cluster
    '''
    def updata(self,url):
        file = url
        data = pd.read_csv(file)
        # data= list(preprocessing.normalize(data))
        # data = np.array(data)
        for _ in data:
            data['cluster'] = self.is_cluster
        return data
    pass

    '''
    排序
    '''
    def sort(self,L):  
        return np.sort(L)
    '''
    distance:返回两个观测点之间的距离
    '''
    def distance(self,x,y):
        x = np.array(x)
        y = np.array(y)
        dist = ((float(x[0]) - float(y[0]))**2 + (float(x[1]) - float(y[1]))**2)**0.5
        self.count_time += 1
        return dist
    pass

    """
    neigghbor:寻找观测点的正向邻居
    return :neighbors
    @parma X:pure_data; x:current dot
    """
    @jit
    def neighbor(self,X,x,k):
        # X = np.array([z for z in X if (z != x).any()])
        neighb_dist = []
        neighb = []
        for y in X:
            neighb_dist.append(self.distance(x[0:2],y[0:2]))
        neighb_dist = self.sort(neighb_dist)
        distanation = neighb_dist[k]
        for z in X:
            if(self.distance(x[0:2],z[0:2]) <= distanation):
                neighb.append(z)
        return np.array(neighb)
    pass

    '''
    r_neighbor:寻找观测点的逆向邻居
    '''
    # @vectorize(["float32(float32, float32)"], target='cuda')
    def r_neighbor(self,X,x,k):
        # X = np.array([z for z in X if (z != x).any()])
        r_neighb = []
        for j in X:
            j_neighb = self.neighbor(X,j,k)
            for l in j_neighb:
                # if(x[0] == l[0] and x[1] == l[1]):
                if(float(x[0]) == float(l[0]) and float(x[1]) == float(l[1])):
                    r_neighb.append(j)
            pass
        pass
        return np.array(r_neighb)
    pass

    '''
    neighbors = neighbor(x) + {y in r_neighbor(x):len(r_neighbor(y) > k}
    '''
    # @vectorize(["float32(float32, float32)"], target='cuda')
    def neighbors(self,X,x,k):
        clu = np.array([x for x in self.r_neighbor(X,x,k) if len(self.r_neighbor(X,x,k)) > k])
        neighbors = np.ndarray.tolist(self.neighbor(X,x,k)) + np.ndarray.tolist(clu)
        return np.array(neighbors)
    '''
    function assignn():将数组X中x、y值等于x的元素的cluster索引置为cluster
    '''
    def assignn(self,x,cluster,X):
        for z in X:
            if(str(x[0]) == str(z[0]) and str(x[1]) == str(z[1])):
                z[2] = cluster

    '''
    function in_assign():取出X中x、y值等于x的值
    '''
    def in_assign(self,x,X):
        for z in X:
            if(str(x[0]) == str(z[0]) and str(x[1]) == str(z[1])):
                return z
              
    '''
    expand_cluster:扩增簇的个数
    '''
    def expand_cluster(self,X,assign,x,cluster,k):
        start = datetime.datetime.now()
        if(len(self.r_neighbor(assign,x,k)) <= k):
            x[2] = 'NOISE'
            self.assignn(x,'NOISE',assign)
            return False
        else:
            queue = deque(self.neighbors(assign,x,k))
            x[2] = cluster            
            for z1 in queue:
                z1[2] = cluster
                self.assignn(z1,cluster,assign)
        while(len(queue) != 0):
            y = queue[0]
            queue.popleft()
            print(len(queue))
            if(len(self.r_neighbor(assign,y,k)) > k):
                neighbor = self.neighbors(assign,y,k)
                for z in neighbor: 
                    '''
                    @从数组assign中获取z并进行比较（因为在循环中neighbor不会实时更新）
                    '''
                    if(self.in_assign(z,assign)[2] == 'UNCLASSIFIED'):
                        z[2] == cluster
                        queue.append(z)
                        '''
                        @在数组asign中将z置位cluster
                        '''
                        self.assignn(z,cluster,assign)                   
                    elif(str(z[2]) == 'NOISE'):
                        z[2] = cluster
                    pass
                pass
            pass
        end = datetime.datetime.now()
        print(end-start)
        return True

    '''
    expand_clusters:将簇拓展成完整的簇
    '''
    def expand_clusters(self,X,k):
        for x in X:
            if(x[2] == 'NOISE'):
                neighb = self.neighbor(X,x,k)
                mincluster = 'NOISE'
                mindist = 999999
                for n in neighb:
                    cluster = n[2]
                    dist = self.distance(x[0:2],n[0:2])
        # print(X)
                

    '''
    作图函数
    '''
    def draw_pic(self,X):
        # 蓝色、蓝绿色、绿色、黑色、品红、红色、白色、黄色
        color = ['b','c','g','k','m','r','w','y']
        for x in range(10):
            for j in X:
                if(str(j[2]) == 'NOISE'):
                    plt.scatter(j[0],j[1],c='y')
                elif(j[2] == x):
                    plt.scatter(j[0],j[1],c=color[x])
        plt.show()

    '''
    RNN-DBSCAN主函数 
    '''
    def main_function(self,X,k):
        cluster = 0
        assign = X
        for x in X:
            q = [j for j in assign if str(j[0]) == str(x[0]) and str(j[1]) == str(x[1])]
            if(q[0][2] == 'UNCLASSIFIED'):
                if(self.expand_cluster(X,assign,x,cluster,k)):
                    cluster+=1
                pass
            pass
        pass
