import numpy as np
import pandas as pd
import Dataset
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import deque
from numba import jit
import datetime
'''
The value of cluster:
99:'UNCLASSIFIED'
98:'NOISE'
'''
class RnnDbcsan(object):
    is_cluster = 99
    count_time = 0

    '''
    distance:返回两个观测点之间的距离
    '''
    def distance(self,x,y):
        x = np.array(x)
        y = np.array(y)
        dist = ((float(x[0]) - float(y[0]))**2 + (float(x[1]) - float(y[1]))**2)**0.5
        return dist
    pass


    '''
    init_set:获取数据集并添加辅助索引:
    对于单个观察点：
    x[0] means the value of x
    x[1] means the value of y
    x[2] means belongs to witch cluster
    x[3] is index
    '''
    def init_set(self,url):
        file = url
        data = pd.read_csv(file)
        data = np.array(data)
        new_data = [[0.0]*4 for i in range(len(data))]
        new_data = np.array(new_data)
        count = 0
        new_data[:,[0]] = data[:,[0]]
        new_data[:,[1]] = data[:,[1]]
        new_data[:,[2]] = self.is_cluster
        for x in new_data:
            x[3] = count
            count += 1
        return new_data
    pass
    '''
    创建距离索引，将任意两个观察点的距离放入数组中
    '''
    def distance_index_struct(self,data):
        distance_index = [[0.0]*3 for i in range(len(data)*len(data))]
        distance_index = np.array(distance_index)
        num = 0
        for a in data:
            for b in data:
                distance_index[num,[0]] = a[3]
                distance_index[num,[1]] = b[3]
                distance_index[num,[2]] = self.distance(a,b)
                num += 1
        return distance_index
    '''
    创建邻居索引，将每个观察点的k近邻放入数组中
    '''
    # @jit
    def neighbor_index(self,data,k,dist_index):
        index = [[0.0]*(k+1) for i in range(len(data))]
        index = np.array(index)
        for da in data:
            que = [i for i in dist_index if i[0] == da[3]]
            que = sorted(que,key=lambda x:x[2])
            que = que[0:k+1]
            que = np.array(que)
            
            for x in range(len(que)):
                if(x == 0):
                    index[int(que[x][0]),[0]] = que[x][1]
                else:
                    index[int(que[x][0]),[x]] = que[x][1]
        return index
    pass
    '''
    创建逆邻居索引，将每个观察点的逆邻居放入数组中
    '''
    # @jit
    def r_neighbor_index(self,data,k,neb_index):
        index = [[] for i in range(len(data))]
        count = 0
        for x in data:
            index[count].append(x[3])
            for z in data:
                if(x[3] in self.find_neighbor(z[3],neb_index)):
                    index[count].append(z[3])
            count += 1
        for q in index:
            q = np.array(q)
        return index

    '''
    通过索引查找观察点的邻居
    '''
    def find_neighbor(self,x_index,neb_index):
        for i in neb_index:
            if(i[0] == x_index):
                return i[1:]

    '''
    通过索引查找观察点的逆向邻居
    '''
    def find_r_neighbor(self,x_index,r_neb_index):
        for z in r_neb_index:
            if(x_index == z[0]):
                return np.array(z[1:])

    '''
    neighbors = neighbor(x) + {y in r_neighbor(x):len(r_neighbor(y) > k}
    '''
    def neighbors(self,x_index,neb_index,r_neb_index,k):
        neighbor = self.find_neighbor(x_index,neb_index)
        r_neighbor = self.find_r_neighbor(x_index,r_neb_index)
        for q in r_neighbor:
            if(len(self.find_r_neighbor(q,r_neb_index)) <= k):
                list(r_neighbor).remove(q)
        neighbors = list(neighbor) + list(r_neighbor)
        neighbors = list(set(neighbors))
        return neighbors
        
    '''
    通过观察点的索引来获取观察点的属性值
    '''
    def get_point_by_index(self,x_index,dataset):
        q = [z for z in dataset if x_index == z[3]]
        return q[0]
    '''
    在聚类结果集中将索引为x_index聚类结果放入
    '''
    def assignn(self,x_index,cluster,X):
        for z in X:
            if(x_index == z[0]):
                z[1] = cluster
    '''
    在结果集中取出索引为x_index的观察点的聚类结果
    '''
    def get_cluster_from_assign(self,x_index,assign):
        for z in assign:
            if(x_index == z[0]):
                return z
    '''
    @prame:当前观察点，数据集，结果集，当前聚类，k，近邻索引表，逆向近邻索引表
    '''
    def expand_cluster(self,x,dataset,assign,cluster,k,neb_index,r_neb_index):
        if(len(self.find_r_neighbor(x[3],r_neb_index)) <= k):
            x[2] = 98
            self.assignn(x[3],98,assign)
            return False
        else:
            queue = deque(self.neighbors(x[3],neb_index,r_neb_index,k))
            x[2] = cluster
            for z in queue:
                self.assignn(z,cluster,assign)
            while(len(queue) != 0):
                y = queue[0]
                queue.popleft()
                if(len(self.find_r_neighbor(y,r_neb_index)) > k):
                    new_neighbor = self.neighbors(y,neb_index,r_neb_index,k)
                    for q in new_neighbor:
                        if(self.get_cluster_from_assign(q,assign)[1] == 99.0):
                            queue.append(q)
                            self.assignn(q,cluster,assign)
                        # elif(self.get_cluster_from_assign(q,assign)[1] == 98.0):
                        #     self.assignn(q,cluster,assign)
                        pass
                    pass
                pass
            pass
            return True
        pass
    
    '''
    作图函数
    '''
    def draw_pic(self,assign,dataset):
        new_assgin = dataset
        # 蓝色、蓝绿色、绿色、黑色、品红、红色、白色、黄色
        color = ['b','c','g','k','m','r','w','y']
        new_assgin[:,2] = assign[:,1]
        for x in range(10):
            for j in new_assgin:
                if(j[2] == 98.0):
                    plt.scatter(j[0],j[1],c='y')
                elif(j[2] == x):
                    plt.scatter(j[0],j[1],c=color[x])
        plt.show()

    '''
    算法主入口
    '''
    def main_function(self,dataset,k):
        start = datetime.datetime.now()
        cluster = 0
        assign = np.array([[0.0]*2 for i in range(len(dataset))])
        assign[:] = dataset[:,[3,2]]
        '''
        初始化近邻索引表和逆向近邻索引表
        '''
        dist_index = self.distance_index_struct(data)
        neb_index = self.neighbor_index(data,k,dist_index)
        r_neb_index = self.r_neighbor_index(data,k,neb_index)
        end1 = datetime.datetime.now()
        print(end1 - start)
        # print(self.neighbors(data[2],neb_index,r_neb_index,k))
        for x in dataset:
            q = [j for j in assign if x[3] == j[0]]
            if(q[0][1] == 99):
                if(self.expand_cluster(x,dataset,assign,cluster,k,neb_index,r_neb_index)):
                    cluster+=1
                pass
            pass
        pass
        end = datetime.datetime.now()
        print(end - start)
        self.draw_pic(assign,dataset)
