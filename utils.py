import numpy as np
import torch
class CenterReward:
    def __init__(self,gamma,paral_size):
        self.paral_size=paral_size
        self.gamma=gamma
class Cluster:
    def __init__(self,distance_threshold,alpha=0.99):
        self.centers=[]
        self.distance_threshold=distance_threshold
        self.alpha=alpha
    def add(self,coord: np.ndarray):
        #coord: (r,g,b)
        if len(self.centers) == 0:
            self.centers.append(coord)
            return 0
        for i, center in enumerate(self.centers):
            if np.linalg.norm(center - coord,ord=1) < self.distance_threshold:
                self.centers[i] = self.alpha * self.centers[i] + (1 - self.alpha) * coord
                return i
        self.centers.append(coord)
        return len(self.centers) - 1

class DiscreteReward:
    def __init__(self,cell_size,paral_size,distance_threshold,group_size):
        self.paral_size=paral_size
        self.group_size=group_size
        self.group_num=int(paral_size/group_size)
        self.cell_size=cell_size
        self.discrete_num=int(1/cell_size)
        self.map_list=[{} for _ in range(self.group_num)]
        self.visited_list=[{} for _ in range(self.group_num)]
        # self.distance_threshold=distance_threshold
        # self.cluster_list=[Cluster(distance_threshold=distance_threshold) for _ in range(self.group_num)]
    def reset(self,i=None):
        if i is None:
            self.visited_list=[{} for _ in range(self.group_num)]
            # for j in range(self.group_num):
            #     for k,v in self.map_list[j].items():
            #         self.map_list[j][k]=v*0.99
            # self.cluster_list=[Cluster(self.distance_threshold) for _ in range(self.group_num)]
        else:
            self.visited_list[i]={}
            # self.cluster_list[i]=Cluster(self.distance_threshold)
    def update(self,coords,dones,room_ids):
        assert coords.shape[0]==self.paral_size
        assert coords.shape[1]==2
        indexs=np.floor(coords/self.cell_size)
        info_gains=[]
        for j in range(self.paral_size):
            i=j//self.group_size
            # center_index=self.cluster_list[i].add(room_ids[j])
            center_index=int(room_ids[j])
            if center_index not in self.map_list[i]:
                self.map_list[i][center_index]=np.ones((self.discrete_num,self.discrete_num))
            if center_index not in self.visited_list[i]:
                self.visited_list[i][center_index]=np.zeros((self.discrete_num,self.discrete_num))
            map=self.map_list[i][center_index]
            visited_map=self.visited_list[i][center_index]
            h_index=max(int(indexs[j,0]),0)
            w_index=max(int(indexs[j,1]),0) 
            map[h_index,w_index]*=0.999
            visited_map[h_index,w_index]+=1
            info_gain=self.infos_cacu(map[h_index,w_index],visited_map[h_index,w_index])
            info_gains.append(info_gain)
            # if dones[i]:
            #     self.reset(i)
        return np.array(info_gains)
    def infos_cacu(self,value,visited_count):
        # return 1/x**2
        return value if visited_count<= 1 else 0
        # return x

class SplattingReward:
    def __init__(self,dot_size,paral_size,):
        #地图大小为(1,1)
        self.paral_size
class ShortermReplay:
    def __init__(self):
        self.buffer=[]
    def add(self,states,actions,rewards,next_states,dones):
        num_epi=rewards.shape[0]
        #bs epi_len ..
        for i in range(num_epi):
            self.buffer.append((states[i],actions[i],rewards[i],next_states[i],dones[i]))
    def sample(self):
        states,actions,rewards,next_states,dones=zip(*self.buffer)
        self.buffer=[]
        return np.array(states),np.array(actions),np.array(rewards),np.array(next_states),np.array(dones)
    def __len__(self):
        return len(self.buffer)


