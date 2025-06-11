import torch
import torch.nn as nn
import torch.nn.functional as F
class CnnHead(nn.Module):
    def __init__(self):
        super(CnnHead,self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=16, stride=8,padding=4),#160->20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=8, stride=4,padding=2),#20->5
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),#5->2
            nn.ReLU(),
            nn.Flatten(),
        )
        #256
    def forward(self,x):
        shape=x.shape
        x=x.reshape(-1,*shape[-3:]).permute(0,3,1,2)
        x=self.encoder(x)
        x=x.reshape(*shape[:-3],-1)
        return x
class Actor(nn.Module):
    def __init__(self,action_dim):
        super(Actor,self).__init__()
        self.head=CnnHead()
        self.fc1=nn.Linear(512,512)
        self.fc2=nn.Linear(512,128)
        self.fc3=nn.Linear(128,action_dim)
        self.activation=nn.ReLU()
    def forward(self,x):
        x=self.head(x)
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        x=self.fc3(x)
        return x
class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.head=CnnHead()
        self.fc1=nn.Linear(512,512)
        self.fc2=nn.Linear(512,128)
        self.fc3=nn.Linear(128,1)
        # self.norm=nn.LayerNorm(128)
        self.activation=nn.ReLU()
        self.sight_size=1
    def forward(self,x):
        x=self.head(x)
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        x=self.fc3(x)
        return x
