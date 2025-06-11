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
class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)

        # Reset gate
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_0=None):
        batch_size, seq_len, _ = x.size()
        x = x.transpose(0, 1)  # (seq_len, batch, input_size)

        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h_0[0]

        hidden_seq = []

        for t in range(seq_len):
            h_t=h_t.detach()  # Detach to prevent backpropagation through the entire sequence
            x_t = x[t]

            z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_t))
            r_t = torch.sigmoid(self.W_r(x_t) + self.U_r(h_t))
            h_hat_t = torch.tanh(self.W_h(x_t) + self.U_h(r_t * h_t))
            h_t = (1 - z_t) * h_t + z_t * h_hat_t

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)

        hidden_seq = hidden_seq.transpose(0, 1)  # back to (batch, seq, hidden)

        return hidden_seq, h_t.unsqueeze(0)
class Actor(nn.Module):
    def __init__(self,action_dim):
        super(Actor,self).__init__()
        self.head=CnnHead()
        self.fc1=nn.Linear(512,512)
        self.fc2=nn.Linear(512,128)
        self.fc3=nn.Linear(128,action_dim)
        self.rnn = CustomGRU(512, 128)
        # self.norm=nn.LayerNorm(512)
        self.activation=nn.ReLU()
        self.sight_size=1
    def forward(self,x,hidden_state=None):
        x=self.head(x)
        bs,epi_len,_=x.shape
        x=self.activation(self.fc1(x))
        x_list=[]
        start_idx=0
        while start_idx<epi_len:
            end_idx=min(start_idx+self.sight_size, epi_len)
            x_,hidden_state=self.rnn(x[:,start_idx:end_idx,:], hidden_state)
            x_list.append(x_)
            hidden_state=hidden_state.detach()
            start_idx=end_idx
        x=torch.cat(x_list, dim=1)
        # x=self.activation(self.fc2(x))
        x=self.fc3(x)
        return x,hidden_state
class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.head=CnnHead()
        self.fc1=nn.Linear(512,512)
        # self.fc2=nn.Linear(512,128)
        self.fc3=nn.Linear(128,1)
        self.rnn = CustomGRU(512, 128)
        # self.norm=nn.LayerNorm(128)
        self.activation=nn.ReLU()
        self.sight_size=1
    def forward(self,x,hidden_state=None):
        x=self.head(x)
        bs,epi_len,_=x.shape
        x=self.activation(self.fc1(x))
        x_list=[]
        start_idx=0
        while start_idx<epi_len:
            end_idx=min(start_idx+self.sight_size, epi_len)
            x_,hidden_state=self.rnn(x[:,start_idx:end_idx,:], hidden_state)
            x_list.append(x_)
            hidden_state=hidden_state.detach()
            start_idx=end_idx
        x=torch.cat(x_list, dim=1)
        # x=self.activation(self.fc2(x))
        x=self.fc3(x)
        return x,hidden_state