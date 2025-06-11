import numpy as np
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
@torch.no_grad
def compute_advantage(gamma, lmbda, td_delta,done):
    #td_delta done : bs epi_len 1
    results=[]
    temp=0.0
    bs,epi_len,_=td_delta.shape
    for i in reversed(range(epi_len)):
        temp=td_delta[:,i,:]+gamma*lmbda*(1-done[:,i,:])*temp
        results.append(temp)
    results.reverse()
    return torch.stack(results,dim=1)
def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)
class PPO:
    ''' PPO算法,采用截断方式 '''
    #规定输入为np.array
    def __init__(self, actor, critic, actor_lr, critic_lr,gamma,
                 lmbda, entropy_coef,epochs, eps, device):
        self.actor =actor.to(device)#返回概率，要能处理高于bs的维度
        self.critic = critic.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.entropy_coef=entropy_coef
        self.hidden_state=None
    def reward_normalize(self,x):
        return x
        # return torch.sign(x)*torch.log(x.abs()+1)
    def tdv(self,x,dtype=None):
        if dtype==None:
            dtype=torch.float32
        return torch.tensor(np.array(x),dtype=dtype).to(self.device)
    @torch.no_grad
    def take_action(self, state):
        #state: num_env dim
        state = self.tdv(state).unsqueeze(1)#num_env 1 dim
        logits,self.hidden_state = self.actor(state)#num_env action_dim
        action_dist = torch.distributions.Categorical(logits=logits.squeeze(1))
        action = action_dist.sample()#num_env
        return action.cpu().numpy() 
    def save(self,prefix):
        os.makedirs("models",exist_ok=True)
        torch.save(self.actor.state_dict(),"models/"+str(prefix)+"_actor.pth")
        torch.save(self.critic.state_dict(),"models/"+str(prefix)+"_critic.pth")
    def load(self,prefix):
        self.actor.load_state_dict(torch.load("models/"+str(prefix)+"_actor.pth"))
        self.critic.load_state_dict(torch.load("models/"+str(prefix)+"_critic.pth"))
    def cal_value_loss(self, old_values, new_values, targets):
        value_clipped = old_values + (new_values - old_values).clamp(-0.2, 0.2)
        error_clipped = targets - value_clipped
        error_original = targets - new_values

        value_loss_clipped = huber_loss(error_clipped, 10.0)
        value_loss_original = huber_loss(error_original, 10.0)

        value_loss = torch.max(value_loss_original, value_loss_clipped)
        value_loss = value_loss.mean()
        return value_loss
    def update(self, states,actions,rewards,next_states,dones):
        #bs epi_len      
        #这里不能把bs epi_len融为一个维度，因为epi_len是规定的回合长度，不是回合结束长度，最后一帧没有done信号
        #这里还输入了next_states，更高效的实现是states保护epi_len+1帧。但为了实现简洁，增加计算量
        with torch.no_grad():
            states = self.tdv(states)# bs epi_len state_dims(h,w,3)
            actions = self.tdv(actions,torch.long).unsqueeze(-1)#bs epi_len 1
            rewards = self.tdv(rewards).unsqueeze(-1)#bs epi_len 1
            rewards = self.reward_normalize(rewards)
            next_states = self.tdv(next_states)#bs epi_len state_dims(h,w,3)
            dones = self.tdv(dones).unsqueeze(-1)#bs epi_len 1

            #这里暂时没实现critic近端约束
            #就在这里有重复计算
            old_values,_=self.critic(states)
            old_next_values,_=self.critic(next_states)
            td_target = rewards + self.gamma * old_next_values * (1 -dones)#bs epi_len 1
            td_delta = td_target - old_values
            advantage = compute_advantage(self.gamma, self.lmbda,td_delta,dones)#bs epi_len 1
            # td_target=advantage+old_values
            old_logits ,_= self.actor(states)
            old_dist=torch.distributions.Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)  # bs epi_len 1
            old_log_probs=old_log_probs.detach()
            old_values=old_values.detach()
            td_target=td_target.detach()
            advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)  
            advantage=advantage.detach()
        total_actor_loss,total_critic_loss,total_entropy,total_ratio=0.0,0.0,0.0,0.0
        for _ in range(self.epochs):
            logits,_=self.actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy().mean()
            log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1) 
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            # critic_loss = self.cal_value_loss(old_values,self.critic(states),td_target)
            critic_loss=F.mse_loss(self.critic(states)[0],td_target)
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            (actor_loss-self.entropy_coef*entropy).backward()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            total_actor_loss+=actor_loss.item()
            total_critic_loss+=critic_loss.item()
            total_entropy+=entropy.item()
            total_ratio+=ratio.mean().item()
        return total_actor_loss/self.epochs,total_critic_loss/self.epochs,total_entropy/self.epochs
            