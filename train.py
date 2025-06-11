import os
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from agent import PPO
from utils import DiscreteReward,ShortermReplay
from models import Actor,Critic
from env import EnvWrapper,ParallelEnv
import numpy as np
def train_on_policy_agent(env, eval_env,agent, intri_rewarder,intri_coef,replay,episode_len,num_episodes,eval_every,batch_size,exp_id):#这里batch_size是指回合数，且必须为num_env的整数倍
    #如果提供exp_id则用，否则自动加1
    #提供控制台打印和tensorboard 两种log
    os.makedirs('logs',exist_ok=True)
    if exp_id==None:
        exist_exp_id=[int(exp) for exp in os.listdir("logs")]
        if len(exist_exp_id)>0:
            exist_exp_id=sorted(exist_exp_id)
            exp_id=exist_exp_id[-1]+1
        else:
            exp_id=0
    exp_dir="logs/%d"%exp_id
    os.makedirs(exp_dir,exist_ok=True)
    writer=SummaryWriter(exp_dir)
    return_list = []
    best_return = 4.0
    intri_list=[]
    global_epi_num=0
    global_a_loss=0.0
    global_c_loss=0.0
    global_e_loss=0.0
    num_update=0
    state,coord,room_id=env.reset()
    for i_epoch in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i_epoch) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                #rollout
                #episode_len 可能包含很多回合
                episode_return = 0.0
                episode_intri=0.0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []} 
                if i_episode%10==0:
                    intri_rewarder.reset()
                    state,coord,room_id=env.reset()
                for i in range(episode_len):
                    action = agent.take_action(state)
                    next_state, reward, done,coord,room_id = env.step(action)
                    intri_reward=intri_rewarder.update(coord,done,room_id)
                    com_reward=reward+intri_coef*intri_reward
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(com_reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward.mean()
                    episode_intri+= intri_reward.mean()
                
                #update
                assert len(replay)<=batch_size
                for key,value in transition_dict.items():
                    transition_dict[key]=np.array(value).swapaxes(0,1)
                replay.add(transition_dict["states"],transition_dict["actions"],transition_dict['rewards'],transition_dict["next_states"],transition_dict["dones"])
                if len(replay)==batch_size:
                    states,actions,rewards,next_states,dones=replay.sample()
                    a_loss,c_loss,e_loss=agent.update(states,actions,rewards,next_states,dones)

                #log
                    writer.add_scalar('actor_loss',a_loss,global_epi_num)
                    writer.add_scalar('critic_loss',c_loss,global_epi_num)
                    writer.add_scalar('entropy_loss',e_loss,global_epi_num)
                    num_update+=1
                    ratio=0.8
                    global_a_loss=ratio*global_a_loss+(1-ratio)*a_loss
                    global_c_loss=ratio*global_c_loss+(1-ratio)*c_loss
                    global_e_loss=ratio*global_e_loss+(1-ratio)*e_loss
                    transition_dict ={'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                writer.add_scalar('train_reward',episode_return,global_epi_num)
                writer.add_scalar('intri_reward',episode_intri,global_epi_num)
                return_list.append(episode_return)
                intri_list.append(episode_intri)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (global_epi_num),
                                      'return': '%.3f' % np.mean(return_list[-10:]),'intri': '%.3f' % np.mean(intri_list[-10:]),
                                      'actor_loss':'%.3f'%global_a_loss,'critic_loss':'%.3f'%global_c_loss,'entropy':'%.3f'%global_e_loss})
                pbar.update(1)
                #eval
                if global_epi_num%eval_every==0:
                    tqdm.write('evaluation')
                    state_,_,_=eval_env.reset()
                    total_reward_=0.0
                    t=0
                    while True:
                        action=agent.take_action([state_])
                        state_,reward_,done_,_,_=eval_env.step(action[0])
                        total_reward_+=reward_
                        t+=1
                        if done_ or t>4000:
                            break
                    tqdm.write('reward: %3.f, length %d'%(total_reward_,t))
                    writer.add_scalar('eval_reward',total_reward_,global_epi_num)
                    writer.add_scalar('eval_length',t,global_epi_num)
                    if total_reward_>best_return:
                        best_return=total_reward_
                        agent.save('best_checkpoint_%d'%int(best_return))
                global_epi_num+=1
                #save 默认保留10个检测点
        agent.save(i_epoch)
    return return_list
#exp
exp_id=None
num_episodes=5000
episode_len=512
eval_every=30
batch_size=64
#env
n_envs=64
group_size=1
#model
action_dim=18

#agent
actor_lr=2e-4
critic_lr=2e-4
gamma=0.98
lmbda=0.95
entropy_coef=0.01
epochs=10
eps=0.15
device=torch.device('cuda:7')



#intrinstic reward
cell_size=0.05
paral_size=n_envs
intri_coef=0.05
distance_threshold=30.0

#replay

if __name__ =='__main__':
    env=ParallelEnv(n_envs,exp_id)
    eval_env=EnvWrapper(save_video=True,exp_id=exp_id)

    actor=Actor(action_dim)
    critic=Critic()

    agent=PPO(actor, critic, actor_lr, critic_lr,gamma,lmbda, entropy_coef,epochs, eps, device)

    intri_rewarder=DiscreteReward(cell_size,paral_size,distance_threshold,group_size)

    replay=ShortermReplay()

    train_on_policy_agent(env, eval_env,agent, intri_rewarder,intri_coef,replay,episode_len,num_episodes,eval_every,batch_size,exp_id)