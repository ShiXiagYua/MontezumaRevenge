import gym
from moviepy.editor import ImageSequenceClip
import os
import cv2
import numpy as np
from multiprocessing import Process, Pipe
import copy

def resize_image(img: np.ndarray, target_size: tuple, interpolation=cv2.INTER_AREA) -> np.ndarray:
    resized = cv2.resize(img, target_size, interpolation=interpolation)
    return resized
class EnvWrapper:
    def __init__(self,save_video=False,seed=None,exp_id=None):
        self.seed=seed
        self.save_video=save_video
        self.width=160
        self.num_steps=0
        self.max_steps=4000
        self.history_len=1
        self.history_obs=[]
        if save_video:
            self.env = gym.make("ALE/MontezumaRevenge-v5",render_mode='rgb_array')
            video_dir='videos'
            os.makedirs(video_dir,exist_ok=True)
            if exp_id==None:
                exist_exp_id=[int(exp) for exp in os.listdir(video_dir)]
                if len(exist_exp_id)>0:
                    exist_exp_id=sorted(exist_exp_id)
                    exp_id=exist_exp_id[-1]+1
                else:
                    exp_id=0
            self.exp_dir=video_dir+'/%d'%exp_id
            os.makedirs(self.exp_dir,exist_ok=True)
            self.frames=[]
            self.video_id=0
        else:
            self.env = gym.make("ALE/MontezumaRevenge-v5")
        self.ale=self.env.unwrapped.ale
    def reset(self):
        self.num_steps=0
        obs,info=self.env.reset(seed=self.seed)
        self.history_obs=[copy.deepcopy(obs) for _ in range(self.history_len)]
        if self.save_video:
            self.frames=[]
            self.frames.append(obs)
        obs,coords,room_id=self.process_obs(obs)
        self.history_obs=[copy.deepcopy(obs) for _ in range(self.history_len)]
        return np.concatenate(self.history_obs,axis=-1),coords,room_id
    def sample_action(self):
        return self.env.action_space.sample()
    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        if reward>0:
            reward=1.0
        self.num_steps+=1
        if self.save_video:
            self.frames.append(obs)
        done= terminated or truncated or self.num_steps>=self.max_steps
        if done:
            if self.save_video:
                self.generate_video()
            obs,coords,room_id=self.reset()
            return obs,reward,done,coords,room_id
        obs,coords,room_id=self.process_obs(obs)
        self.history_obs.pop(0)
        self.history_obs.append(obs)
        return np.concatenate(self.history_obs,axis=-1),reward,done,coords,room_id
    def generate_video(self):
        clip=ImageSequenceClip(self.frames,fps=30)
        video_path=self.exp_dir+'/%d.mp4'%self.video_id
        clip.write_videofile(video_path,fps=30)
        self.video_id+=1
    def process_obs(self,obs):
        img=obs[20:198,:,:]
        img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img=resize_image(img,(self.width,self.width))
        coords=self.human_detect(img)
        # mean_color=np.mean(img,axis=(0,1))
        room_id=self.ale.getRAM()[3]
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=img[...,None]
        # return img,coords,mean_color
        return img,coords,room_id
    def human_detect(self,img):
        low_bound=np.array([60,60,190])
        high_bound=np.array([80,80,215])
        img_mask=np.all((img>=low_bound)&(img<=high_bound),axis=-1)
        coords = np.argwhere(img_mask)
        if coords.size==0:
            return np.array([-1,-1])
        else:
            y_mean, x_mean = coords.mean(axis=0)
            return np.array([y_mean/self.width,x_mean/self.width])
def worker(remote, parent_remote,env_fn_wrapper,exp_id):
    parent_remote.close()
    env = env_fn_wrapper(exp_id=exp_id)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, coords,mean_colors = env.step(data)
            remote.send((obs, reward, done,coords,mean_colors))
        elif cmd == 'reset':
            obs,coords, mean_colors=env.reset()
            remote.send((obs,coords,mean_colors))
        elif cmd == 'close':
            remote.close()
            break

class ParallelEnv:
    def __init__(self, n_envs,exp_id=None):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, EnvWrapper ,exp_id))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
    def step(self, actions):
        #n_env n_agent
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, coords,mean_colors = zip(*results)
        #n_env n_agent d
        return np.array(obs), np.array(rewards), np.array(dones), np.array(coords),np.array(mean_colors)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results=[remote.recv() for remote in self.remotes]
        obs,coords,mean_colors=zip(*results)
        return np.array(obs),np.array(coords),np.array(mean_colors)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()