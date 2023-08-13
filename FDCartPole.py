import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import namedtuple,deque
import random, math
import os
from itertools import count
import torch.nn.functional as F

epoch = 800
batch_size = 128
PATH = "FDCartPole_model/fdcartpolemodel.pth"
GAMMA = 0.99 # 0.99
LR = 1e-4
TAU = 0.005
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

TrainModel = False # True时程序训练模型，False时程序测试已保存的模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if TrainModel == True: # 训练时不需要显示
    env = gym.make("CartPole-v1") # , render_mode="human"
else:
    env = gym.make("CartPole-v1", render_mode="human")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    """简单的3个线性层"""
    def __init__(self, n_observations=4, actions=2):
        super(DQN, self).__init__()
        self.layer0 = nn.Linear(n_observations, 128)
        self.layer1 = nn.Linear(128, 128)
        self.layer2 = nn.Linear(128, actions)
    
    def forward(self, input):
        input = F.relu(self.layer0(input))
        input = F.relu(self.layer1(input))
        input = self.layer2(input)
        return input

class Replay_Memory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

TrainNet = DQN().to(device)
EvalNet = DQN().to(device)
EvalNet.load_state_dict(TrainNet.state_dict())

optimizer = optim.AdamW(TrainNet.parameters(), lr=LR, amsgrad=True)

replay_memory = Replay_Memory(10000)

explore_steps = 0
def select_action(state, MaxReward=False):
    global EvalNet, device, explore_steps, env
    if MaxReward: # 直接返回最大Reward的action
        with torch.no_grad():
            action = EvalNet(state).max(1)[1].view(1,1)
        return action
    eps_threshold = 0.05 + (0.9 - 0.05) * math.exp(-1. * explore_steps / 1000) # 指数衰减：eps_threshold会越来越小
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * explore_steps / EPS_DECAY) # 指数衰减：eps_threshold会越来越小
    explore_steps = explore_steps + 1
    if random.random() > eps_threshold: # 贪婪策略，选择随机值或者策略最大值
        with torch.no_grad():
            action = EvalNet(state).max(1)[1].view(1,1)
    else:
        action = torch.tensor(random.choice([0, 1]), device=device, dtype=torch.long).view(1,1)
    return action

def optimize_model(batch_size):
    global TrainNet, EvalNet, optimizer, replay_memory
    if len(replay_memory) < batch_size: # 采集的数据还不够
        return 0
    memdata = replay_memory.sample(batch_size)
    batch_state, batch_action, batch_state_next, batch_reward  = zip(*memdata)
    batch_state = torch.cat(batch_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    ## 例程
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_state_next)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch_state_next if s is not None]) # shape: torch.Size([121, 4])、torch.Size([123, 4])...
    ## 自己
    batch_state_next = torch.cat(batch_state_next)
    
    TrainNet.train()
    Trained_scores = TrainNet(batch_state).gather(1, batch_action)
        # Trained_scores = TrainNet(batch_state).gather(1, batch_action) # 待训练模型按照当前state得出评分后，按照当前action选择，Trained_action应该是batch_size个评分浮点数
    Eval_scores = torch.zeros(batch_size, device=device) # 例程代码
    with torch.no_grad():
        ## 自己
        Eval_scores = EvalNet(batch_state_next).max(1)[0] # 评估模型按照state_next得出评分后，选择最大评分返回，所以Eval_scores应该是batch_size个state_next的最大评分
        ## 例程
        # Eval_scores[non_final_mask] = EvalNet(non_final_next_states).max(1)[0]
    
    expected_state_action_values = batch_reward + (Eval_scores * GAMMA) # reward更新函数： reward(s) = reward(s) + gamma*reward(s+1)
    criterion = nn.SmoothL1Loss()
    loss = criterion(expected_state_action_values.unsqueeze(1), Trained_scores)
    optimizer.zero_grad() # 清除当前梯度
    loss.backward()
    torch.nn.utils.clip_grad_value_(TrainNet.parameters(), 100) #  梯度裁剪 drop out
    optimizer.step() # 按照梯度更新TrainNet模型参数
    
    # 这一段让EvalNet参数缓慢向TrainNet靠拢
    train_net_state_dict = TrainNet.state_dict() 
    eval_net_state_dict = EvalNet.state_dict()
    for key in train_net_state_dict:
        eval_net_state_dict[key] = train_net_state_dict[key]*TAU + eval_net_state_dict[key]*(1-TAU)
    EvalNet.load_state_dict(eval_net_state_dict)
    return loss

ModuleComplete = 0
### 主程序 ###
if TrainModel == True:
    for EpochStep in range(epoch):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # unsqueeze(0)在tensor扩展出 0 维度，用于后边torch.cat()整合
        for PlayStep in count():
            # 与环境互动，收集数据
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item()) # tensor是标量时可以用.item()方法取出数据。为矩阵时用.tolist()
            # env.render()
            done = terminated or truncated
            ## 自己
            if terminated :
                reward = torch.tensor(-1.0, dtype=torch.float32, device=device).view(1)
            else:
                reward = torch.tensor(reward, dtype=torch.float32, device=device).view(1)
            state_next = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) # unsqueeze(0)在tensor扩展出 0 维度，用于后边torch.cat()整合
            ## 例程
            # reward = torch.tensor(reward, dtype=torch.float32, device=device).view(1)
            # if terminated:
            #     state_next = None
            # else:
            #     state_next = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) # unsqueeze(0)在tensor扩展出 0 维度，用于后边torch.cat()整合
        
            replay_memory.push(state, action, state_next, reward)
            state = state_next
        
            loss = optimize_model(batch_size) # 优化模型
        
            if done:
                print("epoch={}, play times={}, loss={}".format(EpochStep, PlayStep, loss))
                if(PlayStep == 499):
                    ModuleComplete += 1
                else:
                    ModuleComplete = 0
                if ModuleComplete == 8: # 连续跑满 8 次，保存当前模型
                    torch.save(TrainNet.state_dict(), PATH) # 训练一次后，将模型保存到文件
                    ModuleComplete = 0
                    print("已经8次满分，保存模型")
                break
else: # 模型已经训练好，测试模型
    EvalNet.load_state_dict(torch.load(PATH))
    for EpochStep in range(epoch):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # unsqueeze(0)在tensor扩展出 0 维度，用于后边torch.cat()整合
        for PlayStep in count():
            action = select_action(state, MaxReward=True)
            observation, reward, terminated, truncated, _ = env.step(action.item()) # tensor是标量时可以用.item()方法取出数据。为矩阵时用.tolist()
            state_next = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) # unsqueeze(0)在tensor扩展出 0 维度，用于后边torch.cat()整合
            state = state_next
            done = terminated or truncated
            if done:
                print("epoch={}, play times={}".format(EpochStep, PlayStep))
                break
            

        
        
        





















