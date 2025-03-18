from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from .replaybuffer import ReplayBuffer
from .normalization import Normalization
from .normalization import RewardScaling
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime
import time
import torch
import os


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    
class Actor(nn.Module):
    def __init__(self,args,hidden_layers=[64,64]):
        super(Actor, self).__init__()
        self.num_states = args.num_states
        self.num_actions = args.num_actions
        # add in list
        hidden_layers.insert(0,self.num_states)
        hidden_layers.append(self.num_actions)
        print(hidden_layers)

        # create layers
        fc_list = []

        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)
            
            fc_list.append(layer)
            orthogonal_init(fc_list[-1])
        orthogonal_init(fc_list[-1], gain=0.01)

        # put in ModuleList
        self.layers = nn.ModuleList(fc_list)
        self.std = nn.Parameter(torch.zeros(1,self.num_actions))   
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    # when actor(s) will activate the function 
    def forward(self,s):
        for i in range(len(self.layers)-1):
            s = self.tanh(self.layers[i](s))
        s = self.layers[-1](s)

        action_probability = self.softmax(s)
        return action_probability

class Critic(nn.Module):
    def __init__(self, args,hidden_layers=[64,64]):
        super(Critic, self).__init__()
        self.num_states = args.num_states
        # add in list
        hidden_layers.insert(0,self.num_states)
        hidden_layers.append(1)
        print(hidden_layers)

        # create layers
        fc_list = []

        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)
            
            fc_list.append(layer)
            orthogonal_init(fc_list[-1])
        # put in ModuleList
        self.layers = nn.ModuleList(fc_list)
        self.tanh = nn.Tanh()
        
    def forward(self,s):
        
        for i in range(len(self.layers)-1):
            s = self.tanh(self.layers[i](s))
        v_s = self.layers[-1](s)
        return v_s

class Agent():
    def __init__(self,args,hidden_layer_num_list=[64,64]):
        self.actor = Actor(args,hidden_layer_num_list.copy())
        self.critic = Critic(args,hidden_layer_num_list.copy())
        
        self.mini_batch_size = args.mini_batch_size
        self.entropy_coef = args.entropy_coef
        self.num_actions = args.num_actions
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.epochs = args.epochs
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.lr = args.lr
        
        self.max_train_steps = args.max_train_steps
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        
    def evaluate(self , state):
        # numpy convert to tensor.[num_state]->[1,num_state]
        s = torch.unsqueeze(torch.tensor(state, dtype=torch.float),0)
        # if don't use detach() the requires_grad = True . Can't call numpy()
        # or able use torch.no_grad
        with torch.no_grad():
            action_probability = self.actor(s).numpy().flatten()
            action = np.random.choice(self.num_actions, p=action_probability)

        return action

    # actor forward. and choose one action from output(probability)
    def choose_action(self,state):
        s = torch.unsqueeze(torch.tensor(state, dtype=torch.float),0)
        with torch.no_grad():
            action_probability = self.actor(s).numpy().flatten()
            action = np.random.choice(self.num_actions, p=action_probability)
            a_logprob = np.log(action_probability[action])  # The log probability density of the action
        
        return action, a_logprob # 

    def generalized_advantage_estimation(self, vs , vs_ , r , dw , done):
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            # advantage normalization
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        return v_target,adv
    
    def evaluate_policy(self,args,env,render=False):
        times = 3
        evaluate_reward = 0
        print("evaluate_policy")
        for i in range(times):
            s = env.reset()[0]
            done = False
            episode_reward = 0
            episode_steps = 0
            while True:
                #if render:
                    #time.sleep(0.08)
                
                a = self.evaluate(s)  # We use the deterministic policy during the evaluating
            
                s_, r, done, truncted,_ = env.step(a)
                if episode_steps == 0 and done:
                    s = env.reset()[0]
                    continue
                episode_reward += r
                s = s_

                if truncted or done:
                    break
                episode_steps += 1
            evaluate_reward += episode_reward

        return evaluate_reward / times

    def train(self,args,env,env_name):
        
        evaluate_num = 0  # Record the number of evaluations
        evaluate_rewards = []  # Record the rewards during the evaluating
        total_steps = 0  # Record the total steps during the training
        training_count = 0

        replay_buffer = ReplayBuffer(args)

        home_directory = os.path.expanduser( '~' )
        log_dir=home_directory+'/Log/PPO_'+env_name+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print("Log dir : ",log_dir)

        writer = SummaryWriter(log_dir=log_dir)
        
        while total_steps < args.max_train_steps:
            s = env.reset()[0]
            episode_steps = 0
            done = False

            while True:                
                a, a_logprob = self.choose_action(s)  # Action and the corresponding log probability    
                s_, r, done, truncated,_ = env.step(a)
                if episode_steps == 0 and done:
                    s = env.reset()[0]
                    continue

                # done|truncted meaning the game have been truncated whatever win or loss
                replay_buffer.store(s, a, a_logprob, r, s_, done, (done | truncated))
                s = s_
                total_steps += 1

                

                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer.count == args.batch_size:
                    self.update(replay_buffer, total_steps)
                    replay_buffer.count = 0
                    print("Training epoch :",training_count,"\tStep:",total_steps,"/",args.max_train_steps,"\t")
                    training_count += 1

                # Evaluate the policy every 'evaluate_freq' steps
                if total_steps % args.evaluate_freq_steps == 0:
                    evaluate_num += 1
                    evaluate_reward = self.evaluate_policy(args, env)
                    evaluate_rewards.append(evaluate_reward)
                    evaluate_average_reward = np.mean(evaluate_rewards[-50:])
                    print("evaluate_num:{} \t evaluate_reward:{} \t average_reward:{}".format(evaluate_num, evaluate_reward,evaluate_average_reward))
                    writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)

                if truncated or done:
                    break
                episode_steps += 1
                
    def update(self,replay_buffer,total_steps):
        s, a, old_log_prob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data .type is tensor
        a = F.one_hot(a, num_classes=self.num_actions)
        vs = self.critic(s)
        vs_ = self.critic(s_)
        
        v_target , adv = self.generalized_advantage_estimation(vs,vs_,r,dw,done)

        for i in range(self.epochs):
            for j in range(self.mini_batch_size//2):
                index = np.random.choice(self.batch_size,self.mini_batch_size,replace=False)

                # get the current distribution of actor
                new_prob = self.actor(s[index])
                # get entropy of actor distribution
                prob_entropy = -(new_prob * torch.log(new_prob + 1e-10))
                # get the new log probability
                new_prob = (new_prob * a[index]).sum(1 , keepdim=True)

                new_log_prob = torch.log(new_prob)
                # shape = [mini_batch_size , num_action]. Summation over the axis=1 -> [1,num_action]                
                ratios = torch.exp(new_log_prob - old_log_prob[index])  # shape(mini_batch_size X 1)
                # adv.shape = [mini_batch_size,1]
                p1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                # clip ratios to 1-epsilon ~ 1+epsilon
                p2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                # choice the minimum value of p1 or p2. 
                actor_loss = -torch.min(p1, p2) - self.entropy_coef * prob_entropy.mean(dim=1)  # Trick 5: policy entropy

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        self.lr_decay(total_steps=total_steps)
        
    def save_actor_model(self,path):
        torch.save(self.actor, path)
    def save_critic_model(self,path):
        torch.save(self.critic, path)
    def load_actor_model(self,path):
        self.actor = torch.load(path).train()
    def load_critic_model(self,path):
        self.critic = torch.load(path).train()

    def lr_decay(self, total_steps):
        lr_a_now = self.lr * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr * (1 - total_steps / self.max_train_steps)
        for opt in self.optimizer_actor.param_groups:
            opt['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            opt['lr'] = lr_c_now
    
