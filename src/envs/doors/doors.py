from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
from utils.dict2namedtuple import convert

class Doors(MultiAgentEnv):

    def __init__(self, **kwargs):
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        self.n_doors = self.n_agents = args.n_agents
        self.mishear_prob = args.mishear_prob
        self.comm_error = args.comm_error

        self.state = np.zeros(self.n_doors) # 1 if treasure else lion
        self.obs = np.ones((self.n_agents, self.n_agents)) / self.n_agents
        self.rew_c = args.rew_c
        self.rew_p = args.pen_l
        self.treasure = -1
        self.steps = 0
        self.episode_limit = args.episode_limit

        self.state_is_obs = args.state_is_obs
        self.state_obs_concat = args.state_obs_concat

        self.last_obs = np.zeros((self.n_agents, self.n_agents))
        self.reset()

    #0 = listen, 1 = open door
    def step(self, actions):
        """ Returns reward, terminated, info """
        #print(actions)
        self.steps += 1
        if (actions == 1).any():
            #print("OPENED A DOOR!")
            rew = self.rew_c
            for i in range(self.n_agents):
                if actions[i] and i != self.treasure:
                    rew = -self.rew_c
                    break
            #rew = -self.rew_c * np.sum(actions)
            #rew += 2*self.rew_c * actions[self.treasure]
            #self.obs[:,:] = 0
            #self.obs[:, self.treasure] = 1
            return rew, 1, {}
        else:
            #print("listened")
            if self.steps >= self.episode_limit:
                #self.obs[:,:] = 0
                #self.obs[:, self.treasure] = 1
                return -self.rew_p, 1, {}
            new_sample = self.sample_obs()
            self.obs = new_sample
            #new_obs = self.update_obs(new_sample)
            
            return - self.rew_p, 0, {}



    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.obs[agent_id, :]
    def sample_obs_old(self):
        obs = np.zeros((self.n_agents, self.n_agents))
        miscomm = np.random.choice(2, (self.n_agents, self.n_agents), p = [self.comm_error, 1-self.comm_error])
        for i in range(self.n_agents):
            r = np.random.rand()
            if r < self.mishear_prob:
                obs[:,i] = 1- (i == self.treasure)
            else:
                obs[:, i] = (i == self.treasure)
            miscomm[i,i] = 1
        # print(miscomm)
        # print(obs)
        obs = obs*miscomm + (1-obs)*(1-miscomm)
        # print(obs)
        return obs
    
    def sample_obs(self):
        obs = np.zeros((self.n_agents, self.n_agents))
        #miscomm = np.random.choice(2, (self.n_agents, self.n_agents), p = [self.comm_error, 1-self.comm_error])
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                r = np.random.rand()
                if i == j:
                    if r < self.mishear_prob:
                        obs[i,j] = 1- (j == self.treasure)
                    else:
                        obs[i,j] = (j == self.treasure)
                else:
                    if r < self.comm_error:
                        obs[i,j] = 1- (j == self.treasure)
                    else:
                        obs[i,j] = (j == self.treasure)
        # print(miscomm)
        # print(obs)
        #obs = obs*miscomm + (1-obs)*(1-miscomm)
        # print(obs)
        return obs

    def update_obs(self, new_obs):
        prior = self.obs
        lhood = np.ones((self.n_agents, self.n_agents))

        lh_1_s = 1- self.mishear_prob # prob of 1 given 1 in own door
        lh_0_s = self.mishear_prob # prob of 0 given 1 in own door

        lh_1_o = ((1-self.mishear_prob)*(1-self.comm_error)) + (self.mishear_prob*self.comm_error) # prob of 1 given 1 in other door
        lh_0_o = (self.mishear_prob * (1- self.comm_error)) + ((1-self.mishear_prob)* self.comm_error) # prob of 0 given 1 in other door
        
        # print()
        # print(new_obs)
        # print(prior)
        # print(lh_0_o, lh_0_s, lh_1_o, lh_1_s)
        
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if new_obs[i,j]:
                    if i == j:
                        lhood[i,j] = lh_1_s
                    else:
                        lhood[i,j] = lh_1_o
                else:
                    if i == j:
                        lhood[i,j] = lh_0_s
                    else:
                        lhood[i,j] = lh_0_o

        # print(lhood)
        probs = prior * lhood
        # print(probs)
        probs/=np.sum(probs, axis = 1, keepdims=True)
        # print(probs)
        # input("wait")
        self.obs = probs
        return probs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.n_agents

    def get_state(self):
        if self.state_is_obs:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat
        if self.state_obs_concat:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            s = self.state
            obs_concat = np.concatenate((s, obs_concat), axis=0).astype(np.float32)
            return obs_concat
        return self.state
    
    def get_stats(self):
        pass

    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.state_is_obs:
            return ((self.n_agents**2), )
        elif self.state_obs_concat:
            return ((self.n_agents**2) + self.n_agents, )
        return self.n_agents

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1,1]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return 2

    def reset(self):
        """ Returns initial observations and states"""
        self.state = np.zeros(self.n_doors)
        self.treasure = np.random.randint(0, self.n_doors)
        self.state[self.treasure] = 1
        self.obs = self.sample_obs()#np.ones((self.n_agents, self.n_agents)) / self.n_agents
        self.steps = 0
        return self.get_obs(), self.get_state()

    def render(self):
        raise NotImplementedError

    def close(self):
        print("Closing ENV")
        #raise NotImplementedError

    def seed(self):
        pass
        #raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info



