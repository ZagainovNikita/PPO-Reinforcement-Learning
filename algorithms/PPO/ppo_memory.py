import torch
import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
        
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
    def generate_batches(self):
        n_samples = len(self.states)
        batch_starts = np.arange(0, n_samples, self.batch_size)
        indices = np.arange(0, n_samples)
        np.random.shuffle(indices)
        batches = [indices[start:start+self.batch_size] for start in batch_starts]
        
        return (np.array(self.states), np.array(self.probs), np.array(self.vals),
                np.array(self.actions), np.array(self.rewards), np.array(self.dones),
                batches)
        
    def store(self, state, prob, val, action, reward, done):
        self.states.append(state)
        self.probs.append(prob)
        self.vals.append(val)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
