import gym
import numpy as np
import math
from collections import deque
from tilecoding import TileCoder

env = gym.make("CartPole-v1")
alpha = 0.999 # learn rate
gamma = 0.9999999      # discount fact
epsilon = 1
low = env.observation_space.low
high = env.observation_space.high

# best so far: 2, 5, 8, 10; 3
tiles_per_dim = [10, 10, 10, 10]
lims = [(low[0], high[0]), (low[1], high[1]), (low[2], high[2]), (low[3], high[3])]
tilings = 5

T = TileCoder(tiles_per_dim, lims, tilings)

# size of the thing
print(T.n_tiles)

Q = dict()
Ns = dict()
Na = dict()


def initializeN(state, action):
  disc_state = T[state][0]
  if (disc_state,action) not in Na:
    Na[disc_state, action] = 0
  if disc_state not in Ns:
    Ns[disc_state] = 0
  
# continuous to discrete, create key if not exist
def disc(state):
  disc_state = T[state][0]
  if disc_state not in Q:
    Q[disc_state] = np.zeros(2)
  return disc_state

def get_action(state, epsilon):
  state = disc(state)
  return env.action_space.sample() if np.random.uniform(0,1) <= epsilon else np.argmax(Q[state])

def update_q(state, next_state, action, reward, alpha):
  state = disc(state)
  next_state = disc(next_state)
  Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

n_episodes = 10000
scores = deque(maxlen=100)

for e in range(n_episodes):
  state = env.reset()
  score = 0
  done = False
  
  while not done:
    action = get_action(state, epsilon)
    next_state, reward, done, _ = env.step(action)
  

    score += 1
    state = next_state

    initializeN(state, action)

    Ns[disc(state)] += 1
    Na[disc(state),action] += 1

    alpha = 1. / Na[disc(state),action]**0.8
    epsilon = 1. / Ns[disc(state)]**0.8

    update_q(state, next_state, action, reward * e, alpha)

  
  scores.append(score)
  mean_score = np.mean(scores)

  if e % 400 == 0:
    print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))