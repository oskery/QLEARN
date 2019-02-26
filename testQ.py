import gym
import numpy as np
import math
from collections import deque
from tilecoding import TileCoder

env = gym.make("CartPole-v1")
alpha = 0.5        # learn rate
gamma = 0.99       # discount fact
epsilon = 0.2      # explore rate
low = env.observation_space.low
high = env.observation_space.high

tiles_per_dim = [0, 0, 10, 13]
lims = [(low[0], high[0]), (low[1], high[1]), (low[2], high[2]), (low[3], high[3])]
tilings = 7

T = TileCoder(tiles_per_dim, lims, tilings)

# size of the thing
print(T.n_tiles)

Q = dict()

def get_action(state):
  if T[state][0] not in Q:
    Q[T[state][0]] = np.zeros(2)
  return env.action_space.sample() if np.random.random() <= epsilon else np.argmax(Q[T[state][0]])

def update_q(state, next_state, action, reward):
  if T[state][0] not in Q:
    Q[T[state][0]] = np.zeros(2)
  if T[next_state][0] not in Q:
    Q[T[next_state][0]] = np.zeros(2)
  Q[T[state][0]][action] += alpha * (reward + gamma * np.max(Q[T[next_state][0]]) - Q[T[state][0]][action])

n_episodes = 10000
best_score = 0
scores = deque(maxlen=100)

for e in range(n_episodes):
  state = env.reset()
  score = 0
  done = False
  alpha = max(0.01, alpha - alpha * 0.0004)
  epsilon = max(0.01, epsilon - epsilon * 0.0001)

  while not done:
    action = get_action(state)
    next_state, reward, done, _ = env.step(action)
    update_q(state, next_state, action, reward)
    score += 1
    state = next_state
    #env.render()
  
  scores.append(score)
  mean_score = np.mean(scores)
  if mean_score >= 195 and e >= 100:
    print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))

  if score > best_score: 
    best_score = score
    print("Episode: {}, new best: {:.2f}".format(e, best_score))