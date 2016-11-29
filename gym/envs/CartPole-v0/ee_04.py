#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:ts=4:et:ai:


import gym
import random
import math
import numpy as np
import sys
import time


LENGTH = 4  # Observation length
MAX_WIN_STEPS = 200
WIN_THRESHOLD = 195
MU = 0
TRIES = 1
RENDER = False
MAX_EPISODES = 100000
MAX_EPI_LEN = 200
STAGNATION_THRESHOLD = 20
EPS = 0.001
N = 2
T1 = 1.0 / math.sqrt(2 * N)
T2 = 1.0 / math.sqrt(2 / math.sqrt(N))
ALPHA = 0.6
GAMMA = 1.0
MONITOR = True

perturb_sigma = np.vectorize(lambda x: x * math.e ** (T1 * random.gauss(0, 1) +
                                                      T2 * random.gauss(0, 1)))
sigmoid = lambda x: 1 / (1 + math.e ** -x)

perturb_coeff = np.vectorize(lambda x: x * random.gauss(0, 1))
I_EP = 0
ENV = gym.make('CartPole-v0')
HISTORY = [] 


class Individual:
    """An evolutionary individual (solution)
    """
    def __init__(self, tries=10, render=False):
        assert isinstance(tries, int) and tries > 0
        self.coef = np.array([random.uniform(-1, 1) for _ in range(LENGTH)])
        self.sigma = np.array([random.uniform(0, 1) for _ in range(LENGTH)])
        self.sigma[self.sigma < EPS] = EPS
        self.render = render
        self.tries = tries

    def get_action(self, obs):
        ac = sum(self.coef * sigmoid(obs))
        return 0 if ac < 0.5 else 1

    @property
    def fitness(self):
        """Fitness for this individual es the sum of total rewards"""
        global I_EP, HISTORY
        acc = 0.0

        for i_ep in range(self.tries):
            obs = ENV.reset()
            total_rew = 0
            I_EP += 1
            for t in range(MAX_EPI_LEN):
                if self.render:
                    ENV.render()
                action = self.get_action(obs)
                obs, reward, done, info = ENV.step(action)
                total_rew += reward
                if done:
                    break
            HISTORY.append(total_rew)
            HISTORY = HISTORY[-MAX_WIN_STEPS:]
            acc += total_rew

        return acc / self.tries

    def clone(self):
        result = Individual(tries=self.tries, render=self.render)
        result.coef = np.array(self.coef)
        result.sigma = np.array(self.sigma)
        return result

    def mutate(self):
        self.sigma = perturb_sigma(self.sigma)
        self.sigma[self.sigma < EPS] = EPS
        self.coef += perturb_coeff(self.sigma)

    def crossover(self, other):
        result = self.clone()
        result.coef = ALPHA * other.coef + (1 - ALPHA) * self.coef
        result.sigma = ALPHA * other.sigma + (1 - ALPHA) * self.sigma
        return result

    def __repr__(self):
        return str(self.coef) + ' s:' + str(self.sigma)



if __name__ == '__main__':
    if MONITOR:
        ENV.monitor.start('./tmp/cartpole-experiment-3')

    best = Individual(tries=TRIES, render=RENDER)
    nn_f = best_f = best.fitness
    print(best, best_f)
    cnt = 0

    I_EP = 0
    while I_EP < MAX_EPISODES:
        if cnt > STAGNATION_THRESHOLD:
            nn = Individual(tries=TRIES, render=RENDER)
        else:
            nn = best.clone()

        if nn_f < WIN_THRESHOLD * GAMMA:
            nn.mutate()
            cnt += 1

        nn_f = nn.fitness

        if nn_f > best_f or nn_f >= WIN_THRESHOLD:
            best_f = nn_f
            best = best.crossover(nn)
            print("Episode: %i %3.2f %f\n" % (I_EP, best_f, np.array(HISTORY).mean()), best)
            cnt = 0

        if len(HISTORY) == MAX_WIN_STEPS:
            if np.array(HISTORY).mean() >= WIN_THRESHOLD:
                break

    if MONITOR:
        ENV.monitor.close()
    print('Episodes: {}, reward: {}, avg: {}'.format(I_EP, best_f, np.array(HISTORY).mean()))
