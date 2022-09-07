import pulp
import random
import networkx as nx
import simpy
import numpy as np
import pickle
import random
import math
import operator
import itertools
import pandas as pd
import decimal
import time

from copy import *
from geopy import distance
from scipy import optimize
from sklearn.cluster import KMeans
from scipy.spatial.distance import *
from scipy import stats
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import silhouette_score, calinski_harabasz_score

np.printoptions(precision = 2)


def find_queue_probability(b_hc, M, mus, mode):

    L = []
    for i in range(b_hc):
        if mode == 0:
            lambdas = M[i, T]
        else:
            lambdas = np.mean(M[i, T - int(window): T])

        rho = float(lambdas) / float(mus)

        if rho <= 1:
            p0 = 1.0 - rho
            p1 = p0 * rho
            pq = 1.0 - (p0 + p1)

        else:
            pq = 1.0

        L.append(pq)

    mean_pq = np.mean(L)
    return mean_pq


def find_reward(velocity, mean_pq):

    global v
    return float(velocity)/(max(v)) * math.exp(- mean_pq)


class Node(object):

    def __init__(self, env, ID, velocity, dense, Z, pop, b_hc, Q):
        global T, Duration

        self.ID = ID
        self.env = env
        self.z_total = Z

        self.velocity = velocity
        self.dense = dense
        self.pop = pop
        self.Q = Q

        # Number of ICUs 
        self.b_hc = int(b_hc)
        self.bed_queue = {j: [] for j in range(self.b_hc)}

        self.last_state = (0, 0)
        self.current_state = (0, 0)
        self.next_state = None
        self.reward = 0

        self.beta = 1.0
        self.last_mean_pq = None

        # Arrival rates per bed
        self.M = np.zeros((self.b_hc, Duration + 1))

        if self.ID == 1:
            self.env.process(self.time_increment())

        self.env.process(self.opt())
        self.env.process(self.inject_new_infection())
        self.env.process(self.treatment())
        self.env.process(self.learn())

    def time_increment(self):

        global T, iho, epsilon, decay, window, x, yR, yV, v, Z_list, T_list, eB, state_size

        while True:

            T = T + 1
            # if T % iho == 0:
              # print (T, epsilon)

            if T > 0 and T % window == 0:
              epsilon *= decay

            if T % 10 == 0:
              Z_list.append([entities[b].z_total[3] for b in range(eB)]) #CEK
              T_list.append(T)

            if T > 0 and T % (100 * 24) == 0:
              # epsilon = 0.6
              self.Q = np.zeros((state_size, state_size))

            yield self.env.timeout(minimumWaitingTime)

    def treatment(self):

        global T, iho, epsilon, decay, window, recovery_time, hospital_recover

        while True:

            for i in range(self.b_hc):
                if len(self.bed_queue[i]) == 0:
                    continue

                t = self.bed_queue[i][0]
                if t - T > recovery_time:
                    self.bed_queue[i].pop(0)
                    if random.uniform(0, 1) < hospital_recover:
                        self.z_total = [self.z_total[0], self.z_total[1], self.z_total[2] - 1,
                                        self.z_total[3], self.z_total[4] + 1, self.z_total[5], 
                                        self.z_total[6]]
                    else:
                        self.z_total = [self.z_total[0], self.z_total[1], self.z_total[2] - 1,
                                        self.z_total[3], self.z_total[4], self.z_total[5] + 1,
                                        self.z_total[6]]

            yield self.env.timeout(minimumWaitingTime)

    def inject_new_infection(self):

        global pI, cI

        while True:
            if T % fI == 0:
                n = float(cI) / float(T + 1) * 5
                self.z_total = [self.z_total[0], self.z_total[1], self.z_total[2] + (track * n), 
                                self.z_total[3] + ((1 - track) * n), self.z_total[4], 
                                self.z_total[5], self.z_total[6]]

            yield self.env.timeout(minimumWaitingTime)

    def opt(self):

        global iho, sigma, gamma, alphas, track, rho, p, pH, DHP
        while True:

            if T % iho == 0:
                z0 = deepcopy(self.z_total)

                self.beta = p * math.sqrt(2) * math.pi * self.velocity * (self.dense * math.pow(10, -6)) * 1

                z0[0] = z0[0] - (self.beta * z0[0] * z0[3]) / self.pop #S
                
                new_infected = sigma * track * z0[1]
                z0[1] = z0[1] + (self.beta * z0[0] * z0[3]) / self.pop - new_infected #E
                z0[2] = z0[2] + new_infected - gamma * z0[2] #I_r
                z0[3] = z0[3] + sigma * (1 - track) * z0[1] - gamma * z0[3] #I_u
                z0[4] = z0[4] + gamma * (1.0 - alphas) * z0[2] #R
                z0[5] = z0[5] + gamma * alphas * z0[2] #D
                z0[6] = z0[6] + gamma * z0[3] #Re

                self.z_total = deepcopy(z0)

                # Number of patients hospitalized
                nH = int(pH * new_infected)

                # Empty beds
                empty_beds = []
                for j in range(self.b_hc):
                    if len(self.bed_queue[j]) == 0:
                        empty_beds.append(j)

                # Assign random hospital beds to patients if no beds are empty
                for i in range(nH):
                    if len(empty_beds) > 0:
                        bed = empty_beds.pop(0)
                    else:
                        bed = int(random.uniform(0, self.b_hc - 1))

                    self.M[bed, T] += 1
                    self.bed_queue[bed].append(T)

                # Find mean probability of queue (pq)
                mean_pq = find_queue_probability(self.b_hc, self.M, mus, 0)
                DHP[self.ID].append((nH, mean_pq, T))

                self.reward = find_reward(self.velocity, mean_pq)

            yield self.env.timeout(minimumWaitingTime)

    def learn(self):
        global epsilon, last_state, current_state, state_size, lr, gamma_RL, mus, window, v, capacity, reward, yV, thr

        while True:
            if T > 10 and T % window == 0:

                mean_pq = find_queue_probability(self.b_hc, self.M, mus, 1)
                self.reward = find_reward(self.velocity, mean_pq)

                if self.last_mean_pq is not None and abs(self.last_mean_pq - mean_pq) < thr:
                    self.last_mean_pq = mean_pq
                else:

                    # Find capacity index
                    cp = None
                    for cp in range(len(capacity)):
                        if mean_pq >= capacity[cp]:
                            break

                    self.current_state = (cp, v.index(self.velocity))

                    if random.uniform(0, 1) < epsilon:
                        self.velocity = v[random.choice([i for i in range(state_size)])]
                    else:
                        self.velocity = v[np.argmax(self.Q[cp])]

                    if self.ID == 4:
                        yV.append(self.velocity)
                        yR.append(mean_pq)
                        x.append(T)

                    self.last_mean_pq = mean_pq

                    self.Q[self.current_state] += lr * (self.reward + gamma_RL * np.max(self.Q[self.current_state]) - self.Q[self.last_state])
                    self.last_state = self.current_state

                    print(self.Q)

            yield self.env.timeout(minimumWaitingTime)


# Number of boroughs
eB = 5

# Interact how often in hours
iho = 6

# Simulation time in hours
Duration = 24 * 60

# Fraction of patients needing hospitalization
pH = 0.4

# tracing rate
fr = 90 # proportion of reported
fu = 10 # proportion of unreported
track = fr / (fr + fu) # tracing rate

# SEIRD parameters
sigma, gamma, alphas = 0.25, 0.5, 0.05

# Minimum waiting time
minimumWaitingTime = 1

B = {0: 'Bantul', 1: 'Gunung Kidul', 2: 'Kulon Progo', 3: 'Sleman', 4: 'Yogyakarta'}

# Real trends in infection in NYC
actual_inf = [1932, 1461, 1510, 1106, 507, 398, 279, 212, 111, 105, 71] # inf cumulative
actual_inf = [float(v)/float(eB) for v in actual_inf]
unrep_inf = [float(fu * v) / float(fr) for v in actual_inf] # inf unreported

# Population
P = [1418207, 2559903, 1628706, 2253858, 476143]
# Infected
Ir = [actual_inf[0] for i in range(eB)]
Iu = [unrep_inf[0] for i in range(eB)]

# Death
D = [0, 0, 0, 0, 0]

# Beta parameters
# Density
density = [1900, 500, 740, 2000, 11000]
# Infection rate
p = 0.01

# Proportion of exposed
pe = 4.07031434e-04

# monitoring system for duration
window = 10 * iho + iho/2

# Threshold to invoke RL
thr = 0

# Action space
# Velocity
v = [100.0, 250.0, 500.0, 750.0, 1000.0]

# Queue probability less than
capacity = [0.66, 0.33, 0.0]
state_size = len(v)
indices = [(i, j) for i in range(len(capacity)) for j in range(len(v))]
# print (indices)
# exit(1)

# Percentage and count of new infections
pI = 0.0005
cI = 100000

# Frequency of new infections
fI = 24 * 30

# bed capacity on healthcare
real_bed = [380, 231, 132, 686, 229]

# queue model parameter
recovery_time = 14.0
hospital_recover = 0.8
mus = 1.0/float(recovery_time * 24)

iterate = 100

# Correlation list:
C_List = []

fig, ax1 = plt.subplots()
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Velocity(PPKM level-)', color='green')

ax2 = ax1.twinx()
ax2.set_ylabel('Probability of queue', color='red')

v_change = []

# Set the percent you want to explore (RL)
while True :    
    epsilon = 0.75
    decay = 0.99
    lr = 0.3
    gamma_RL = 0.8

    Z_list, T_list = [], []

    # List for hospitalization and p(queue) correlation over time for each borough
    DHP = {b: [] for b in range(eB)}

    # Global time
    T = 0

    # Initial population
    Z = []
    for b in B.keys():
        zb = [P[b] - (pe * P[b] + Ir[b] + Iu[b] + D[b]), pe * P[b], Ir[b], Iu[b], 0, D[b], 0]
        # print (zb, sum(zb))
        Z.append(zb)

    # Visualization
    yR = []
    yV = []
    x = []

    # Create SimPy environment and assign nodes to it.
    env = simpy.Environment()

    entities = [Node(env, i, v[0], density[i], Z[i], P[i], real_bed[i], np.zeros((len(capacity), state_size))) for i in range(eB)]
    env.run(until = Duration)

    #print (yV)
    # # Number of time velocity changed
    chn = 0
    for j in range(1, len(yV)):
        if yV[j] != yV[j - 1]:
            chn += 1
    
    v_change.append(chn)
    # print(chn, np.mean(v_change), np.std(v_change))

    # print (x)
    # print ("yR =", yR)

    # set of points with a dip in
    # declines in probability of queue
    dec = []

    # Correlation between the increase in p(queue) and decrease in velocity
    Cx = []
    Cy = []
    
    for i in range(len(yR) - 1):
    
        xv = v.index(yV[i])
        yv = yR[i]
    
        Cx.append(-xv)
        Cy.append(yv)
    # print("Cx =", Cx)
    # print("Cy =", Cx)
    # print("np.corrcoef(Cx, Cy)",np.corrcoef(Cx, Cy))
    curr = np.corrcoef(Cx, Cy)[0, 1]
    if curr > 0:
        print("curr = ", curr)


    if curr > 0.4:
      print("curr > 0.4")
      # print (curr)
      for pt in dec:
        plt.axvline(x = pt, alpha = 0.5)
      for i in range(len(yR) - 1):
        if yR[i] > yR[i + 1]:
          ax1.scatter((i + 1) * window, v.index(yV[i]), c='green', s=4)
          ax2.scatter((i + 1) * window, yR[i], c='red', s=4)
      ax1.plot(x, [v.index(pt) for pt in yV], color='green', alpha=0.5)
      ax2.plot(x, yR, color='red', alpha = 0.5)
      plt.grid()
      fig.tight_layout()  # otherwise the right y-label is slightly clipped
      plt.savefig('Inflection.png', dpi = 300)
      plt.show()
      break
    else : 
      # print("coef not valid")
      continue
    # pickle.dump(yV, open('yV-100.p', 'wb'))
    # C_List.append(np.corrcoef(Cx, Cy)[0, 1])
    # #
    # print (np.mean(C_List))
    # print (np.std(C_List))
    # print ('\n')

    # time.sleep(2)

    # print ('Infected')
    # print (np.mean([entities[u].z_total[2] + entities[u].z_total[3] + entities[u].z_total[4] for u in range(eB)]))
    # print (np.std([entities[u].z_total[2] + entities[u].z_total[3] + entities[u].z_total[4] for u in range(eB)]))
    #
    # print ('Death')
    # print (np.mean([entities[u].z_total[4] for u in range(eB)]))
    # print (np.std([entities[u].z_total[4] for u in range(eB)]))

    X = []
    L = DHP[0]
    X.append([pt[0] for pt in L])

    Hos_mean = [np.mean([DHP[b][t][0] for b in range(eB)]) for t in range(len(L))]
    Hos_std = [np.std([DHP[b][t][0] for b in range(eB)]) for t in range(len(L))]

    pqueue_mean = [np.mean([DHP[b][t][1] for b in range(eB)]) for t in range(len(L))]
    pqueue_std = [np.std([DHP[b][t][1] for b in range(eB)]) for t in range(len(L))]

    # print (pqueue_mean)
    # print (pqueue_std)

    pickle.dump(Hos_mean, open('Hos_mean.p', 'wb'))
    pickle.dump(Hos_std, open('Hos_std.p', 'wb'))
    pickle.dump(pqueue_mean, open('pqueue_mean.p', 'wb'))
    pickle.dump(pqueue_std, open('pqueue_std.p', 'wb'))

# print (C_List)
# print (np.mean(v_change), np.std(v_change))
plt.show()
