import math
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import least_squares


def instantiate():
    Duration = 61 # duration in days
    eB = 5 #number of borough
    iho = 6 # monitoring per _ day
    fr = 90 # proportion of reported
    fu = 10 # proportion of unreported

    monitor = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

    actual_inf = [1932, 1461, 1510, 1106, 507, 398, 279, 212, 111, 105, 71] # inf cumulative
    actual_inf = [float(v) / float(eB) for v in actual_inf] # inf per borough
    unrep_inf = [float(fu * v) / float(fr) for v in actual_inf] # inf unreported

    # Density
    density = [1900, 500, 740, 2000, 11000]

    # Infection rate
    p = 0.01
    
    # borough list : Bantul, Gunung Kidul, Kulon Progo, Sleman, Yogyakarta
    P = [985770, 747161, 436395, 1125804, 373589] # population of each borough
    Ir = [actual_inf[0] for i in range(eB)]
    Iu = [unrep_inf[0] for i in range(eB)]
    D = [0, 0, 0, 0, 0]
    
    return Duration, eB, iho, fr, fu, monitor, actual_inf, unrep_inf, density, p, P, Ir, Iu, D

def opt(x, b, mode):

    pe = x[0] # number of class E
    velocity = x[1] # velocity min
    gamma = x[2] # gamma opt

    global Duration, P, sigma, alphas, track, iho, p, density, monitor, actual_inf, unrep_inf

    dense = density[b]
    pop = P[b]

    z0 = [P[b] - (pe * P[b] + Ir[b] + Iu[b] + D[b]), pe * P[b], Ir[b], Iu[b], 0, D[b], 0]
    V = [z0[2]]
    for t in range(iho, Duration):

        if t % iho == 0:
            beta = p * math.sqrt(2) * math.pi * velocity * (dense * math.pow(10, -6)) * 1

            z0[0] = z0[0] - (beta * z0[0] * z0[3]) / pop #S
            
            new_infected = sigma * track * z0[1]
            z0[1] = z0[1] + (beta * z0[0] * z0[3]) / pop - new_infected #E
            z0[2] = z0[2] + new_infected - gamma * z0[2] #I_r
            z0[3] = z0[3] + sigma * (1 - track) * z0[1] - gamma * z0[3] #I_u
            z0[4] = z0[4] + gamma * (1.0 - alphas) * z0[2] #R
            z0[5] = z0[5] + gamma * alphas * z0[2] #D
            z0[6] = z0[6] + gamma * z0[3] #Re

            if t in monitor:
                V.append(z0[2])
    
    # initialization for KL Differgence
    a = np.asarray(actual_inf, dtype=np.float) # infected actual
    b = np.asarray(V, dtype=np.float) # infected predicted

    if mode == 1:
        # for printing graph in per _ day
        plt.clf()
        
        plt.plot([t * 6 for t in range(len(actual_inf))], actual_inf, marker = 'o', label = 'actual')
        plt.plot([t * 6 for t in range(len(actual_inf))], V, marker = 'o', label = 'predicted')
        print ('\nactual infected = ', actual_inf)
        print ('\npredicted infected = ',V)
        
        res_lt = [ abs(actual_inf[x] - V[x]) for x in range (len (actual_inf))]  
        print ('\ndifference = ', res_lt)
        print ('\ntotal difference = ', sum(res_lt))

        plt.legend(fontsize = 15)
        plt.xlabel('Duration in days', fontsize = 15)
        plt.ylabel('Number of infected', fontsize = 15)

        plt.tight_layout()

    if mode == 0:
        # for printing graph if not in _ day using KL
        return np.sum(np.where(a != 0, a * np.log(a / b), 0)) # counting KLd
    return V

Duration, eB, iho, fr, fu, monitor, actual_inf, unrep_inf, density, p, P, Ir, Iu, D = instantiate()

# SEIRD parameters
sigma, alphas, track = 0.25, 0.05, fr / (fr + fu)
print('tracing contact rate = ', track)

x0 = [0.2, 100, 0.5]
res_1 = least_squares(opt, x0 = x0, bounds = ([0.0, 0.0, 0.0], [0.2, 1000.0, 0.5]), args = (3, 0))
x = res_1.x
print ("\nx =",x)

Duration, eB, iho, fr, fu, monitor, actual_inf, unrep_inf, density, p, P, Ir, Iu, D = instantiate()

# finding opt value of E, velocity, gamma
opt(x, 0, 1)

V = []
