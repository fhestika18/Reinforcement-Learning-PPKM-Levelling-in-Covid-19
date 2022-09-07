# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:48:43 2022

@author: FITRI HESTIKARANI
"""

import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
import ast
import statsmodels.stats.api as sms
import scipy.stats as st
import pandas as pd

# Exponential decay

V = [100.0, 250.0, 500.0, 750.0, 1000.0]
for v in V:
    print (float(v)/(max(V)))
    plt.plot([i * 0.1 for i in range(11)], [float(v)/(max(V)) * math.exp(-i * 0.1) for i in range(11)]
             , marker = 'o', label = str(float(v)/1000.0) + ' km/h')

plt.xlabel('Probability of queue', fontsize = 15)
plt.ylabel('Reward', fontsize = 15)
plt.legend(fontsize = 15)

plt.tight_layout()
plt.show()

# Correlation box plot
# construct some data like what you have:

data = [0.2132952023150317, 0.21694869584598675, 0.36266107904271633, 0.24127881171888246, 0.2594781412437297, 
        0.3650450603930366, 0.23732870006406467, 0.010984975296262085, 0.33031829464624646, 0.5221840794977476]

# create boxplot
plt.boxplot(data)

plt.xlabel('Provinsi DIY', fontsize = 15)
plt.ylabel('Correlation coefficient', fontsize = 15)

plt.tight_layout()
plt.show()