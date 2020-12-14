# -*- coding: utf-8 -*-
"""
test_radial_spacing.py

Test our radial spacing function

Created on Mon Dec  7 16:19:31 2020

@author: d34763s
"""

import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar

# %% Test

r = np.linspace(2.5, 10, 101)

spac = pydar.radial_spacing(r)

plt.plot(r, spac[:,0], label='az')
plt.plot(r, spac[:,1], label='r')