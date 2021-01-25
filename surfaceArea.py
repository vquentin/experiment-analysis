from math import sqrt, pi
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from statistics import mean, median

from mpl_toolkits.axes_grid1 import AxesGrid

diameters = list(range (50,76200,5)) # diameter in um
r_i = 380 #etch thickness in um

initial_area = [pi*d*d/4 for d in diameters] # area in um2
final_area_increase = [0.5*pi*pi*(d+r_i)*r_i for d in diameters]
proportional = [dA/iA for dA,iA in zip(final_area_increase, initial_area)]

plt.semilogy(diameters, proportional)
plt.xlabel('cavity diameter (um)')
plt.ylabel('increase in surface area (-)')
plt.show()