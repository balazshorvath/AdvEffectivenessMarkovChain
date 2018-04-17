from math import *
from sys import argv

# import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

alpha1 = float(argv[1])
mu1 = float(argv[2])
sigma1 = float(argv[3])
alpha2 = float(argv[4])
mu2 = float(argv[5])
sigma2 = float(argv[6])

alpha1_sqrt_pi_sigma1 = (alpha1 / (sqrt(2 * pi) * sigma1))
alpha1_sqrt_pi_sigma2 = (alpha2 / (sqrt(2 * pi) * sigma2))
time = pl.linspace(0, 30, 2000)

pl.interactive(False)


def normal_distribution(k):
    length = len(k)
    result = np.zeros(length)
    for i in range(0, length):
        result[i] = alpha1_sqrt_pi_sigma2 * (e ** ((-0.5) * pow((k[i] - mu2) / sigma2, 2)))
    return result


def upside_down_normal_distribution(k):
    length = len(k)
    result = np.zeros(length)
    for i in range(0, length):
        result[i] = 1 - alpha1_sqrt_pi_sigma1 * (e ** ((-0.5) * pow((k[i] - mu1) / sigma1, 2)))
    return result


result1 = normal_distribution(time)
result2 = upside_down_normal_distribution(time)
pl.figure()
pl.plot(time, result1), pl.grid()
pl.figure()
pl.plot(time, result2), pl.grid()
pl.show()
