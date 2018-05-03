from math import *
from sys import argv

# import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

"""
    After retrieving the parameters for the two models, this script will plot the results.
    
    IN: alpha11, mu11, sigma11, alpha12, mu12, sigma12
"""
alpha1 = float(argv[1])
mu1 = float(argv[2])
sigma1 = float(argv[3])
alpha2 = float(argv[4])
mu2 = float(argv[5])
sigma2 = float(argv[6])

alpha1_sqrt_pi_sigma1 = (alpha1 / (sqrt(2 * pi) * sigma1))
alpha1_sqrt_pi_sigma2 = (alpha2 / (sqrt(2 * pi) * sigma2))
time = pl.linspace(0, 30)

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


result1 = upside_down_normal_distribution(time)
result2 = normal_distribution(time)
pl.figure(1)
pl.subplot(2, 1, 1)
pl.plot(time, result1), pl.grid(True)
pl.xlabel("$t$"), pl.ylabel("$p11(k)$")
pl.title("$p_{11}(k)=1-\\frac{\\alpha}{\sqrt{(2\pi)}\sigma}e^{(-0.5(\\frac{k-\mu}{\sigma})^2)}$")
pl.title("$\\alpha=" + str(alpha1) + ", \mu=" + str(mu1) + ", \sigma=" + str(sigma1) + "$", loc='right')
pl.subplot(2, 1, 2)
pl.plot(time, result2), pl.grid(True)
pl.xlabel("$t$"), pl.ylabel("$p12(k)$")
pl.title("$p_{12}(k)=\\frac{\\alpha}{\sqrt{(2\pi)}\sigma}e^{(-0.5(\\frac{k-\mu}{\sigma})^2)}$")
pl.title("$\\alpha=" + str(alpha2) + ", \mu=" + str(mu2) + ", \sigma=" + str(sigma2) + "$", loc='right')

"""
    TODO: Stationary market status should be around [0.43 0.57]T
"""
s0 = np.matrix([0.01, 0.99]).T
time = range(30)

result1 = upside_down_normal_distribution(time)
result2 = normal_distribution(time)

state_vectors = [s0]

for i in time:
    transition = np.matrix([[result1[i], result2[i]], [1-result1[i], 1-result2[i]]])
    state_vectors.append(transition * state_vectors[i])
    print("tmx: " + str(transition) + "\nstate(" + str(i) + "): " + str(state_vectors[i]) + "\nstate(" + str(i+1) + "): "
          + str(state_vectors[i+1]))

state_vectors = np.array(state_vectors)
pl.figure(2)
pl.plot(time, state_vectors[:30, 0, 0])
pl.plot(time, state_vectors[:30, 1, 0])
pl.show()
