from math import *
from random import *
from sys import argv

import numpy as np
from scipy.optimize import *

"""
    
"""


def upside_down_normal_distribution(params, k1, k2, k3, p1, p2, p3):
    # For clarification
    alpha = params[0]
    mu = params[1]
    sigma = params[2]

    result = np.zeros(3)

    sqrt_pi = sqrt(2 * pi)
    alpha_sqrt_pi_sigma_ = (alpha / (sqrt_pi * sigma))
    result[0] = 1 - (alpha_sqrt_pi_sigma_ * (e ** ((-0.5) * pow((k1 - mu) / sigma, 2)))) - p1
    result[1] = 1 - (alpha_sqrt_pi_sigma_ * (e ** ((-0.5) * pow((k2 - mu) / sigma, 2)))) - p2
    result[2] = 1 - (alpha_sqrt_pi_sigma_ * (e ** ((-0.5) * pow((k3 - mu) / sigma, 2)))) - p3
    # result[2] = 1 - ((alpha / (sqrt_pi * sigma)) * (e ** ((-1 / 2) * pow((2 - mu) / sigma, 2)))) - 0.970

    return result


def normal_distribution(params, k1, k2, k3, p1, p2, p3):
    # For clarification
    alpha = params[0]
    mu = params[1]
    sigma = params[2]

    result = np.zeros(3)

    sqrt_pi = sqrt(2 * pi)
    alpha_sqrt_pi_sigma_ = (alpha / (sqrt_pi * sigma))
    result[0] = (alpha_sqrt_pi_sigma_ * (e ** ((-1 / 2) * pow((k1 - mu) / sigma, 2)))) - p1
    result[1] = (alpha_sqrt_pi_sigma_ * (e ** ((-1 / 2) * pow((k2 - mu) / sigma, 2)))) - p2
    result[2] = (alpha_sqrt_pi_sigma_ * (e ** ((-1 / 2) * pow((k3 - mu) / sigma, 2)))) - p3
    # result[2] = 1 - ((alpha / (sqrt_pi * sigma)) * (e ** ((-1 / 2) * pow((2 - mu) / sigma, 2)))) - 0.970

    return result


values_p11 = tuple((0, 1, 2, float(argv[1]), float(argv[2]), float(argv[3])))
values_p12 = tuple((0, 1, 2, float(argv[4]), float(argv[5]), float(argv[6])))
max_tries = int(argv[7])

inputs = np.zeros(3)
for i in range(0, max_tries):
    inputs[0] = random() + 1.0 * 10.0
    inputs[1] = random() + 1.0 * 10.0
    inputs[2] = random() + 1.0 * 10.0

    result_p11, infodict, ier, mesg = fsolve(
        upside_down_normal_distribution,
        inputs,
        args=values_p11,
        xtol=1e-6,
        maxfev=200,
        full_output=True
    )
    if ier == 1:
        break

print("Return value\t\t" + str(ier))
print("Message:\t\t" + str(mesg))
# print("Idk:\t\t" + str(infodict))
print("Results:\t\t" + str(result_p11))
if ier != 1:
    print("Out of luck.")

for i in range(0, max_tries):
    inputs[0] = random() + 1.0 * 10.0
    inputs[1] = random() + 1.0 * 10.0
    inputs[2] = random() + 1.0 * 10.0
    result_p12, infodict, ier, mesg = fsolve(
        normal_distribution,
        inputs,
        args=values_p12,
        xtol=1e-6,
        maxfev=200,
        full_output=True
    )

    if ier == 1:
        break

print("Return value\t\t" + str(ier))
print("Message:\t\t" + str(mesg))
# print("Idk:\t\t" + str(infodict))
print("Results:\t\t" + str(result_p12))

if ier != 1:
    print("Out of luck.")
