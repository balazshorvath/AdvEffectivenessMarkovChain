from math import *

import numpy as np
from scipy.optimize import *

"""
    
"""


def upside_down_normal_distribution(params):
    # For clarification
    alpha = params[0]
    mu = params[1]
    sigma = params[2]

    result = np.zeros(3)

    sqrt_pi = sqrt(2 * pi)
    result[0] = 1 - ((alpha / (sqrt_pi * sigma)) * (e ** ((-1 / 2) * pow((0 - mu) / sigma, 2)))) - 0.998
    result[1] = 1 - ((alpha / (sqrt_pi * sigma)) * (e ** ((-1 / 2) * pow((1 - mu) / sigma, 2)))) - 0.994
    result[2] = 1 - ((alpha / (sqrt_pi * sigma)) * (e ** ((-1 / 2) * pow((2 - mu) / sigma, 2)))) - 0.984
    # result[2] = 1 - ((alpha / (sqrt_pi * sigma)) * (e ** ((-1 / 2) * pow((2 - mu) / sigma, 2)))) - 0.970

    return result


def normal_distribution(params):
    # For clarification
    alpha = params[0]
    mu = params[1]
    sigma = params[2]

    result = np.zeros(3)

    sqrt_pi = sqrt(2 * pi)
    result[0] = ((alpha / (sqrt_pi * sigma)) * (e ** ((-1 / 2) * pow((0 - mu) / sigma, 2)))) - 0.231
    result[1] = ((alpha / (sqrt_pi * sigma)) * (e ** ((-1 / 2) * pow((1 - mu) / sigma, 2)))) - 0.289
    result[2] = ((alpha / (sqrt_pi * sigma)) * (e ** ((-1 / 2) * pow((2 - mu) / sigma, 2)))) - 0.350
    # result[2] = 1 - ((alpha / (sqrt_pi * sigma)) * (e ** ((-1 / 2) * pow((2 - mu) / sigma, 2)))) - 0.970

    return result


values = fsolve(upside_down_normal_distribution, np.array([2.0, 9.0, 3.0]))

print(values)
