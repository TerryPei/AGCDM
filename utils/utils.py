import numpy as np

def r4beta(shape1, shape2, a, b, size):
    x = np.random.beta(shape1, shape2, size)
    return (b - a) * x + a

def get_log_beta_pdf(slip, guess):

    return np.log(0.6 - guess) + np.log(0.6 - slip)

def get_log_normal_pdf(x):

    return x ** 2 * -0.5

def get_log_lognormal_pdf(x):

    return np.log(1.0 / x) + get_log_normal_pd(np.log(x))
