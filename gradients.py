import numpy as np
import math
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.special import digamma


def pdf_split_gaussian(x, mean, sigma, tau):
    c = math.sqrt(2 / math.pi) * sigma ** -1 * (1 + tau) ** -1
    return c * math.exp(-1 / (2 * sigma ** 2) * (x - mean) ** 2) if x <= mean \
        else c * math.exp(-1 / (2 * tau ** 2 * sigma ** 2) * (x - mean) ** 2)


def pdf_multivariate_split_gaussian(x, mean, W, sigma, tau, d=2):
    return np.abs(np.linalg.det(W)) * np.prod(
        [pdf_split_gaussian(np.dot(W[:, j], (x - mean)), 0, sigma[j], tau[j]) for j in range(d)])


def pdf_split_gamma(x, mean, sigma_l, sigma_r, c):
    a_sigma = sigma_l * math.sqrt(gamma(1 / c) / gamma(3 / c))
    a_tau = sigma_r * math.sqrt(gamma(1 / c) / gamma(3 / c))
    return c / ((a_sigma + a_tau) * gamma(1 / c)) * math.exp(- abs(x - mean) ** c / (a_sigma ** c)) if x <= mean \
        else c / ((a_sigma + a_tau) * gamma(1 / c)) * math.exp(- abs(x - mean) ** c / (a_tau ** c))


def pdf_multivariate_split_gamma(x, mean, W, sigma_l, sigma_r, c, d=2):  # abs?
    return np.abs(np.linalg.det(W)) * np.prod(
        [pdf_split_gamma(np.dot(W[:, j], (x - mean)), 0, sigma_l[j], sigma_r[j], c) for j in range(d)])


def s_function(X, W, m, j, c=2):
    v = np.dot((X - m), W[:, j])
    return np.array([np.sum(np.abs(v[v <= 0]) ** c), np.sum(np.abs(v[v > 0]) ** c)])


def l_function(X, m, W):
    d = m.shape[0]
    s = np.array([s_function(X, W, m, j) for j in range(d)], dtype='float64')
    s[s == 0] = np.finfo(float).eps
    return np.abs(np.linalg.det(W)) ** (-2 / 3) * np.prod(s[:, 0] ** (1 / 3) + s[:, 1] ** (1 / 3))


def l_function_gamma(X, m, W, c):
    #     n = X.shape[0]
    d = m.shape[0]
    s = np.array([s_function(X, W, m, j, c) for j in range(d)], dtype='float64')
    s[s == 0] = np.finfo(float).eps
    #     kappa = (1/c * gamma(1/c)) ** -c
    return np.abs(np.linalg.det(W)) ** (-c / (c + 1)) * np.prod(s[:, 0] ** (1 / (c + 1)) + s[:, 1] ** (1 / (c + 1)))


def log_likelihood_gamma(X, m, W, c):
    n = X.shape[0]
    d = m.shape[0]
    kappa = (1 / c * gamma(1 / c)) ** -c
    kappa_l = kappa * n / (c * np.e)
    if kappa_l < np.finfo(float).eps:
        kappa_l = np.finfo(float).eps
    return d * n / c * np.log(kappa_l) - n * (c + 1) / c * np.log(l_function_gamma(X, m, W, c))


def get_sigma_tau(X, m, W):
    d = m.shape[0]
    s = np.array([s_function(X, W, m, j) for j in range(d)])
    s[s == 0] = np.finfo(float).eps
    sigma = np.array([1 / X.shape[0] * s[j, 0] ** (2 / 3) * np.sum(s[j, :] ** (1 / 3)) for j in range(d)])
    tau = np.array([(s[j, 1] / s[j, 0]) ** (1 / 3) for j in range(d)])  # here square as well
    return sigma ** (1 / 2), tau


def get_gamma_sigmas(X, m, W, c):
    d = m.shape[0]
    s = np.array([s_function(X, W, m, j, c) for j in range(d)])
    s[s == 0] = np.finfo(float).eps
    beta = gamma(3 / c) / gamma(1 / c)
    sigma_l = np.array(
        [c / X.shape[0] * beta ** (c / 2) * s[j, 0] ** (c / (c + 1)) * np.sum(s[j, :] ** (1 / (c + 1))) for j in
         range(d)])
    # * beta ** (c/2) ? w appendixie
    tau = np.array([(s[j, 1] / s[j, 0]) ** (1 / (c + 1)) for j in range(d)])
    #     sigma_r = sigma_l * tau
    #     sigma_r = np.array([c / X.shape[0] * beta ** (c/2) * s[j, 0] * s[j, 1] ** (c / (c + 1)) * np.sum(s[j, :] ** (1 / (c + 1))) for j in range(d)])
    return sigma_l ** (1 / c), sigma_l ** (1 / c) * tau, tau


def gg(X, m, W, k, c):
    v = np.dot((X - m), W)
    I = v <= 0
    v = np.abs(v) ** (c - 1) * W[k, :]  # ?
    return np.array([np.sum(v * I, axis=0), np.sum(v * (1 - I), axis=0)])


def gg2(X, m, W, k, p, c):
    v = np.dot((X - m), W)
    I = v <= 0
    v = ((np.abs(v) ** (c - 1)).T * (X[:, k] - m[k])).T
    return np.array([np.sum(v * I, axis=0)[p], np.sum(v * (1 - I), axis=0)[p]])


def gc(X, m, W, c):
    v = np.dot((X - m), W)
    I = v <= 0
    v = np.abs(v) ** c * np.log(np.abs(v))  # ?
    return np.array([np.sum(v * I, axis=0), np.sum(v * (1 - I), axis=0)])


def gradient_split_gamma(X, m, W, c):
    n = X.shape[0]
    d = m.shape[0]
    s = np.array([s_function(X, W, m, j, c) for j in range(d)], dtype='float64')  # 64?
    s[s == 0] = np.finfo(float).eps
    dm = np.array([c / (c + 1) *
                   np.sum(
                       1 / np.sum(s ** (1 / (c + 1)), axis=1) *
                       (1 / (s[:, 0] ** (c / (c + 1))) * gg(X, m, W, k, c)[0] -
                        1 / (s[:, 1] ** (c / (c + 1))) * gg(X, m, W, k, c)[1]))
                   for k in range(d)])

    invW = np.linalg.inv(W)
    dW = np.array([[-(c / (c + 1)) * invW.T[k, p] + 1 / np.sum(s[p, :] ** (1 / (c + 1))) *
                    (-c / ((c + 1) * s[p, 0] ** (c / (c + 1))) * gg2(X, m, W, k, p, c)[0] +
                     c / ((c + 1) * s[p, 1] ** (c / (c + 1))) * gg2(X, m, W, k, p, c)[1]) for k in range(d)] for p in
                   range(d)])
    dc = np.array([-1 / (c + 1) ** 2 * np.log(np.abs(np.linalg.det(W))) +
                   np.sum(1 / np.sum(s ** (1 / (c + 1)), axis=1) *
                          (1 / (c + 1) * (s[:, 0] ** (-c / (c + 1))) * gc(X, m, W, c)[0] - s[:, 0] ** (1 / (c + 1)) / (
                              c + 1) ** 2 * np.log(s[:, 0]) +
                           1 / (c + 1) * (s[:, 1] ** (-c / (c + 1))) * gc(X, m, W, c)[1] - s[:, 1] ** (1 / (c + 1)) / (
                               c + 1) ** 2 * np.log(s[:, 1])))
                   ])

    dlm = dm * -n * (c + 1) / c
    dlW = dW * -n * (c + 1) / c
    dlc = d * n / c ** 2 * (np.log(c * np.e / n) - 1 + c + digamma(1 / c)) + \
          n / c ** 2 * np.log(l_function_gamma(X, m, W, c)) - n * (c + 1) / c * dc
    return np.append(np.vstack([dlm, dlW.T]).flatten(), dlc)  # dm, dW.T #


def optimize_split_gamma(X, m, W, c, method="L-BFGS-B", jac=True):
    d = m.shape[0]
    m0 = m
    W0 = W
    c0 = c
    x0 = np.append(np.vstack([m0, W0]).flatten(), c0)
    if jac:
        #         jac = (lambda x: gradient_split_gamma(X, x[:-1].reshape((3, 2))[0, :], x[:-1].reshape((3, 2))[1:, :], x[-1]))
        jac = (lambda x: -gradient_split_gamma(X, x[:-1].reshape((d + 1, d))[0, :], x[:-1].reshape((d + 1, d))[1:, :], x[-1]))
    # params = minimize(lambda x: np.log(l_function_gamma(X, x[-1].reshape((3, 2))[0, :],
    #                                x[:-1].reshape((3, 2))[1:, :], x[-1])),
    #          x0,
    #          jac=jac,
    #          method=method)
    params = minimize(lambda x: -log_likelihood_gamma(X, x[:-1].reshape((d + 1, d))[0, :],
                                                      x[:-1].reshape((d + 1, d))[1:, :], x[-1]),
                      x0,
                      jac=jac, method=method,
                      bounds=(
                          (None, None), (None, None), (None, None), (None, None), (None, None), (None, None),
                          (0.001, 10)))
    m0 = params.x[:-1].reshape((d + 1, d))[0, :]
    W0 = params.x[:-1].reshape((d + 1, d))[1:, :]
    c0 = params.x[-1]
    return m0, W0, c0, params


def g(X, m, W, k):
    v = np.dot((X - m), W)
    I = v <= 0
    v = 2 * v * W[k, :]  # ?
    return np.array([np.sum(v * I, axis=0), np.sum(v * (1 - I), axis=0)])


def g2(X, m, W, k, p):
    v = np.dot((X - m), W)
    I = v <= 0
    v = ((2 * v).T * (X[:, k] - m[k])).T
    return np.array([np.sum(v * I, axis=0)[p], np.sum(v * (1 - I), axis=0)[p]])


def gradient_split(X, m, W):
    d = m.shape[0]
    s = np.array([s_function(X, W, m, j) for j in range(d)], dtype='float')
    s[s == 0] = np.finfo(float).eps
    dm = np.array([
        np.sum(
            -1 / np.sum(s ** (1 / 3), axis=1) *
            (1 / (3 * s[:, 0] ** (2 / 3)) * g(X, m, W, k)[0] +
             1 / (3 * s[:, 1] ** (2 / 3)) * g(X, m, W, k)[1]))
        for k in range(d)])

    invW = np.linalg.inv(W)
    dW = np.array([[-(2 / 3) * invW.T[k, p] + 1 / np.sum(s[p, :] ** (1 / 3)) *
                    (1 / (3 * s[p, 0] ** (2 / 3)) * g2(X, m, W, k, p)[0] +
                     1 / (3 * s[p, 1] ** (2 / 3)) * g2(X, m, W, k, p)[1]) for k in range(d)] for p in range(d)])
    return np.vstack([dm, dW.T]).flatten()  # dm, dW.T #


def optimize_split_gaussian(X, m, W, method="BFGS", jac=True):
    d = m.shape[0]
    m0 = m
    W0 = W
    x0 = np.vstack([m0, W0]).flatten()
    if jac:
        jac = (lambda x: gradient_split(X, x.reshape((d + 1, d))[0, :], x.reshape((d + 1, d))[1:, :]))
    params = minimize(lambda x: np.log(l_function(X, x.reshape((d + 1, d))[0, :],
                                                  x.reshape((d + 1, d))[1:, :])),
                      x0,
                      jac=jac,
                      method=method)
    m0 = params.x.reshape((d + 1, d))[0, :]
    W0 = params.x.reshape((d + 1, d))[1:, :]
    return m0, W0, params
