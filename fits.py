import numpy as np
from scipy.optimize import curve_fit


def _exp(x, a, b):
    return a * np.exp(b * x)


def exp_decay(times, values):
    """
    fits a properly constrained exponential decay to the times and values give, retursn the fitted values
    of the exponential function and the equivalent time constant Tau
    :param times: np.array. 1D, Time points in seconds, same shape as values
    :param values: np.array. 1D, y values, same shape as times
    :return:
    """
    if len(times) != len(values):
        times = times[:len(values)]

    #removes nan padding
    not_nan = np.logical_not(np.isnan(values))
    values = values[not_nan]
    times = times[not_nan]

    popt, pvar = curve_fit(_exp, times, values, p0=[1, 0], bounds=([0, -np.inf], [np.inf, 0]))
    return popt, pvar