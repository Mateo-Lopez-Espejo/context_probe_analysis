import numpy as np
from scipy.optimize import curve_fit


def _exp(x, a, b):
    return a * np.exp(b * x)


def exp_decay(times, values, skip_error=False):
    """
    fits a properly constrained exponential decay to the times and values give, returns the fitted values
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

    if skip_error is False:
        popt, pvar = curve_fit(_exp, times, values, p0=[1, 0], bounds=([0, -np.inf], [np.inf, 0]))
    elif skip_error is True:
        try:
            popt, pvar = curve_fit(_exp, times, values, p0=[1, 0], bounds=([0, -np.inf], [np.inf, 0]))
        except:
            print('Optimal parameters not found, returnin Nan')
            popt = np.empty((2)); popt[:] = np.nan
            pvar = np.empty((2,2)); pvar[:] = np.nan

    else:
        raise ValueError('skip_error must be boolean')

    # calculates the goodness of fit as R2
    fx = _exp(times, *popt)
    R2 = 1 - (np.sum((values - fx) ** 2) / np.sum((values - np.mean(values)) ** 2))

    return popt, R2