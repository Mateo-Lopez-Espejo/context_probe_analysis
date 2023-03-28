import numpy as np

def square_rows_cols(n: int) -> tuple[int, int]:
    """
    for n items, organized is the most square arrangement of rows and columns. prioritizing increasing width over height.
    consider that the end array might have more slots than specified items
    """
    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(np.sqrt(n)))
    if rows * cols < n:
        cols += 1

    return rows, cols


def squarefy(t, y, y0=None):
    """
    Takes a vector or 2d array with dimensions Time x Repetitiosn meant to be plotted as lines,
    and transforms into a format what when plotted, displays square time bins, like a histogram or PSTH.
    note: SVD hates it, MLE loves it.
    :param y: vector, or 2d array
    :param t: vector of time samples of the same shape as y dim-0
    :return:
    """
    # duplicates y values to define left and right edges of square
    # duplicates and rolls t values so  y values are connected by either horizontal or vertical lines
    yy = np.repeat(y,2, axis=0)
    tt = np.roll(np.repeat(t, 2), -1)

    # since we rolled time, we don't want the last time to be the same as the first,
    # but rather to be equal to the last time point, plus the delta time
    dt = t[-1] - t[-2]
    tt[-1] = t[-1] + dt

    # normally the squarefied PSTHs start with a horizontal segment
    # sometimes we want the lines to start with a vertical segment,
    # y0 defines the starting height of that vertical segment
    # usefull to connect adyacent squarefied PSTHs that end and start at different y values
    if y0 is not None:
        tt = np.insert(tt, 0, tt[0])
        yy = np.insert(yy, 0, y0)

    return tt, yy
