from itertools import product, permutations
from math import factorial
import numpy as np

def formulate_ctx_prb(n_vals):
    ind_val = range(1, n_vals + 1)
    # defines universe of pairs
    pairs = ([(0, i) for i in ind_val])  # sound afte silence
    pairs.extend(list(product(ind_val, repeat=2)))  # sound pairs

    print(f'{len(pairs)} pairs, {factorial(n_vals) * n_vals} sequences')

    # create all possible sequences of n values, whith one of them contiguosly duplicated
    Y = dict()  # different format of all_seq pairs
    for i in ind_val:
        for perm in permutations(ind_val):
            perm = list(perm)
            perm.insert(perm.index(i), i)  # inserts duplicated sound
            perm.insert(0, 0)  # inserts leading silence
            perm_str = '_'.join(str(p) for p in perm)

            # finds which pairs this permutations contains and fills coverage matrix
            Y[perm_str] = list()
            for p0, p1 in zip(perm[:-1], perm[1:]):
                Y[perm_str].append(f'{p0}_{p1}')

    # creates a dictionary of definine in which sequences each pair is contained
    X = {f'{p[0]}_{p[1]}': set() for p in pairs}
    for seq in Y:
        for pair in Y[seq]:
            X[pair].add(seq)

    return X, Y

def formulate_ctx_prb_2(n_vals):
    ind_val = range(1, n_vals + 1)
    # defines universe of pairs
    pairs = ([(0, i) for i in ind_val])  # sound afte silence
    pairs.extend(list(permutations(ind_val, r=2)))  # sound pairs
    print(f'{len(pairs)} pairs, {factorial(n_vals)} sequences')

    # create all possible sequences of n values, whith one of them contiguosly duplicated
    Y = dict()  # different format of all_seq pairs
    for perm in permutations(ind_val):
        perm = list(perm)  # inserts duplicated sound
        perm.insert(0, 0)  # inserts leading silence
        perm_str = '_'.join(str(p) for p in perm)

        # finds which pairs this permutations contains and fills coverage matrix
        Y[perm_str] = list()
        for p0, p1 in zip(perm[:-1], perm[1:]):
            Y[perm_str].append(f'{p0}_{p1}')

    # creates a dictionary defining in which sequences each pair is contained
    X = {f'{p[0]}_{p[1]}': set() for p in pairs}
    for seq in Y:
        for pair in Y[seq]:
            X[pair].add(seq)

    return X, Y


# implements Algorithm X, see https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html

# # returns a list with all the valid tilings
def solve_all(X, Y, solution=[]):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve_all(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()


# returns the first valid perfect tiling.
def solve_one(X, Y, solution=[]):
    if not X:
        return solution
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            s = solve_one(X, Y, solution)
            if s:
                return solution
            else:
                deselect(X, Y, r, cols)
                solution.pop()


def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)
