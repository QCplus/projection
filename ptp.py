import numpy as np

from octave_funcs import *

OPTIMAL                 =   0
OVERSIZED_BASIS         = -19
LINDEP_BASIS            = -20
INIT_ERROR              = -21
NO_WAY_TOI              = -22
CHOLIN_CRASH            = -23
BAD_OLD_L               = -24
MULTI_IN                = -25
DBLE_VTX                = -26
EXCESS_ITER             = -27
RUNNING                 = -28
NEGGRAMM                = -29


def ptp(X, maxit, eps, verbose, kvec0, R0):
    curr = dict()
    info = RUNNING
    ifac = 0

    if len(kvec0) == 0:
        ifac, curr = cold_start(X, curr)
    else:
        if R0.shape[0] != len(kvec0):
            info = INIT_ERROR
            print("XXXX INIT_ERROR: nonmatching sizes of kvec0 {}, {}".format(kvec0.shape[0], 1))
            print(" and R0 {}, {}.".format(*R0.shape))
            print(" Reverting to the cold start.")
            ifac, curr = cold_start(X, curr)
        else:
            curr['kvec'] = kvec0
            curr['R'] = R0
            lmb = np.linalg.solve(R0, np.linalg.solve(R0.T, np.ones(kvec0.shape[0])))
            curr['lmb'] = lmb / sum(lmb)
    curr['z'] = X[:, curr['kvec']].dot(curr['lmb'])

    report = {'zz': sumsq(curr['z']), 'zx': 0, 'in': ifac, 'out': 0,
              'iter': np.zeros(2, dtype=int), 'lenbas': len(curr['kvec'])}

    if verbose >= 0:
        report_iter(report)

    vmin, ifac = get_ifac(X, curr['z'], eps)

    if ifac == 0:
        return curr['z'], eps, report['iter'], curr['lmb'], curr['kvec'], curr['R'], OPTIMAL

    while ifac and (report['iter'] < maxit).all():
        curr['kvec'], curr['lmb'], curr['R'], del_iter = newbas(curr['kvec'], curr['lmb'], ifac, curr['R'], X, eps)
        curr['z'] = X[:, curr['kvec']].dot(curr['lmb'])

        report['iter'][0] += 1
        report['iter'][1] += del_iter
        report['zz'] = sumsq(curr['z'])
        report['zx'] = vmin
        report['in'] = ifac
        report['lenbas'] = len(curr['kvec'])
        if verbose > 0 and (report['iter'][0] % verbose == 0):
            report_iter(report)

        vmin, ifac = get_ifac(X, curr['z'], eps)
        if ifac == 0:
            info = OPTIMAL
            break

    report['in'] = ifac
    report['zz'] = sumsq(curr['z'])
    report['zx'] = vmin

    if verbose >= 0:
        if info == OPTIMAL:
            print("\nXXXX Solved to optimality\n")
    return curr['z'], eps, report['iter'], curr['lmb'], curr['kvec'], curr['R'], info


def proplus(kvec, ifac, R, X):
    g = X[:, ifac]
    r = g.dot(X[:, kvec])
    if len(kvec) == X.shape[0]:
        lamb = np.linalg.solve(-R, np.linalg.solve(R.T, r))
        return 1 / (1 + sum(lamb)) * np.c_[lamb, 1]
    Re = np.c_[r, np.ones(r.shape)]
    Z = np.linalg.solve(R, np.linalg.solve(R.T, Re))
    A = np.array([[sumsq(g), 1], [1, 0]]) - Re.T.dot(Z)
    xit = np.linalg.inv(A)[:, 1]
    return np.r_[-Z.dot(xit), xit[0]]


def newbas(kvec, lmb_old, ifac, R, X, eps):
    iter = 0
    lmb_new = proplus(kvec, ifac, R, X)
    kvec_new = kvec
    if all(lmb_new >= -eps):
        kvec_new = np.array([*kvec_new, ifac])
        R_new = lastadd(X[:, kvec_new], R)
        return kvec_new, lmb_new, R_new, iter

    lmb_m, izero = mid_lambda(np.array([*lmb_old, 0]), lmb_new)
    if izero == -1:
        print(" ST-OPT !!!")
        exit(-1)

    kvec_new = np.array([*kvec_new[:izero], *kvec_new[(izero + 1):len(kvec_new)]])
    lmb = np.array([*lmb_m[:izero], *lmb_m[(izero + 1):len(lmb_m)]])
    R_new = choldelete(R, izero)
    kvec_new = np.array([*kvec_new, ifac])
    R_new = lastadd(X[:, kvec_new], R_new)
    lmb_new = baric(R_new)

    while any(lmb_new < -eps):
        lmb_m, izero = mid_lambda(lmb, lmb_new)
        kvec_new = np.array([*kvec_new[:izero], *kvec_new[(izero + 1):len(kvec_new)]])
        lmb = np.array([*lmb_m[:izero], *lmb_m[(izero + 1):len(lmb_m)]])
        R_new = choldelete(R_new, izero)
        lmb_new = baric(R_new)
        iter += 1
    return kvec_new, lmb_new, R_new, iter


def get_ifac(X, z, epstol):
    v = z.dot(X) - sumsq(z)
    ifac = np.argmin(v)
    reps = epstol * np.linalg.norm(X[:, ifac])
    if v[ifac] > -reps:
        ifac = 0
    return v[ifac], ifac


def lastadd(X, R):
    u = X[:, X.shape[1]-1].dot(X)
    q = np.linalg.solve(R.T, u[0:len(u) - 1].T)
    zz = np.sqrt(abs(u[-1] - sumsq(q)))
    RU = np.r_[R, np.zeros((1, R.shape[1]))]
    return np.c_[RU, np.r_[q, zz]]


def cold_start(X, curr):
    ifac = np.argmin(sumsq(X))
    curr['R'] = np.array([[np.linalg.norm(X[:, ifac])]])
    curr['kvec'] = np.array([ifac])
    curr['lmb'] = np.array([1])
    return ifac, curr


def report_iter(report):
    print(" ++", end='')
    print(" iter {iter[0]:4d}(+) {iter[1]:4d}(-)".format(iter=report["iter"]), end='')
    print(" len {:4d}".format(report["lenbas"]), end='')
    print(" zx {:12.4e}".format(report["zx"]), end='')
    print(" in {:6d}".format(report["in"]), end='')
    print(" zz {:16.8e}".format(report["zz"]))


def baric(R):
    lamb = np.linalg.solve(R, np.linalg.solve(R.T, np.ones(R.shape[0])))
    return lamb / sum(lamb)


def mid_lambda(lmb_old, lmb_new):
    if all(lmb_new < 0):
        return lmb_new, -1
    lmb = (lmb_old / (lmb_old - lmb_new))[lmb_new < 0]
    imin = np.argmin(lmb)
    izero = np.array([i for i in range(lmb_new.shape[0])])[lmb_new < 0][imin]
    return lmb[imin] * lmb_new + (1 - lmb[imin]) * lmb_old, izero
