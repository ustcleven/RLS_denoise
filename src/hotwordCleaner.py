import numpy as np
import params


def stft_domain_fast_rls_defered_filter_coefs(x_1,
                                              x_2,
                                              d=params.d,
                                              L=params.L,
                                              forgetting_factor=params.FORGETTING_FACTOR,
                                              delta=params.DELTA):
    h = [np.zeros(d)]
    x_2 = [np.zeros(d)].append(x_2)
    P = [np.eye(L)*(1/delta)]

    E = np.zeros(d)
    g = np.zeros(d)
    epsilon = np.zeros(d)
    frames = np.arange(1, d)
    for m in frames:
        temp = np.conjugate(x_2)[m]@P[m-1]
        E[m] = x_1[m] - np.conjugate(h)[m - 1] @ x_2[m]
        g[m] = (P[m-1]@x_2[m])/(forgetting_factor + temp@x_2[m])
        P[m] = (1/forgetting_factor)*(P[m-1]-g[m]@temp)
        h[m] = h[m-1] + g[m]@np.conj(E[m])
        epsilon[m] = x_1[m] - np.conjugate(h)[m - d] @ x_2[m]
    return epsilon
