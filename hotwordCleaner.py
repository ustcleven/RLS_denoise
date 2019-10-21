import numpy as np


def stft_domain_fast_rls_defered_filter_coefs(X_1, x_2, d, L, forgetting_factor, delta, num_frames):
    h = np.zeros(L)
    P = np.eye(L)*(1/delta)
    E = np.zeros(num_frames)
    g = np.zeros(num_frames)
    epsilon = np.zeros(num_frames)
    frames = np.arange(1, num_frames)
    for m in frames:
        temp = np.conj(x_2.T)[m]@P[m-1]
        E[m] = X_1[m]-np.conj(h.T)[m-1]@x_2[m]
        g[m] = (P[m-1]@x_2[m])/(forgetting_factor + temp@x_2[m])
        P[m] = (1/forgetting_factor)*(P[m-1]-g[m]@temp)
        h[m] = h[m-1] + g[m]@np.conj(E[m])
        epsilon[m] = X_1[m]-np.conj(h.T)[m-d]@x_2[m]
    return epsilon
