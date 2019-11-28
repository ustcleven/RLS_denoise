import numpy as np
import params


def stft_domain_fast_rls_defered_filter_coefs(X_1,
                                              X_2,
                                              d=params.d,
                                              filter_length=params.filter_length,
                                              f_factor=params.F_FACTOR,
                                              delta=params.DELTA):

    # Retrieve useful informations
    n_frames = X_1.shape[0]
    n_freqs = X_1.shape[1]
    dtype = X_1.dtype

    # Initialize variables
    h = np.zeros((n_frames, n_freqs, filter_length, 1), dtype=dtype)

    x_2 = np.zeros(filter_length, dtype=dtype)

    P = np.zeros((n_frames, filter_length, filter_length), dtype=dtype)
    P[0] = np.eye(filter_length) * (1/delta)

    # More initializations
    E = np.zeros(X_1.shape, dtype=dtype)
    g = np.zeros((n_frames, filter_length, 1), dtype=dtype)
    epsilon = np.zeros(X_1.shape, dtype=dtype)

    for m in np.arange(n_frames-1)+1:
        print("m = {}".format(m))
        for f in np.arange(n_freqs):
            # print("f = {}".format(f))

            temp = np.reshape(np.conj(x_2).transpose() @ P[m-1], (1, filter_length))

            # A-priori error
            E[m][f] = X_1[m][f]
            E[m][f] -= np.transpose(np.conj(h[m-1][f])) @ x_2

            # Kalmann gain vector
            num = np.asarray(P[m-1] @ x_2)
            denom = np.asarray(f_factor + temp @ x_2)[0]
            g[m] = np.reshape(num/denom, (filter_length, 1))

            # Update
            P[m] = (P[m-1] - g[m] @ temp)/f_factor
            h[m][f] = h[m-1][f] + g[m] * np.conj(E[m][f])

            # Output
            #epsilon[m][f] = X_1[m][f]
            epsilon[m][f] = 0
            if m >= d:
                epsilon[m][f] -= np.conjugate(h[m-d][f]).transpose() @ x_2

            # If we still have not reached the last frame, refresh x_2
            if m != n_frames-1:
                x_2 = np.roll(x_2, 1)
                x_2[0] = X_2[m+1][f]
    return epsilon
