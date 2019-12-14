import numpy as np
import helper


def stft_fast_rls_def_coefs(X_1, X_2, d, filter_length, f_factor, delta):
    # Retrieve useful information
    n_frames = X_1.shape[1]
    n_freqs = X_1.shape[0]
    dtype = X_1.dtype

    # Initialize variables
    h = np.zeros((n_freqs, n_frames, filter_length, 1), dtype=dtype)

    x_2 = np.zeros((n_freqs, filter_length), dtype=dtype)

    P = np.zeros((n_frames, filter_length, filter_length), dtype=dtype)
    P[0] = np.eye(filter_length, dtype=dtype) * (1 / delta)

    # More initializations
    E = np.zeros(X_1.shape, dtype=dtype)
    g = np.zeros((n_frames, filter_length, 1), dtype=dtype)
    epsilon = np.zeros(X_1.shape, dtype=dtype)

    for m in np.arange(n_frames - 1) + 1:
        helper.update_progress(m/n_frames)
        print('Frame {}/{}'.format(m, n_frames))
        for f in np.arange(n_freqs):
            temp = np.reshape(np.conj(x_2[f]).transpose() @ P[m - 1], (1, filter_length))

            # A-priori error
            E[f][m] = X_1[f][m] - np.transpose(np.conj(2*h[f][m - 1])) @ x_2[f]

            # Kalmann gain vector
            num = np.asarray(P[m - 1] @ x_2[f])
            denom = np.asarray(f_factor + temp @ x_2[f])[0]
            g[m] = np.reshape(num / denom, (filter_length, 1))

            # Update
            P[m] = (P[m - 1] - g[m] @ temp) / f_factor
            h[f][m] = h[f][m - 1] + g[m] * np.conj(E[f][m])

            # Output
            epsilon[f][m] = X_1[f][m]
            if m >= d:
                epsilon[f][m] -= np.transpose(np.conjugate(2*h[f][m - d])) @ x_2[f]
        # If we still have not reached the last frame, refresh x_2
        if m != n_frames - 1:
            x_2 = np.roll(np.transpose(x_2), 1)
            x_2[0] = X_2[:, m + 1]
            x_2 = np.transpose(x_2)
    helper.update_progress(1)
    return epsilon


def stft_wiener_filter(X_1, X_2, alpha):
    n_freqs = X_1.shape[0]
    n_frames = X_1.shape[1]

    # Init variables
    r_0_1_prev = np.zeros(n_freqs, dtype=complex)
    r_0_1 = np.zeros(n_freqs, dtype=complex)

    r_0_prev = np.zeros(n_freqs, dtype=complex)
    r_0 = np.zeros(n_freqs, dtype=complex)

    h = np.zeros(n_freqs, dtype=complex)
    output = np.zeros(X_1.shape, dtype=complex)

    # Fill the output matrix
    for m in np.arange(n_frames):
        helper.update_progress(m / n_frames)
        print('Frame {}/{}'.format(m, n_frames))
        for f in range(X_1.shape[0]):
            r_0_1[f] = alpha * X_2[f, m] * np.conj(X_1[f, m]) + (1 - alpha) * r_0_1_prev[f]
            r_0[f] = alpha * np.square(np.abs(X_1[f, m])) + (1 - alpha) * r_0_prev[f]
            if r_0[f] != 0:
                h[f] = r_0_1[f] / r_0[f]
            else:
                print('Encountered a zero in the division')
        output[:, m] = np.multiply(h, X_1[:, m])
        r_0_1_prev = r_0_1
        r_0_prev = r_0
    helper.update_progress(1)
    return output
