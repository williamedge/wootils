import numpy as np

def gaussian_uweights(center, coord, sigma):
    """
    Calculate the Gaussian weights of irregular data.

    Parameters
    ----------
    coord : np.array
        Coordinates corresponding to the data.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    weights : np.array
        Gaussian weights of the data.
    """
    weights = np.exp(-0.5 * ((coord - center) / sigma) ** 2)
    weights /= np.sum(weights)  # Normalize the weights
    return weights


def boxcar_weights(coord):
    """
    Calculate boxcar (uniform) weights for the given coordinates.

    Parameters
    ----------
    coord : np.array
        Coordinates corresponding to the data.

    Returns
    -------
    weights : np.array
        Boxcar weights for the coordinates.
    """
    weights = np.ones_like(coord)
    weights /= weights.sum()  # Normalize the weights
    return weights


def tukey_weights(coord, alpha=0.5):
    """
    Calculate Tukey weights for the given coordinates.

    Parameters
    ----------
    coord : np.array
        Coordinates corresponding to the data.
    alpha : float
        Shape parameter of the Tukey window. Default is 0.5.

    Returns
    -------
    weights : np.array
        Tukey weights for the coordinates.
    """
    N = len(coord)
    weights = np.ones(N)
    for n in range(N):
        if n < alpha * (N - 1) / 2:
            weights[n] = 0.5 * (1 + np.cos(np.pi * (2 * n / (alpha * (N - 1)) - 1)))
        elif n > (N - 1) * (1 - alpha / 2):
            weights[n] = 0.5 * (1 + np.cos(np.pi * (2 * n / (alpha * (N - 1)) - 2 / alpha + 1)))
    weights /= weights.sum()  # Normalize the weights
    return weights

def blackman_weights(coord):
    """
    Calculate Blackman weights for the given coordinates.

    Parameters
    ----------
    coord : np.array
        Coordinates corresponding to the data.

    Returns
    -------
    weights : np.array
        Blackman weights for the coordinates.
    """
    N = len(coord)
    weights = 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1)) + 0.08 * np.cos(4 * np.pi * np.arange(N) / (N - 1))
    weights /= weights.sum()  # Normalize the weights
    return weights

def hann_weights(coord):
    """
    Calculate Hann weights for the given coordinates.

    Parameters
    ----------
    coord : np.array
        Coordinates corresponding to the data.

    Returns
    -------
    weights : np.array
        Hann weights for the coordinates.
    """
    N = len(coord)
    weights = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    weights /= weights.sum()  # Normalize the weights
    return weights

def hamming_weights(coord):
    """
    Calculate Hamming weights for the given coordinates.

    Parameters
    ----------
    coord : np.array
        Coordinates corresponding to the data.

    Returns
    -------
    weights : np.array
        Hamming weights for the coordinates.
    """
    N = len(coord)
    weights = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    weights /= weights.sum()  # Normalize the weights
    return weights