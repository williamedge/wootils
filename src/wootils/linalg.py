import numpy as np


def np_linreg2D(x,y):
    A = np.vstack([x, np.ones(len(x.T))]).T
    y = y[:, np.newaxis]
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
    return alpha


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))


def vector_corr(v1, v2):
    """ Returns the vector correlation
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)


def norm(a):
    return (a - np.mean(a)) / (np.std(a))


def normcorr(a, b, mode='valid', denan=True):
    if denan:
        nanx = ~np.isnan(a) | ~np.isnan(b)
    else:
        nanx = np.full(a.shape, True)
    return np.correlate(norm(a[nanx])/len(a[nanx]), norm(b[nanx]), mode=mode)


def polar_mean(angles):
    # Calculate the unit vectors of theta
    theta_x = np.cos(angles)
    theta_y = np.sin(angles)

    # Get the mean of the unit vectors
    theta_x_mean = np.mean(theta_x)
    theta_y_mean = np.mean(theta_y)

    # Compute the mean angle
    theta_mean = np.arctan2(theta_y_mean, theta_x_mean)
    return theta_mean

# Calculate the standard deviation of polar data
def polar_stddev(angles):
    la = len(angles)
    sins = 0
    coss = 0
    for i in range(la):
        sins += np.sin(angles[i])
        coss += np.cos(angles[i])
    
    sins /= la
    coss /= la

    stddev = np.sqrt(-1* np.log(sins*sins + coss*coss))
    return stddev