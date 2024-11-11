import numpy as np
from wootils.ofreqs import get_annual_harmonic

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



def harmonic_lstsq(x, freqs):
    A = np.vstack([np.sin(2 * np.pi * f * x) for f in freqs] +
                  [np.cos(2 * np.pi * f * x) for f in freqs]).T
    return A

def fit_annual_harmonic(x, y, n_harmonics=3):
    # Get the first n harmonics
    ann_freqs = get_annual_harmonic(n_harmonics=n_harmonics)

    # Use least squares with sin and cos with multiple frequencies
    A = harmonic_lstsq(x, ann_freqs)
    
    # Fit the model
    model = np.linalg.lstsq(A, y, rcond=None)
    return model

def predict_annual_harmonic(x, coeffs, n_harmonics=3):
    ann_freqs = get_annual_harmonic(n_harmonics=n_harmonics)
    A = harmonic_lstsq(x, ann_freqs)
    return np.dot(A, coeffs)

def eval_model_var(y_obs, y_pred):
    return 1 - np.var(y_obs - y_pred) / np.var(y_obs)

def fit_annual_harmonic_model(time, data, n_harmonics=3, axis=0):
    time_sec = (time - time[0]).astype('timedelta64[s]').astype(float)
    data -= np.mean(data)
    model = fit_annual_harmonic(time_sec, data, n_harmonics=n_harmonics)
    y_pred = predict_annual_harmonic(time_sec, model[0], n_harmonics=n_harmonics)
    return model, y_pred


from wootils.ofreqs import get_annual_harmonic

def create_design_matrix(x, freqs, mean=True):
    """
    Create the design matrix for the harmonic model.
    
    Parameters:
    time (np.array): Time values.
    n_harmonics (int): Number of harmonics to include.
    
    Returns:
    np.array: Design matrix.
    """
    N = len(x)
    if mean:
        A = np.ones((N, 2 * len(freqs) + 1))
        ll = 1
    else:
        A = np.zeros((N, 2 * len(freqs)))
        ll = 0
    for i,f in enumerate(freqs):
        A[:, 2 * (i+ll) - 1] = np.cos(2 * np.pi * f * x)
        A[:, 2 * (i+ll)] = np.sin(2 * np.pi * f * x)
    return A

def fit_harmonic_model(time, data, n_harmonics=2, axis=0, mean=True):
    """
    Fit the harmonic model to 3D data along a chosen dimension using lstsq.
    
    Parameters:
    time (np.array): Time values.
    data (np.array): 3D data array.
    n_harmonics (int): Number of harmonics to fit.
    axis (int): Axis along which to fit the harmonic model.
    
    Returns:
    np.array: Fitted harmonic coefficients.
    """
    # Get the first n harmonics
    ann_freqs = get_annual_harmonic(n_harmonics=n_harmonics)

    # Move the chosen axis to the first dimension
    if axis != 0:
        data = np.moveaxis(data, axis, 0)
    
    # Create the design matrix
    A = create_design_matrix(time, ann_freqs, mean=mean)
    
    # Initialize the coefficients array
    shape = list(data.shape)
    if mean:
        shape[0] = 2 * n_harmonics + 1
    else:
        shape[0] = 2 * n_harmonics
    coeffs = np.zeros(shape)
    
    # Fit the harmonic model along the chosen dimension
    for idx in np.ndindex(data.shape[1:]):
        y = data[(slice(None),) + idx]
        popt, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        coeffs[(slice(None),) + idx] = popt
    
    return coeffs

def predict_harmonic_model(time, coeffs, axis=0, mean=True):
    """
    Predict the harmonic model using the fitted coefficients.
    
    Parameters:
    time (np.array): Time values.
    coeffs (np.array): Fitted harmonic coefficients.
    axis (int): Axis along which the harmonic model was fitted.
    
    Returns:
    np.array: Predicted harmonic model values.
    """
    # Create the design matrix
    if mean:
        n_harmonics = (coeffs.shape[0] - 1) // 2
    else:
        n_harmonics = coeffs.shape[0] // 2
    
    # Get the first n harmonics
    ann_freqs = get_annual_harmonic(n_harmonics=n_harmonics)

    A = create_design_matrix(time, ann_freqs, mean=mean)
    
    # Initialize the predicted data array
    shape = list(coeffs.shape)
    shape[0] = len(time)
    predicted = np.zeros(shape)
    
    # Predict the harmonic model along the chosen dimension
    for idx in np.ndindex(coeffs.shape[1:]):
        params = coeffs[(slice(None),) + idx]
        predicted[(slice(None),) + idx] = A @ params
    
    # Move the first dimension back to the chosen axis
    if axis != 0:
        predicted = np.moveaxis(predicted, 0, axis)
    
    return predicted