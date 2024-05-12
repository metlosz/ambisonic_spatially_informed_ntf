from numpy import log, trace, exp, ndarray, pi, prod, sum
from numpy.linalg import det, pinv
from scipy.special import gamma


def pdf(covariance_matrices: ndarray, scale_matrices: ndarray, degrees_of_freedom: float) -> float:
    """
    Probability density function of the Wishart distribution.

    Parameters
    ----------
    covariance_matrices
        Stacked covariance matrices. Shape: [... x channel x channel]
    scale_matrices
        Stacked covariance matrices. Shape: [... x channel x channel]
    degrees_of_freedom
        Degrees of freedom of the Wishart distribution.

    Returns
    -------
    probability
        Probability density of the Wishart distribution.
    """
    number_of_channels = covariance_matrices.shape[-1]
    probability = (
            det(scale_matrices) ** (-1 * degrees_of_freedom) *
            det(covariance_matrices) ** (degrees_of_freedom - number_of_channels) *
            exp(-1 * trace(pinv(scale_matrices) @ covariance_matrices, axis1=-1, axis2=-2)) /
            (pi ** (number_of_channels * (number_of_channels - 1) / 2) *
             prod([gamma(degrees_of_freedom - channel_index + 1) for channel_index in range(1, number_of_channels + 1)])
             )
    )
    return probability


def log_pdf(covariance_matrices: ndarray, scale_matrices: ndarray, degrees_of_freedom: float) -> float:
    """
    Log-probability density function of the Wishart distribution.

    Parameters
    ----------
    covariance_matrices
        Stacked covariance matrices. Shape: [... x channel x channel]
    scale_matrices
        Stacked covariance matrices. Shape: [... x channel x channel]
    degrees_of_freedom
        Degrees of freedom of the Wishart distribution.

    Returns
    -------
    log_probability
        Log-probability density of the Wishart distribution.
    """
    number_of_channels = covariance_matrices.shape[-1]
    log_probability = (
            log(det(scale_matrices) ** (-1 * degrees_of_freedom)) +
            log(det(covariance_matrices) ** (degrees_of_freedom - number_of_channels)) -
            trace(pinv(scale_matrices) @ covariance_matrices, axis1=-1, axis2=-2) -
            log(pi ** (number_of_channels * (number_of_channels - 1) / 2)) -
            sum(log(
                [gamma(degrees_of_freedom - channel_index + 1) for channel_index in range(1, number_of_channels + 1)])))
    return log_probability
