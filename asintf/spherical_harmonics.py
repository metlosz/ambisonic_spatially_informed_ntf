from numpy import floor, imag, ndarray, real, sqrt, zeros
from scipy.special import sph_harm


def number_of_channels_to_order(number_of_channels: int) -> int:
    """
    Calculate Spherical Harmonic order from number of channels.

    Parameters
    ----------
    number_of_channels
        Number of channels.

    Returns
    -------
    order
        Spherical Harmonic order.
    """
    return int(floor(sqrt(number_of_channels) - 1))


def order_to_number_of_channels(order: int) -> int:
    """
    Calculate number of channels from Spherical Harmonic order.

    Parameters
    ----------
    order
        Spherical Harmonic order.
    Returns
    -------
    number_of_channels
        Number of channels.
    """
    return int((order + 1) ** 2)


def matrix(angles, order) -> ndarray:
    """
    Spherical Harmonic coefficients in the N3D-ACN convention.

    Parameters
    ----------
    angles
        Spherical coordinate system angles θ and φ, where 0 < θ < π and 0 < φ < 2π. Shape: [direction x 2 (θ, phi)]
    order
        Maximum Spherical Harmonic Transform order.

    Returns
    -------
    spherical_harmonic_matrix
        Spherical Harmonic coefficients. Shape: [direction x coefficient]
    """
    spherical_harmonic_matrix = zeros((angles.shape[0], order_to_number_of_channels(order)))
    for i, ang in enumerate(angles):
        j = 0
        for n in range(order + 1):
            for m in range(-n, n + 1):
                if m == 0:
                    spherical_harmonic_matrix[i, j] = real(sph_harm(0, n, ang[1], ang[0]))
                elif m < 0:
                    spherical_harmonic_matrix[i, j] = sqrt(2) * (-1) ** abs(m) * imag(
                        sph_harm(abs(m), n, ang[1], ang[0]))
                elif m > 0:
                    spherical_harmonic_matrix[i, j] = sqrt(2) * (-1) ** m * real(sph_harm(m, n, ang[1], ang[0]))
                j += 1
    return spherical_harmonic_matrix
