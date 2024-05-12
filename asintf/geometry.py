from typing import Optional

from numpy import arccos, arctan2, cos, ndarray, pi, sin, sqrt, zeros
from numpy.linalg import norm


def cartesian_to_spherical(cartesian_coordinates: ndarray) -> ndarray:
    """
    Converts cartesian to spherical coordinates.

    Parameters
    ----------
    cartesian_coordinates
       Cartesian coordinates. Shape: [direction x 3 (x, y, z)]

    Returns
    -------
    spherical_coordinates
       Spherical coordinates r, θ and φ, where 0 < r < +∞, 0 < θ < π and 0 < φ < 2π. Shape: [direction x 3 (r, θ, phi)]
    """
    spherical_coordinates = zeros(cartesian_coordinates.shape)
    spherical_coordinates[:, 0] = norm(cartesian_coordinates, axis=1)
    spherical_coordinates[:, 1] = arccos(cartesian_coordinates[:, 2] / spherical_coordinates[:, 0])
    spherical_coordinates[:, 2] = arctan2(cartesian_coordinates[:, 1], cartesian_coordinates[:, 0])
    spherical_coordinates[spherical_coordinates[:, 2] < 0, 2] += 2 * pi
    return spherical_coordinates


def fibonacci_sphere(number_of_points: Optional[int] = 162) -> ndarray:
    """
    Returns cartesian coordinates of points distributed quasi-uniformly on a spherical manifold.

    Parameters
    ----------
    number_of_points
        Number of points.

    Returns
    -------
    cartesian_coordinates
        Cartesian coordinates. Shape: [direction x 3 (x, y, z)]
    """
    offset = 2. / number_of_points
    increment = pi * (3. - sqrt(5.))
    cartesian_coordinates = zeros((number_of_points, 3))
    for i in range(number_of_points):
        cartesian_coordinates[i, 1] = ((i * offset) - 1) + (offset / 2)
        r = sqrt(1 - cartesian_coordinates[i, 1] ** 2)
        phi = ((i + 1) % number_of_points) * increment
        cartesian_coordinates[i, 0] = r * cos(phi)
        cartesian_coordinates[i, 2] = r * sin(phi)
    return cartesian_coordinates
