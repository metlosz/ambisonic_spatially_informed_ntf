from abc import ABC

from numpy import ndarray, eye, arange, einsum, trace, asarray, isfinite, argmax
from numpy.linalg import norm
from numpy.linalg import pinv

from asintf.geometry import cartesian_to_spherical
from asintf.reconstruction import mimo_pwd, pwd_mimo_mwf
from asintf.spherical_harmonics import matrix, number_of_channels_to_order
from asintf.wishart import pdf


class PriorBase(ABC):
    @staticmethod
    def _calculate_prior_matrix(directions_of_arrival_cartesian: ndarray, number_of_channels: int,
                                direct_to_reverb_ratio: float) -> ndarray:
        directions_of_arrival_spherical = cartesian_to_spherical(directions_of_arrival_cartesian)
        order = number_of_channels_to_order(number_of_channels)
        spherical_harmonic_matrix = matrix(directions_of_arrival_spherical[:, 1:], order)
        Psi = spherical_harmonic_matrix[..., None] @ spherical_harmonic_matrix[:, None]
        Psi /= Psi[..., 0, 0, None, None]
        Psi += direct_to_reverb_ratio ** (-1) * eye(number_of_channels)[None]
        Psi /= Psi[..., 0, 0, None, None]
        return Psi

    @staticmethod
    def _estimate_direct_to_reverb_ratio(stft: ndarray, directions_of_arrival_cartesian: ndarray) -> float:
        number_of_channels = stft.shape[0]
        order = number_of_channels_to_order(number_of_channels)
        directions_of_arrival_spherical = cartesian_to_spherical(directions_of_arrival_cartesian)
        spherical_harmonic_matrix = matrix(directions_of_arrival_spherical[:, 1:], order)
        spherical_harmonic_matrix /= norm(spherical_harmonic_matrix, axis=1)[:, None]
        direct_stft = mimo_pwd(stft, spherical_harmonic_matrix[:, None, None]).sum(axis=0)
        reverberant_stft = stft - direct_stft
        direct_to_reverb_ratio = norm(direct_stft) ** 2 / norm(reverberant_stft) ** 2
        return direct_to_reverb_ratio

    @staticmethod
    def _estimate_degrees_of_freedom(stft: ndarray, direct_to_reverb_ratio: float,
                                     directions_of_arrival_cartesian: ndarray) -> float:
        number_of_channels, number_of_frequencies, number_of_frames = stft.shape
        order = number_of_channels_to_order(number_of_channels)
        directions_of_arrival_spherical = cartesian_to_spherical(directions_of_arrival_cartesian)
        spherical_harmonic_matrix = matrix(directions_of_arrival_spherical[:, 1:], order)

        Psi = spherical_harmonic_matrix[..., None] @ spherical_harmonic_matrix[:, None]
        Psi /= Psi[..., 0, 0, None, None]
        Psi += direct_to_reverb_ratio ** (-1) * eye(number_of_channels)[None]
        Psi /= Psi[..., 0, 0, None, None]

        direct_stft = pwd_mimo_mwf(stft, spherical_harmonic_matrix[:, None, None])
        direct_covariance_matrices = einsum('jaft, jbft -> jab', direct_stft, direct_stft.conj())
        direct_covariance_matrices /= direct_covariance_matrices[..., 0, 0, None, None]

        probability_list = []
        dn = 0.1
        nu_grid = arange(dn, 10 * number_of_channels, dn)
        for nu in nu_grid:
            Psi_scaled = 1 / nu * Psi.copy()
            alpha = number_of_channels * nu / trace(
                pinv(Psi_scaled) @ direct_covariance_matrices, axis1=-1, axis2=-2).real
            jacobian = alpha ** (number_of_channels ** 2)
            probability = jacobian + pdf(alpha[..., None, None] * direct_covariance_matrices, Psi_scaled, nu)
            probability_list.append(sum(probability.real))

        indices = isfinite(probability_list)
        probability_list = asarray(probability_list)[indices]
        nu_grid = nu_grid[indices]
        nu = nu_grid[argmax(probability_list)]
        return nu.item()
