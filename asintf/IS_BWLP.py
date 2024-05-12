from typing import Optional

from numpy import ndarray
from numpy.random import Generator

from asintf.IS_WLP import IS_WLP
from asintf.stft import estimate_covariance_matrices


class IS_BWLP(IS_WLP):
    """
    Non-negative Tensor Factorization with Ambisonic Spatial Covariance Matrix Model and Wishart Localization Prior,
    based on Itakura-Saito divergence with blind estimation of prior hyperparameters [1].

    References
    ----------
    [1] M. Guzik and K. Kowalczyk, "On Ambisonic Source Separation with Spatially Informed Non-negative Tensor
    Factorization". (to be completed after publication)
    Notes
    -----
    The documentation only covers changes introduced in this class - for further description see the base class.
    """

    def __init__(self, stft: ndarray, number_of_sources: int, components_per_source: int,
                 number_of_directions: int, directions_of_arrival_cartesian: ndarray,
                 random_generator: Optional[Generator] = None):
        """

        Parameters
        ----------
        stft
            Multichannel Short Time Fourier Transform coefficients. Shape: [channel x frequency x frame]
        """
        direct_to_reverb_ratio = self._estimate_direct_to_reverb_ratio(
            stft, directions_of_arrival_cartesian)
        degrees_of_freedom = self._estimate_degrees_of_freedom(stft, direct_to_reverb_ratio,
                                                               directions_of_arrival_cartesian)
        covariance_matrices = estimate_covariance_matrices(stft)
        super().__init__(covariance_matrices, number_of_sources, components_per_source, number_of_directions,
                         directions_of_arrival_cartesian, direct_to_reverb_ratio, degrees_of_freedom, random_generator)
