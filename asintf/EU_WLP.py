from copy import deepcopy
from functools import cached_property
from typing import List, Optional, Tuple, Union

from numpy import einsum, einsum_path, empty, log, ndarray, pi, sqrt, trace
from numpy.linalg import det, norm, pinv
from numpy.random import Generator

from asintf.EU import EU
from asintf.NTFBase import NTFBase
from asintf.PriorBase import PriorBase


class EU_WLP(EU, PriorBase):
    """
    Non-negative Tensor Factorization with Ambisonic Spatial Covariance Matrix Model and Wishart Localization Prior,
    based on Euclidean distance [1], [2].

    References
    ----------
    [1] M. Guzik and K. Kowalczyk, "Wishart Localization Prior On Spatial Covariance Matrix In Ambisonic Source
    Separation Using Non-Negative Tensor Factorization", ICASSP 2022 - 2022 IEEE International Conference on Acoustics,
    Speech and Signal Processing (ICASSP), Singapore, Singapore, 2022, pp. 446-450,
    doi: 10.1109/ICASSP43922.2022.9746222.
    [2] M. Guzik and K. Kowalczyk, "On Ambisonic Source Separation with Spatially Informed Non-negative Tensor
    Factorization". (to be completed after publication)

    Notes
    -----
    The documentation only covers changes introduced in this class - for further description see the base class.
    """

    def __init__(self, covariance_matrices: ndarray, number_of_sources: int, components_per_source: int,
                 number_of_directions: int, standard_deviation, directions_of_arrival_cartesian: ndarray,
                 direct_to_reverb_ratio: float, degrees_of_freedom: float,
                 random_generator: Optional[Generator] = None):
        """
        Parameters
        ----------
        standard_deviation: float
            Standard deviation of the complex Gaussian distribution.
        directions_of_arrival_cartesian: ndarray
            Directions of arrival in Cartesian coordinate system. Shape: source x 3 (x, y, z)
        direct_to_reverb_ratio: float
            Direct-to-reverb magnitude ratio.
        degrees_of_freedom: float
            Degrees of freedom of the Wishart distribution.
        """
        super().__init__(
            covariance_matrices, number_of_sources, components_per_source, number_of_directions, random_generator)
        # 0th order magnitude normalization to ensure constant prior strength
        self._R /= norm(sqrt(self._R[..., 0, 0].real))[..., None, None] ** 2
        self._std = standard_deviation
        self._doa = deepcopy(directions_of_arrival_cartesian)
        self._dtrr = direct_to_reverb_ratio
        self._nu = degrees_of_freedom

    def update_Z(self) -> None:
        zn = einsum('jft, ftab, dab -> jd', self._V, self._R, self._S, optimize=self._Z_path[0])
        zd = einsum('jft, ftab, dab -> jd', self._V, self._hatR.sum(0), self._S, optimize=self._Z_path[0])
        trXIinvS = einsum('jab, dab -> jd', pinv(self._XI), self._S, optimize=self._Z_path[1])
        self._Z *= ((2 * zn / (self._F * self._T * pi * self._std ** 2) + self._nu * trXIinvS) /
                    (2 * zd / (self._F * self._T * pi * self._std ** 2) + self._L * trXIinvS +
                     self._nu * self._trPsiinvS)).real
        NTFBase.update_Z(self)

    @property
    def cost_function(self) -> float:
        return ((pi * self._std ** 2) ** (-1) * super().cost_function +
                self._nu * trace(self._Psiinv @ self._XI, axis1=-2, axis2=-1).sum() +
                (self._L - self._nu) * log(det(self._XI)).sum()).real

    @cached_property
    def _Psi(self) -> ndarray:
        return self._calculate_prior_matrix(self._doa, self._L, self._dtrr)

    @cached_property
    def _Psiinv(self) -> ndarray:
        return pinv(self._Psi)

    @cached_property
    def _trPsiinvS(self) -> ndarray:
        return einsum('jab, dab -> jd', self._Psiinv, self._S, optimize='optimal')

    @cached_property
    def _Z_path(self) -> List[List[Union[str, Tuple[int]]]]:
        Z_path = [
            super()._Z_path,
            einsum_path('jab, dab -> jd', empty((self._J, self._L, self._L)), self._S, optimize='optimal')[0]
        ]
        return Z_path
