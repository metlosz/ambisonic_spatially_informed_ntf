from copy import deepcopy
from functools import cached_property
from typing import Optional

from numpy import einsum, einsum_path, empty, log, ndarray, pi, sqrt, trace
from numpy.linalg import det, norm, pinv
from numpy.random import Generator

from asintf.EU import EU
from asintf.NTFBase import NTFBase
from asintf.PriorBase import PriorBase


class EU_IWLP(EU, PriorBase):
    """
    Non-negative Tensor Factorization with Ambisonic Spatial Covariance Matrix Model and Inverse Wishart Localization
    Prior, based on Euclidean distance [1].

    References
    ----------
    [1] M. Guzik and K. Kowalczyk, "On Ambisonic Source Separation with Spatially Informed Non-negative Tensor
    Factorization". (to be completed after publication)

    Notes
    -----
    The documentation only covers changes introduced in this class - for further description see the base class.
    """

    def __init__(self, covariance_matrices: ndarray, number_of_sources: int, components_per_source: int,
                 number_of_directions: int, standard_deviation, cartesian_coordinates: ndarray,
                 direct_to_reverb_ratio: float, degrees_of_freedom: float,
                 random_generator: Optional[Generator] = None):
        """
        Parameters
        ----------
        standard_deviation: float
            Standard deviation of the complex Gaussian distribution.
        cartesian_coordinates: ndarray
            Directions of arrival in Cartesian coordinate system. Shape: nb. of sources x 3 ([source x coordinate])
        direct_to_reverb_ratio: float
            Direct-to-reverb magnitude ratio.
        degrees_of_freedom: float
            Degrees of freedom of the Inverse Wishart distribution.
        """
        super().__init__(
            covariance_matrices, number_of_sources, components_per_source, number_of_directions, random_generator)
        # 0th order magnitude normalization to ensure constant prior strength
        self._R /= norm(sqrt(self._R[..., 0, 0].real))[..., None, None] ** 2
        self._std = standard_deviation
        self._doa = deepcopy(cartesian_coordinates)
        self._dtrr = direct_to_reverb_ratio
        self._nu = degrees_of_freedom

    def update_Z(self) -> None:
        XIinv = pinv(self._XI)
        z_n = einsum('jft, ftab, dab -> jd', self._V, self._R, self._S, optimize=self._Z_path[0])
        z_d = einsum('jft, ftab, dab -> jd', self._V, self._hatR.sum(0), self._S, optimize=self._Z_path[0])
        trXIinvS = einsum('jab, dab -> jd', XIinv, self._S, optimize=self._Z_path[1])
        trPsiXIinvSXIinv = einsum('jab, jab, dab, jab -> jd', self._Psi, XIinv, self._S, XIinv, optimize=self._Z_path[2])
        self._Z *= ((2 * z_n / (self._F * self._T * pi * self._std ** 2) + self._nu * trPsiXIinvSXIinv) /
                    (2 * z_d / (self._F * self._T * pi * self._std ** 2) + self._L * trPsiXIinvSXIinv +
                     (self._nu + self._L) * trXIinvS)).real
        NTFBase.update_Z(self)

    @property
    def cost_function(self) -> float:
        return ((pi * self._std ** 2) ** (-1) * super().cost_function +
                (self._nu + self._L) * log(det(self._XI)).sum() + (self._nu - self._L) *
                trace(self._Psi @ pinv(self._XI), axis1=-2, axis2=-1).sum()).real

    @cached_property
    def _Psi(self) -> ndarray:
        return self._calculate_prior_matrix(self._doa, self._L, self._dtrr)

    @cached_property
    def _Z_path(self):
        Z_path = [
            super()._Z_path,
            einsum_path('jab, dab -> jd', empty((self._J, self._L, self._L)), self._S, optimize='optimal')[0],
            einsum_path('jab, jab, dab, jab -> jd', self._Psi, empty((self._J, self._L, self._L)), self._S,
                        empty((self._J, self._L, self._L)), optimize='optimal')[0]
        ]
        return Z_path
