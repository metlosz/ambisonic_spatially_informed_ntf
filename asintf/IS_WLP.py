from copy import deepcopy
from functools import cached_property
from typing import List, Tuple, Union, Optional

from numpy import einsum, einsum_path, empty, log, ndarray, sqrt, trace
from numpy.linalg import det, norm, pinv
from numpy.random import Generator

from asintf.IS import IS
from asintf.NTFBase import NTFBase
from asintf.PriorBase import PriorBase


class IS_WLP(IS, PriorBase):
    """
    Non-negative Tensor Factorization with Ambisonic Spatial Covariance Matrix Model and Wishart Localization Prior,
    based on Itakura-Saito divergence [1].

    References
    ----------
    [1] M. Guzik and K. Kowalczyk, "On Ambisonic Source Separation with Spatially Informed Non-negative Tensor
    Factorization". (to be completed after publication)

    Notes
    -----
    The documentation only covers changes introduced in this class - for further description see the base class.
    """

    def __init__(self, covariance_matrices: ndarray, number_of_sources: int, components_per_source: int,
                 number_of_directions: int, cartesian_coordinates: ndarray, direct_to_reverb_ratio: float,
                 degrees_of_freedom: float, random_generator: Optional[Generator] = None):
        """
        Parameters
        ----------
        cartesian_coordinates: ndarray
            Directions of arrival in Cartesian coordinate system. Shape: source x 3 coordinates (x, y, z)
        direct_to_reverb_ratio: float
            Direct-to-reverb magnitude ratio.
        degrees_of_freedom: float
            Degrees of freedom of the Wishart distribution.
        """
        super().__init__(
            covariance_matrices, number_of_sources, components_per_source, number_of_directions, random_generator)
        # 0th order magnitude normalization to ensure constant prior strength
        self._R /= norm(sqrt(self._R[..., 0, 0].real))[..., None, None] ** 2
        self._doa = deepcopy(cartesian_coordinates)
        self._dtrr = direct_to_reverb_ratio
        self._nu = degrees_of_freedom

    def update_Z(self) -> None:
        hatRinv = pinv(self._hatR.sum(0))
        z_n = trace(einsum('jft, ftab, ftbc, ftce, deg -> jdag', self._V, hatRinv, self._R, hatRinv, self._S,
                           optimize=self._Z_path[0]), axis1=-1, axis2=-2)
        z_d = einsum('jft, ftab, dab -> jd', self._V, hatRinv, self._S, optimize=self._Z_path[1])
        trXIinvS = einsum('jab, dab -> jd', pinv(self._XI), self._S, optimize=self._Z_path[2])
        self._Z *= ((z_n / (self._F * self._T) + self._nu * trXIinvS) /
                    (z_d / (self._F * self._T) + self._L * trXIinvS + self._nu * self._trPsiinvS)).real
        NTFBase.update_Z(self)

    @property
    def cost_function(self) -> float:
        return (super().cost_function + self._nu * trace(self._Psiinv @ self._XI, axis1=-2, axis2=-1).sum() -
                (self._nu - self._L) * log(det(self._XI)).sum()).real

    @cached_property
    def _Psi(self) -> ndarray:
        return self._calculate_prior_matrix(self._doa, self._L, self._dtrr)

    @cached_property
    def _Psiinv(self) -> ndarray:
        return pinv(self._Psi)

    @cached_property
    def _trPsiinvS(self) -> ndarray:
        return einsum('jab, dab -> jd', self._Psiinv, self._S)

    @cached_property
    def _Z_path(self) -> List[List[Union[str, Tuple[int]]]]:
        Z_path = super()._Z_path
        Z_path.append(einsum_path('jab, dab -> jd', empty((self._J, self._L, self._L)), self._S, optimize='optimal')[0])
        return Z_path
