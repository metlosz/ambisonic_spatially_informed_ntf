from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property
from typing import List, Optional, Union, Tuple

from numpy import einsum, einsum_path, ndarray
from numpy.random import default_rng, Generator

from asintf.geometry import cartesian_to_spherical, fibonacci_sphere
from asintf.spherical_harmonics import matrix, number_of_channels_to_order


class NTFBase(ABC):
    """
    Attributes
    ----------
    covariance_matrices
        Estimated covariance matrices. Shape: [source x frequency x frame x channel x channel]
    cost_function
        Current value of the cost function.
    spatial_covariance_matrices
        Estimated spatial covariance matrices. Shape: [source x channel x channel]
    spectrograms
        Estimated spectrograms. Shape: [source x frequency x frame]
    """

    def __init__(self, covariance_matrices: ndarray, number_of_sources: int, components_per_source: int,
                 number_of_directions: int, random_generator: Optional[Generator] = None) -> None:
        """
        Parameters
        ----------
        covariance_matrices
            Empirical covariance matrices. Shape: [frequency x frame x channel x channel]
        number_of_sources
            Number of sources.
        components_per_source
            Number of components per source.
        number_of_directions
            Number of directions.
        random_generator
            Random number generator. For details see: https://numpy.org/doc/stable/reference/random/generator.html
        """
        self._R = deepcopy(covariance_matrices)
        self._J = number_of_sources
        self._Kpj = components_per_source
        self._D = number_of_directions
        if random_generator is None:
            random_generator = default_rng()
        self._rnd_gn = random_generator
        self._F, self._T, self._L, _ = self._R.shape
        self._initialize_QWHZ()
        self._calculate_V()
        self._calculate_XI()
        self._calculate_hatR()

    def _initialize_QWHZ(self) -> None:
        self._Q = self._rnd_gn.random((self._J, self._K))
        self._W = self._rnd_gn.random((self._F, self._K))
        self._H = self._rnd_gn.random((self._T, self._K))
        self._Z = self._rnd_gn.random((self._J, self._D))
        self._normalize_QWHZ()

    def set_Q(self, Q: ndarray) -> None:
        self._Q = deepcopy(Q)
        self._normalize_QWHZ()
        self._calculate_V()
        self._calculate_hatR()

    def set_W(self, W: ndarray) -> None:
        self._W = deepcopy(W)
        self._normalize_QWHZ()
        self._calculate_V()
        self._calculate_hatR()

    def set_H(self, H: ndarray) -> None:
        self._H = deepcopy(H)
        self._normalize_QWHZ()
        self._calculate_V()
        self._calculate_hatR()

    def set_Z(self, Z: ndarray) -> None:
        self._Z = deepcopy(Z)
        self._normalize_QWHZ()
        self._calculate_V()
        self._calculate_XI()
        self._calculate_hatR()

    def _normalize_QWHZ(self) -> None:
        self._Q *= self._Z.sum(axis=-1)[..., None]
        self._Z /= self._Z.sum(axis=-1)[..., None]
        self._W *= self._Q.sum(axis=0)[None]
        self._Q /= self._Q.sum(axis=0)[None]
        self._H *= self._W.sum(axis=0)[None]
        self._W /= self._W.sum(axis=0)[None]

    def _calculate_V(self) -> None:
        self._V = einsum('jk, fk, tk -> jft', self._Q, self._W, self._H, optimize=self._V_path)

    def _calculate_XI(self) -> None:
        self._XI = einsum('jd, dab -> jab', self._Z, self._S, optimize=self._XI_path)

    def _calculate_hatR(self) -> None:
        self._hatR = einsum('jft, jab -> jftab', self._V, self._XI, optimize=self._hatR_path)

    def iteration(self) -> None:
        self.update_Q()
        self.update_W()
        self.update_H()
        self.update_Z()

    @abstractmethod
    def update_Q(self) -> None:
        self._normalize_QWHZ()
        self._calculate_V()
        self._calculate_hatR()

    @abstractmethod
    def update_W(self) -> None:
        self._normalize_QWHZ()
        self._calculate_V()
        self._calculate_hatR()

    @abstractmethod
    def update_H(self) -> None:
        self._normalize_QWHZ()
        self._calculate_V()
        self._calculate_hatR()

    @abstractmethod
    def update_Z(self) -> None:
        self._normalize_QWHZ()
        self._calculate_V()
        self._calculate_XI()
        self._calculate_hatR()

    @property
    def covariance_matrices(self) -> ndarray:
        return deepcopy(self._hatR)

    @property
    @abstractmethod
    def cost_function(self) -> float:
        raise NotImplementedError

    @property
    def spatial_covariance_matrices(self) -> ndarray:
        return deepcopy(self._XI)

    @property
    def spectrograms(self) -> ndarray:
        return deepcopy(self._V)

    @cached_property
    def _hatR_path(self) -> List[Union[str, Tuple[int]]]:
        return einsum_path('jft, jab -> jftab', self._V, self._XI, optimize='optimal')[0]

    @cached_property
    def _K(self) -> int:
        return int(self._J * self._Kpj)

    @cached_property
    def _S(self) -> ndarray:
        doa_gc = fibonacci_sphere(self._D)
        doa_gs = cartesian_to_spherical(doa_gc)
        order = number_of_channels_to_order(self._L)
        shm = matrix(doa_gs[:, 1:], order)
        S = shm[..., None] @ shm[:, None]
        S /= S[..., 0, 0, None, None]  # 0th order normalization
        return S

    @cached_property
    def _V_path(self) -> List[Union[str, Tuple[int]]]:
        return einsum_path('jk, fk, tk -> jft', self._Q, self._W, self._H, optimize='optimal')[0]

    @cached_property
    def _XI_path(self) -> List[Union[str, Tuple[int]]]:
        return einsum_path('jd, dab -> jab', self._Z, self._S, optimize='optimal')[0]
