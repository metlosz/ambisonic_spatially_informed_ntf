from functools import cached_property
from typing import List, Union, Tuple

from numpy import einsum, einsum_path
from numpy.linalg import norm

from asintf.NTFBase import NTFBase


class EU(NTFBase):
    """
    Non-negative Tensor Factorization with Ambisonic Spatial Covariance Matrix Model, based on Euclidean distance [1].

    References
    ----------
    [1] J. Nikunen and A. Politis, "Multichannel NMF for Source Separation with Ambisonic Signals", 2018 16th
    International Workshop on Acoustic Signal Enhancement (IWAENC), Tokyo, Japan, 2018, pp. 251-255,
    doi: 10.1109/IWAENC.2018.8521344.

    Notes
    -----
    The documentation only covers changes introduced in this class - for further description see the base class.
    """

    def update_Q(self):
        q_n = einsum('fk, tk, ftab, jab -> jk', self._W, self._H, self._R, self._XI, optimize=self._Q_path)
        q_d = einsum('fk, tk, ftab, jab -> jk', self._W, self._H, self._hatR.sum(0), self._XI, optimize=self._Q_path)
        self._Q *= (q_n / q_d).real
        super().update_Q()

    def update_W(self):
        w_n = einsum('jk, tk, ftab, jab -> fk', self._Q, self._H, self._R, self._XI, optimize=self._W_path)
        w_d = einsum('jk, tk, ftab, jab -> fk', self._Q, self._H, self._hatR.sum(0), self._XI, optimize=self._W_path)
        self._W *= (w_n / w_d).real
        super().update_W()

    def update_H(self):
        h_n = einsum('jk, fk, ftab, jab -> tk', self._Q, self._W, self._R, self._XI, optimize=self._H_path)
        h_d = einsum('jk, fk, ftab, jab -> tk', self._Q, self._W, self._hatR.sum(0), self._XI, optimize=self._H_path)
        self._H *= (h_n / h_d).real
        super().update_H()

    def update_Z(self):
        z_n = einsum('jft, ftab, dab -> jd', self._V, self._R, self._S, optimize=self._Z_path)
        z_d = einsum('jft, ftab, dab -> jd', self._V, self._hatR.sum(0), self._S, optimize=self._Z_path)
        self._Z *= (z_n / z_d).real
        super().update_Z()

    @property
    def cost_function(self) -> float:
        return (norm(self._R - self._hatR.sum(0), axis=(-1, -2)) ** 2).mean().real

    @cached_property
    def _H_path(self) -> List[Union[str, Tuple[int]]]:
        return einsum_path('jk, fk, ftab, jab -> tk', self._Q, self._W, self._R, self._XI, optimize='optimal')[0]

    @cached_property
    def _Q_path(self) -> List[Union[str, Tuple[int]]]:
        return einsum_path('fk, tk, ftab, jab -> jk', self._W, self._H, self._R, self._XI, optimize='optimal')[0]

    @cached_property
    def _W_path(self) -> List[Union[str, Tuple[int]]]:
        return einsum_path('jk, tk, ftab, jab -> fk', self._Q, self._H, self._R, self._XI, optimize='optimal')[0]

    @cached_property
    def _Z_path(self) -> List[Union[str, Tuple[int]]]:
        return einsum_path('jft, ftab, dab -> jd', self._V, self._R, self._S, optimize='optimal')[0]
