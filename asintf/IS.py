from functools import cached_property

from numpy import einsum, einsum_path, empty, log, trace
from numpy.linalg import det, pinv

from asintf.NTFBase import NTFBase


class IS(NTFBase):
    """
    Non-negative Tensor Factorization with Ambisonic Spatial Covariance Matrix Model, based on Itakura-Saito divergence
    [1].

    References
    ----------
    [1] A. J. Muñoz-Montoro, J. J. Carabias-Orti and P. Vera-Candeas, "Ambisonics domain Singing Voice Separation
    combining Deep Neural Network and Direction Aware Multichannel NMF," 2021 IEEE 23rd International Workshop on
    Multimedia Signal Processing (MMSP), Tampere, Finland, 2021, pp. 1-6, doi: 10.1109/MMSP53017.2021.9733494.

    Notes
    -----
    The documentation only covers changes introduced in this class - for further description see the base class.
    """

    def update_Q(self):
        hatRinv = pinv(self._hatR.sum(0))
        q_n = trace(einsum('fk, tk, ftab, ftbc, ftce, jeg -> jkag', self._W, self._H, hatRinv, self._R, hatRinv,
                           self._XI, optimize=self._Q_path[0]), axis1=-1, axis2=-2)
        q_d = einsum('fk, tk, ftab, jab -> jk', self._W, self._H, hatRinv, self._XI, optimize=self._Q_path[1])
        self._Q *= (q_n / q_d).real
        super().update_Q()

    def update_W(self):
        hatRinv = pinv(self._hatR.sum(0))
        w_n = trace(einsum('jk, tk, ftab, ftbc, ftce, jeg -> fkag', self._Q, self._H, hatRinv, self._R, hatRinv,
                           self._XI, optimize=self._W_path[0]), axis1=-1, axis2=-2)
        w_d = einsum('jk, tk, ftab, jab -> fk', self._Q, self._H, hatRinv, self._XI, optimize=self._W_path[1])
        self._W *= (w_n / w_d).real
        super().update_W()

    def update_H(self):
        hatRinv = pinv(self._hatR.sum(0))
        h_n = trace(einsum('jk, fk, ftab, ftbc, ftce, jeg -> tkag', self._Q, self._W, hatRinv, self._R, hatRinv,
                           self._XI, optimize=self._H_path[0]), axis1=-1, axis2=-2)
        h_d = einsum('jk, fk, ftab, jab -> tk', self._Q, self._W, hatRinv, self._XI, optimize=self._H_path[1])
        self._H *= (h_n / h_d).real
        super().update_H()

    def update_Z(self):
        hatRinv = pinv(self._hatR.sum(0))
        z_n = trace(einsum('jft, ftab, ftbc, ftce, deg -> jdag', self._V, hatRinv, self._R, hatRinv, self._S,
                           optimize=self._Z_path[0]), axis1=-1, axis2=-2)
        z_d = einsum('jft, ftab, dab -> jd', self._V, hatRinv, self._S, optimize=self._Z_path[1])
        self._Z *= (z_n / z_d).real
        super().update_Z()

    @property
    def cost_function(self) -> float:
        return (trace(self._R @ pinv(self._hatR.sum(0)), axis1=-1, axis2=-2) + log(det(self._hatR.sum(0)))).mean().real

    @cached_property
    def _H_path(self):
        H_path = [
            einsum_path('jk, fk, ftab, ftbc, ftce, jeg -> tkag', self._Q, self._W,
                        empty((self._F, self._T, self._L, self._L)), self._R,
                        empty((self._F, self._T, self._L, self._L)), self._XI, optimize='optimal')[0],
            einsum_path('jk, fk, ftab, jab -> tk', self._Q, self._W, empty((self._F, self._T, self._L, self._L)),
                        self._XI, optimize='optimal')[0]
        ]
        return H_path

    @cached_property
    def _Q_path(self):
        Q_path = [
            einsum_path('fk, tk, ftab, ftbc, ftce, jeg -> jkag', self._W, self._H,
                        empty((self._F, self._T, self._L, self._L)), self._R,
                        empty((self._F, self._T, self._L, self._L)), self._XI, optimize='optimal')[0],
            einsum_path('fk, tk, ftab, jab -> jk', self._W, self._H, empty((self._F, self._T, self._L, self._L)),
                        self._XI, optimize='optimal')[0]
        ]
        return Q_path

    @cached_property
    def _W_path(self):
        W_path = [
            einsum_path('jk, tk, ftab, ftbc, ftce, jeg -> fkag', self._Q, self._H,
                        empty((self._F, self._T, self._L, self._L)), self._R,
                        empty((self._F, self._T, self._L, self._L)), self._XI, optimize='optimal')[0],
            einsum_path('jk, tk, ftab, jab -> fk', self._Q, self._H, empty((self._F, self._T, self._L, self._L)),
                        self._XI, optimize='optimal')[0]
        ]
        return W_path


    @cached_property
    def _Z_path(self):
        Z_path = [
            einsum_path('jft, ftab, ftab, ftab, dab -> jd', self._V, empty((self._F, self._T, self._L, self._L)),
                        self._R, empty((self._F, self._T, self._L, self._L)), self._S, optimize='optimal')[0],
            einsum_path('jft, ftab, dab -> jd', self._V, empty((self._F, self._T, self._L, self._L)), self._S,
                        optimize='optimal')[0]
        ]
        return Z_path
