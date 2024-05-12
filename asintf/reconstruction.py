from numpy import einsum, eye, ndarray
from numpy.linalg import pinv


def mimo_mwf(stft: ndarray, covariance_matrices: ndarray) -> ndarray:
    """
    Multiple-Input Multiple-Output Multichannel Wiener Filter.

    Parameters
    ----------
    stft
        Short Time Fourier Transform coefficients. Shape: [channel x frequency x frame]
    covariance_matrices
        Covariance matrices. Shape: [source x frequency x frame x channel x channel]

    Returns
    -------
    source_signals
        Reconstructed source images. Shape: [source x channel x frequency x frame]
    """
    return einsum('ftab, jftbc, cft-> jaft', pinv(covariance_matrices.sum(0)), covariance_matrices, stft, optimize=True)


def pwd(stft: ndarray, steering_vectors: ndarray) -> ndarray:
    """
    Plane Wave Decomposition beamformer.

    Parameters
    ----------
    stft
        Short Time Fourier Transform coefficients. Shape: [channel x frequency x frame]
    steering_vectors
        Steering vectors. Shape: [source x frequency x frame x channel]

    Returns
    -------
    source_signals
        Reconstructed source signals. Shape: [source x frequency x frame]
    """
    return einsum('jftl, lft -> jft', steering_vectors, stft, optimize=True)


def mimo_pwd(stft: ndarray, steering_vectors: ndarray) -> ndarray:
    """
    Multiple-Input Multiple-Output Plane Wave Decomposition beamformer.

    Parameters
    ----------
    stft
        Short Time Fourier Transform coefficients. Shape: [channel x frequency x frame]
    steering_vectors
        Steering vectors. Shape: [source x frequency x frame x channel]

    Returns
    -------
    source_signals
        Reconstructed source images. Shape: [source x channel x frequency x frame]
    """
    return einsum('jftl, jft -> jlft', steering_vectors, pwd(stft, steering_vectors), optimize=True)


def pwd_mimo_mwf(stft: ndarray, steering_vectors: ndarray) -> ndarray:
    """
    Plane Wave Decomposition beamformer followed by Multiple-Input Multiple-Output Multichannel Wiener Filter.

    Parameters
    ----------
    stft
        Short Time Fourier Transform coefficients. Shape: [channel x frequency x frame]
    steering_vectors
        Steering vectors. Shape: [source x frequency x frame x channel]

    Returns
    -------
    source_signals
        Reconstructed source images. Shape: [source x channel x frequency x frame]
    """
    return mimo_mwf(
        stft, abs(pwd(stft, steering_vectors))[..., None, None] * eye(steering_vectors.shape[-1])[None, None, None])
