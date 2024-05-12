from typing import Tuple

from numpy import asarray, einsum, ndarray
from scipy.signal import istft as scipy_istft
from scipy.signal import stft as scipy_stft


def analysis(audio: ndarray, sampling_frequency: int, window: str, nperseg: int,
             noverlap: int) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Short Time Fourier Transform.
    
    Parameters
    ----------
    audio
        Multichannel audio file. Shape: [channel x sample]
    sampling_frequency
        Sampling frequency.
    window
        Analysis window. For details see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
    nperseg
        Number of samples per segment.
    noverlap
        Number of overlapping samples.
    
    Returns
    -------
    frequencies
        Discrete frequencies in Hertz.
    timestamps
        Timestamps in seconds.
    stft
        Short Time Fourier Transform coefficients. Shape: [channel x frequency x frame]
    """
    f, t, X = scipy_stft(audio, sampling_frequency, window, nperseg, noverlap)
    if audio.ndim <= 2:
        f, t, X = scipy_stft(audio, sampling_frequency, window, nperseg, noverlap)
    elif audio.ndim == 3:  # additional dimension for sound sources
        X = []
        for j in range(audio.shape[0]):
            f, t, _X = scipy_stft(audio[j], sampling_frequency, window, nperseg, noverlap)
            X.append(_X)
    return f, t, asarray(X)


def synthesis(stft: ndarray, sampling_frequency: int, window: str, nperseg: int, noverlap: int):
    """
    Inverse Short Time Fourier Transform.
    
    Parameters
    ----------
    stft
        Short Time Fourier Transform coefficients. Shape: [channel x frequency x frame]
    sampling_frequency
        Sampling frequency.
    window
        String specifying analysis window. For details see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html#scipy.signal.istft
    nperseg
        Number of samples per segment.
    noverlap
        Number of overlapping samples.
    
    Returns
    -------
    audio
        Multichannel audio file. Shape: [channel x sample] or [source x channel x sample]
    timestamps
        Timestamps in seconds.
    """
    if stft.ndim <= 3:
        t, x = scipy_istft(stft, sampling_frequency, window, nperseg, noverlap)
    elif stft.ndim == 4:  # additional dimension for sources
        x = []
        for j in range(stft.shape[0]):
            t, _x = scipy_istft(stft[j], sampling_frequency, window, nperseg, noverlap)
            x.append(_x)
    return t, asarray(x)


def magnitude_compression(stft: ndarray, compression_factor: int = 2) -> ndarray:
    """
    Magnitude compression of Short Time Fourier Transform coefficients.

    Parameters
    ----------
    stft
        Short Time Fourier Transform coefficients. Shape: [...]
    compression_factor
        Magnitude compression factor.

    Returns
    -------
    stft
        Short Time Fourier Transform coefficients. Shape: [...]
    """
    return abs(stft) ** (1 / compression_factor - 1) * stft


def estimate_covariance_matrices(stft: ndarray) -> ndarray:
    """
    Computes spatial covariance matrices from stacked Short Time Fourier Transform coefficients.

    Parameters
    ----------
    stft
        Short Time Fourier Transform coefficients. Shape: [channel x ...]

    Returns
    -------
    covariances
        Covariance matrices. Shape: [... x channel x channel]
    """
    return einsum('a..., b... -> ...ab', stft, stft.conj())
