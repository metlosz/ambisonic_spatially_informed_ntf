from typing import Optional

from IPython.display import Audio, display
from ipywidgets.widgets import Output, VBox
from numpy import ndarray
from spaudiopy.decoder import magls_bin, sh2bin
from spaudiopy.io import load_hrirs

from asintf.spherical_harmonics import number_of_channels_to_order


def player(audio: ndarray, sampling_frequency: int, title: Optional[str] = None) -> None:
    """
    Creates an audio player.

    Parameters
    ----------
    audio
        Multichannel audio file. Shape: [channel x sample]
    sampling_frequency
        Sampling frequency.
    title
        If given, it is displayed above the player.
    """
    widget_list = []

    if title is not None:
        out = Output()
        with out:
            print(title)
        widget_list.append(out)

    out = Output()
    with out:
        display(Audio(audio, rate=sampling_frequency))
    widget_list.append(out)

    display(VBox(widget_list))


def binauralize(audio: ndarray, sampling_frequency: int, order_limit: Optional[int] = None) -> ndarray:
    """
    Binauralize an Ambisonic audio file with the MagLS method.

    Notes
    -----
    This function uses the spaudiopy package: https://github.com/chris-hld/spaudiopy

    Parameters
    ----------
    audio
        Multichannel audio file. Shape: [channel x sample]
    sampling_frequency
        Sampling frequency.
    order_limit
        If given, it limits the binauralization order.

    Returns
    -------
    audio:
        Binauralized audio file. Shape: [2 (left channel, right channel) x sample]
    """
    if order_limit is None:
        order_limit = number_of_channels_to_order(audio.shape[0])
    hrirs = load_hrirs(sampling_frequency)
    hrirs = magls_bin(hrirs, order_limit)
    return sh2bin(audio, hrirs)
