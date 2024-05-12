from pickle import load as load_pickle
from typing import Tuple

from numpy import ndarray


def load_file(file_path: str) -> Tuple[int, ndarray, ndarray, ndarray]:
    """
    Loads and unpacks a file from the dataset.

    Parameters
    ----------
    file_path
        Path to dataset file.

    Returns
    -------
    sampling_frequency
        Sampling frequency.
    ambisonic_source_images
        Multichannel audio file. Shape: [channel x sample]
    early_ambisonic_source_images
        Multichannel audio file. Shape: [channel x sample]
    directions_of_arrival_cartesian
        Cartesian coordinates. Shape: [source x 3 (x, y, z)]
    """
    with open(file_path, 'rb') as pickle_file:
        loaded_dictionary = load_pickle(pickle_file)

    sampling_frequency = loaded_dictionary['fs']
    ambisonic_source_images = loaded_dictionary['s']
    early_ambisonic_source_images = loaded_dictionary['srly']
    directions_of_arrival_cartesian = loaded_dictionary['doa']

    return sampling_frequency, ambisonic_source_images, early_ambisonic_source_images, directions_of_arrival_cartesian
