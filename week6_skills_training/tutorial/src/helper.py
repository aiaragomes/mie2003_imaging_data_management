# -*- coding: utf-8 -*-
# Based on:
# https://pydicom.github.io/pydicom/stable/auto_examples/

import os
import pydicom
import numpy as np


def load_scan(files_path: int) -> np.array:
    """ Load DICOM files

    Parameters
    ----------
    files_path
        Directory path of DICOM scans of a particular subject

    Returns
    -------
    img3d
        3D matrix with all scan slices of a particular subject
    """
    # Read all slices
    slices = []
    for filename in np.sort(os.listdir(files_path)):
        filename = os.path.join(files_path, filename)
        slices.append(pydicom.dcmread(filename))

    # Ensure the slices are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # Create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # Fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    return img3d
