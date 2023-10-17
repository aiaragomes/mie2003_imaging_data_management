# -*- coding: utf-8 -*-
# Based on:
# https://pydicom.github.io/pydicom/stable/auto_examples/
# https://github.com/zhenweishi/O-RAW/

import os
import pydicom
import numpy as np
import SimpleITK as sitk
from skimage import draw


def load_scan(files_path: int) -> list:
    """ Load DICOM files

    Parameters
    ----------
    files_path
        Directory path of DICOM scans of a particular subject

    Returns
    -------
    slices
        List with all slices for a particular subject
    """
    # Read all slices
    slices = []
    for filename in np.sort(os.listdir(files_path)):
        filename = os.path.join(files_path, filename)
        slices.append(pydicom.dcmread(filename))

    # Ensure the slices are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    return slices


def get_pixels_hu(slices: list) -> np.array:
    """ Stack slices into a 3D-array and convert pixels to Hounsfield Unit (HU)

    Parameters
    ----------
    slices
        List with all slices for a particular subject

    Returns
    -------
    image
        3D matrix with all scan slices of a particular subject in HU
    """
    # Stack all slices
    image = np.stack([s.pixel_array for s in slices])

    # Get slope and intercept
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    # Convert to Hounsfield Unit
    if slope != 1:
        image = slope * image.astype(np.float64)
    image += np.int16(intercept)
    image = image.astype(np.float32)

    return image


def get_ct_image(slices: list):
    """ Convert CT slices into an image and transform pixels to Hounsfield Unit

    Parameters
    ----------
    slices
        List with all slices for a particular subject

    Returns
    -------
    image
        SimpleITK image of DICOM CT scan
    """
    # Get 3D array with pixels in Hounsfield Unit
    image = get_pixels_hu(slices)

    # Get spacing: slices usually have the same basic information including
    # slice size, patient position, etc
    x_resolution = np.float64(np.array(slices[0].PixelSpacing[0]))
    y_resolution = np.float64(np.array(slices[0].PixelSpacing[1]))
    slice_thickness = np.float64(np.abs(
        slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]
    ))

    # Convert 3D array to image
    image = sitk.GetImageFromArray(image)
    image.SetSpacing([x_resolution, y_resolution, slice_thickness])

    return image


def get_roi_id(rtstruct, roi_name: str = 'GTV') -> int:
    """ Match ROI id in RTSTURCT to a given ROI name in the parameter file

    Parameters
    ----------
    rtstruct
        DICOM RTSTRUCT
    roi_name
        Name of the region of interest from the RTSTRUCT

    Returns
    -------
    roi_id
        ID of the region of interest
    """
    roi_sequence = rtstruct.StructureSetROISequence
    for i in range(len(roi_sequence)):
        if roi_name == roi_sequence[i].ROIName:
            roi_number = roi_sequence[i].ROINumber
            break

    roi_contour = rtstruct.ROIContourSequence
    for roi_id in range(len(roi_sequence)):
        if roi_number == roi_contour[roi_id].ReferencedROINumber:
            break

    return roi_id


def poly2mask(vertex_x: np.array, vertex_y: np.array, shape: list) -> np.array:
    """ Mask interpolation

    Parameters
    ----------
    vertex_x
        Vertex X coordinates
    vertex_y
        Vertex Y coordinates
    shape
        XY shape of image

    Returns
    -------
    mask
        2D binary mask
    """
    fill_x_coords, fill_y_coords = draw.polygon(vertex_x, vertex_y, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[fill_x_coords, fill_y_coords] = True
    return mask


def get_gtv_mask(slices: list, rtstruct_path: str):
    """ Convert RTSTRUCT to a binary mask and transform into an image

    Parameters
    ----------
    slices
        List with all slices for a particular subject
    rtstruct_path
        Path to the DICOM RTSTRUCT

    Returns
    -------
    mask
        SimpleITK image of binary mask
    """
    # Create 3D array for binary mask: slices usually have the same basic
    # information including slice size, patient position, etc
    n_slices = len(slices)
    rows = slices[0].Rows
    columns = slices[0].Columns
    mask = np.zeros([n_slices, rows, columns], dtype=np.uint8)

    # Get spacing
    x_resolution = np.float64(np.array(slices[0].PixelSpacing[0]))
    y_resolution = np.float64(np.array(slices[0].PixelSpacing[1]))
    slice_thickness = np.float64(np.abs(
        slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]
    ))

    # Load RTSTRUCT
    rtstruct = pydicom.dcmread(rtstruct_path, force=True)

    # Get ID of region of interest (ROI)
    roi_name = 'GTV'
    roi_id = get_roi_id(rtstruct, roi_name)

    # Loop through contour sequence to construct binary mask
    roi_contour_sequence = rtstruct.ROIContourSequence[roi_id].ContourSequence
    for k in range(len(roi_contour_sequence)):
        # Current contour data and position
        contour = roi_contour_sequence[k].ContourData
        cposition = round(contour[2], 1)

        # Match countour with its corresponding slice
        sliceOK = None
        for i in range(n_slices):
            pposition = round(slices[i].ImagePositionPatient[2], 1)
            diff = np.abs(pposition - cposition)
            if diff <= slice_thickness/2:
                sliceOK = i
                break

        # Stop if no match was found
        if not sliceOK:
            break

        # Creating binary mask
        x = []
        y = []
        z = []
        for i in range(0, len(contour), 3):
            x.append(contour[i+1])
            y.append(contour[i+0])
            z.append(contour[i+2])
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        x -= slices[0].ImagePositionPatient[1]
        y -= slices[0].ImagePositionPatient[0]
        z -= slices[0].ImagePositionPatient[2]

        pts = np.zeros([len(x), 3])
        pts[:, 0] = x
        pts[:, 1] = y
        pts[:, 2] = z

        slice = slices[sliceOK]
        m = np.zeros([2, 2])
        m[0, 0] = slice.ImageOrientationPatient[0]*x_resolution
        m[0, 1] = slice.ImageOrientationPatient[3]*y_resolution
        m[1, 0] = slice.ImageOrientationPatient[1]*x_resolution
        m[1, 1] = slice.ImageOrientationPatient[4]*y_resolution

        # Transform points from reference frame to image coordinates
        m_inv = np.linalg.inv(m)
        pts = (np.matmul((m_inv), (pts[:, [0, 1]]).T)).T
        mask[sliceOK, :, :] = np.logical_or(
            mask[sliceOK, :, :],
            poly2mask(pts[:, 0], pts[:, 1],[rows, columns])
        )

    if sliceOK:
        # Transform binary mask to image
        mask = sitk.GetImageFromArray(mask)

        # Set voxel spacing
        mask.SetSpacing([x_resolution, y_resolution, slice_thickness])
    else:
        mask = None

    return mask
