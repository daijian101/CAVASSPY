import os
import shutil

import nibabel as nib
import numpy as np

from pydicom.uid import generate_uid

from cavass._log import logger
from cavass.dicom import init_dicom_dataset
from cavass.nifti import reorient


def get_patient_orientation_from_nifti_affine(I_direction, J_direction):
    axes = []
    for direction in [I_direction, J_direction]:
        idx = np.argmax(np.abs(direction))
        if idx == 0:
            axis = 'R' if direction[idx] < 0 else 'L'
        elif idx == 1:
            axis = 'A' if direction[idx] < 0 else 'P'
        else:
            axis = 'F' if direction[idx] < 0 else 'H'
        axes.append(axis)
    return axis


def nifti2dicom(input_nifti_file, output_dicom_dir, modality: str, signed: int = 1, orientation: str | None = None,
                overwrite=False, **kwargs):
    if not os.path.isfile(input_nifti_file):
        raise FileNotFoundError(f'Input NIfTI file {input_nifti_file} not found.')
    nifti_image = nib.load(input_nifti_file)
    nifti_obj2dicom(input_nifti_image=nifti_image, output_dicom_dir=output_dicom_dir, modality=modality, signed=signed,
                    orientation=orientation, overwrite=overwrite, **kwargs)


def nifti_obj2dicom(input_nifti_image, output_dicom_dir, modality: str, signed: int = 1,
                    orientation:str ='PLS',overwrite=False, **kwargs):
    """
    Convert NIfTI image to DICOM image series.

    Args:
        input_nifti_image ():
        output_dicom_dir (str):
        modality (str):
        signed (int): 0: unsigned integer, 1: signed integer.
        orientation (str, optional, default=None): If provided, use the orientation for the output DICOM series.
        overwrite (bool, optional, default=False): if `Ture`, overwrite `output_dicom_dir` if it exists.`

    Returns:

    """
    if os.path.exists(output_dicom_dir):
        if overwrite:
            logger.info(f'Overwrite {output_dicom_dir} as it already exists.')
            shutil.rmtree(output_dicom_dir)
        else:
            raise ValueError(f'Output DICOM series dir {output_dicom_dir} already exists.')

    attrs = {}
    attrs['Modality'] = modality
    attrs['StudyInstanceUID'] = generate_uid()
    attrs['SeriesInstanceUID'] = generate_uid()
    attrs['FrameOfReferenceUID'] = generate_uid()
    attrs['ImageType'] = ['SECONDARY', 'DERIVED']

    data = reorient(input_nifti_image, orientation=orientation)
    header = data.header
    img = data.get_fdata()

    coordinate_system_transform = np.diag([-1, -1, 1, 1])
    dicom_affine = coordinate_system_transform @ data.affine
    
    i_vec = dicom_affine[:3, 1]
    j_vec = dicom_affine[:3, 0]
    k_vec = dicom_affine[:3, 2]

    i_normal = i_vec / np.linalg.norm(i_vec)
    j_normal = j_vec / np.linalg.norm(j_vec)
    k_normal = np.cross(i_normal, j_normal)
    k_normal = k_normal / np.linalg.norm(k_normal)

    attrs['PixelSpacing'] = [str(np.linalg.norm(i_vec)), str(np.linalg.norm(j_vec))]
    attrs['SliceThickness'] = np.abs(np.dot(k_vec, k_normal))
    attrs['SpacingBetweenSlices'] = attrs['SliceThickness']
    attrs['PatientOrientation'] = get_patient_orientation_from_nifti_affine(i_normal, j_normal)
    attrs['ImageOrientationPatient'] = [float(x) for x in np.concatenate([i_normal, j_normal])]

    rescale_slope = 1.0
    rescale_intercept = 0.0
    if 'scl_slope' in header and not np.isnan(header['scl_slope']):
        rescale_slope = header['scl_slope']
    if 'scl_inter' in header and not np.isnan(header['scl_inter']):
        rescale_intercept = header['scl_inter']

    if not signed:
        v_min = img.min()
        if v_min < 0:
            img = img - v_min
            rescale_intercept = rescale_intercept + v_min

    attrs['RescaleSlope'] = rescale_slope
    attrs['RescaleIntercept'] = rescale_intercept

    attrs['PixelRepresentation'] = signed
    if modality in ['CT']:
        if signed:
            dtype = np.int16
        else:
            dtype = np.uint16
        attrs['BitsAllocated'] = 16
        attrs['BitsStored'] = 16

    elif modality in ['PET']:
        if signed:
            dtype = np.float32
            attrs['BitsAllocated'] = 32
            attrs['BitsStored'] = 32
            attrs['PixelRepresentation'] = 0
        else:
            dtype = np.uint16
            if kwargs.get('rescale_PET_value', False):
                slope = kwargs.get('PET_value_rescale_slope', 100)
                img = img * slope
                attrs['RescaleSlope'] = 1 / slope
            attrs['BitsAllocated'] = 16
            attrs['BitsStored'] = 16
    else:
        raise ValueError(f'Modality {modality} is not supported.')

    os.makedirs(output_dicom_dir)
    attrs = kwargs | attrs
    for i in range(img.shape[2]):
        slice_data = img[..., i].astype(dtype)
        attrs['Rows'] = slice_data.shape[0]
        attrs['Columns'] = slice_data.shape[1]
        if dtype in [np.int16, np.uint16]:
            attrs['SmallestImagePixelValue'] = int(slice_data.min())
            attrs['LargestImagePixelValue'] = int(slice_data.max())

        origin = dicom_affine @ np.array([0, 0, i, 1])
        origin = origin[:-1]
        attrs['ImagePositionPatient'] = [float(x) for x in origin]
        attrs['SliceLocation'] = float(np.dot(k_normal, origin))
        attrs['InstanceNumber'] = i + 1
        attrs['PixelData'] = slice_data.tobytes()
        ds = init_dicom_dataset(**attrs)
        file_name = os.path.join(output_dicom_dir, f'slice_{i + 1:03d}.dcm')
        ds.save_as(file_name)
