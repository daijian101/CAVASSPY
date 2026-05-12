import os
import shutil
from pathlib import Path
from uuid import uuid1

import nibabel as nib
import numpy as np

from cavass._io import ensure_output_file_dir_existence
from cavass.medimgconvert.dicom_cavass import dicom2cavass
from cavass.medimgconvert.dicom_nifti import nifti_obj2dicom
from cavass.nifti import save_nifti, reorient
from cavass.ops import get_voxel_spacing, read_cavass_file, save_cavass_file
from cavass.utils import one_hot


def nifti_obj2cavass(input_nifti_image, output_file, modality, offset=0, pose_reference_file=None, **kwargs):
    """
    Convert NIfTI image to cavass image.

    Args:
        input_nifti_image ():
        output_file (str):
        modality (Modality):
        offset (int, optional, default=0):
        pose_reference_file (str, optional, default=None):
    """

    if pose_reference_file is not None:
        if not os.path.isfile(pose_reference_file):
            raise FileNotFoundError(f'Copy pose file {pose_reference_file} not found.')

    made_output_dir, output_dir = ensure_output_file_dir_existence(output_file)

    tmp_dicom_dir = os.path.join(output_dir, f'{uuid1()}')
    try:
        r1 = nifti_obj2dicom(input_nifti_image, tmp_dicom_dir, modality=modality, orientation='PLS', overwrite=True,
                             **kwargs)
        r2 = dicom2cavass(tmp_dicom_dir, output_file, offset, pose_reference_file)
    except Exception as e:
        if made_output_dir and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        if os.path.isdir(tmp_dicom_dir):
            shutil.rmtree(tmp_dicom_dir)
        if os.path.exists(output_file):
            os.remove(output_file)

        raise e
    shutil.rmtree(tmp_dicom_dir)
    return r1, r2


def nifti2cavass(input_nifti_file, output_file, modality, offset=0, pose_reference_file=None, **kwargs):
    """
    Convert NIfTI image to cavass image.

    Args:
        input_nifti_file (str):
        output_file (str):
        modality (Modality):
        offset (int, optional, default=0):
        pose_reference_file (str, optional, default=None):
    """

    if not os.path.isfile(input_nifti_file):
        raise FileNotFoundError(f'Input NIfTI file {input_nifti_file} not found.')
    nifti_image = nib.load(input_nifti_file)
    return nifti_obj2cavass(nifti_image, output_file, modality, offset, pose_reference_file, **kwargs)


def nifti_label2cavass(input_nifti_file:str| Path, output_dir, objects, study_name: str|None = None,
                       modality='CT', discard_background=True, pose_reference_file=None, with_dicom=False):
    """
    Convert NIfTI format segmentation file to cavass BIM format file. A NIfTI file in where contains arbitrary categories
    of objects will convert to multiple CAVASS BIM files, which matches to the number of object categories.

    Args:
        input_nifti_file (str or Path):
        output_dir (str): For each object, the output directory is `output+dir`/object
        objects (sequence or str): objects is an array or a string with comma splitter of object categories,
        study_name (str, optional, default=None): The filename of the generated BMI file: `study_name`_`obj`.BMI, if specified.
        where the index of the category in the array is the number that indicates the category in the segmentation.
        modality (Modality, optional, default=CT):
        discard_background (bool, optional, default True): if True, the regions with label of 0 in the segmentation
        (typically refer to the background region) will not be saved.
        pose_reference_file (str, optional, default=None): Pose reference file.
        with_dicom (bool, optional, default=False): If True, convert NIfTI to DICOM to BIM. If False, convert NIfTI to BIM directly.

    Returns:

    """
    input_nifti_file = Path(input_nifti_file)
    if not input_nifti_file.exists():
        raise FileNotFoundError(f'Input NIfTI file {input_nifti_file} not found.')

    if pose_reference_file is not None:
        if not os.path.isfile(pose_reference_file):
            raise FileNotFoundError(f'Copy pose file {pose_reference_file} not found.')

    if study_name is None:
        file_name = input_nifti_file.name
        if file_name.endswith('.nii.gz'):
            study_name = file_name.replace('.nii.gz', '')
        elif file_name.endswith('.nii'):
            study_name = file_name.replace('.nii', '')
        else:
            study_name = file_name

    input_data = nib.load(input_nifti_file)
    image_data = input_data.get_fdata()

    if isinstance(objects, str):
        objects = objects.split(',')
    n_classes = len(objects) + 1 if discard_background else len(objects)
    one_hot_arr = one_hot(image_data, num_classes=n_classes)

    start = 1 if discard_background else 0
    for i in range(start, one_hot_arr.shape[3]):
        nifti_label_image = nib.Nifti1Image(one_hot_arr[..., i], input_data.affine, input_data.header, dtype=np.uint8)
        if discard_background:
            obj = objects[i - 1]
        else:
            obj = objects[i]

        output_file = os.path.join(output_dir, obj, f'{study_name}_{obj}.BIM')
        if with_dicom:
            nifti_obj2cavass(nifti_label_image, output_file, modality, pose_reference_file=pose_reference_file)
        else:
            nifti_label_image = reorient(nifti_label_image, orientation='PLS')
            data = nifti_label_image.get_fdata().astype(np.uint8)
            save_cavass_file(output_file, data, binary=True, pose_reference_file=pose_reference_file)


def cavass2nifti(input_file, output_file, orientation='ARI'):
    """
    Convert cavass IM0 and BIM formats to NIfTI.

    Args:
        input_file (str):
        output_file (str):
        orientation (str, optional, default='ARI'): image orientation of NIfTI file, 'ARI' or 'LPI'

    Returns:

    """

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f'Input file {input_file} not found.')

    spacing = get_voxel_spacing(input_file)
    data = read_cavass_file(input_file)
    save_nifti(output_file, data, spacing, orientation=orientation)
