import os
import shutil

from cavass._io import ensure_output_file_dir_existence
from cavass.ops import execute_cavass_cmd, copy_pose, get_image_resolution


def dicom2cavass(input_dicom_dir, output_file, offset=0, pose_reference_file=None):
    """
    Note that if the output file path is too long, this command may be failed.

    Args:
        input_dicom_dir (str):
        output_file (str):
        offset (int, optional, default=0):
        pose_reference_file (str, optional, default=None): if `pose_reference_file` is given, copy pose of this
        file to the output file.

    """

    if not os.path.exists(input_dicom_dir):
        raise ValueError(f'Input DICOM series {input_dicom_dir} not found.')
    if pose_reference_file is not None:
        if not os.path.isfile(pose_reference_file):
            raise FileNotFoundError(f'Copy pose file {pose_reference_file} not found.')

    tmp_files = []
    made_output_dir, output_dir = ensure_output_file_dir_existence(output_file)

    # input_dicom_dir = input_dicom_dir.replace(' ', '\\ ')
    # output_file = output_file.replace(' ', '\\ ')
    try:
        if pose_reference_file is None:
            r = execute_cavass_cmd(f'from_dicom {input_dicom_dir}/* {output_file} +{offset}')
        else:
            split = os.path.splitext(output_file)
            root = split[0]
            extension = split[1]
            output_tmp_file = root + '_TMP' + extension
            r = execute_cavass_cmd(f'from_dicom {input_dicom_dir}/* {output_tmp_file} +{offset}')
            copy_pose(output_tmp_file, pose_reference_file, output_file)
            tmp_files.append(output_tmp_file)

    except Exception as e:
        if made_output_dir and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        for each in tmp_files:
            if os.path.exists(each):
                os.remove(each)

        if os.path.exists(output_file):
            os.remove(output_file)
        raise e

    for each in tmp_files:
        if os.path.exists(each):
            os.remove(each)
    return r


def cavass2dicom(input_file, output_dicom_file, start_slice: int | None = None,
                 end_slice: int | None = None):
    """
    Convert CAVASS file (IM0 and BIM files) to DICOM series.
    Args:
        input_file (str):
        output_dicom_file (str): Output DICOM series filename without extension.
        start_slice (int, optional, default=None): The start slice number for conversion. If `None` (default), conversion starts from the first slice.
        end_slice (int, optional, default=None): The end slice number for conversion. If `None` (default), conversion continues to the final slice.

    Returns:

    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f'Input file {input_file} not found.')

    if start_slice is None:
        start_slice = 0
    else:
        start_slice = start_slice - 1

    total_slice_number = get_image_resolution(input_file)[2]
    if end_slice is None:
        end_slice = total_slice_number - 1
    else:
        end_slice = end_slice - 1

    if start_slice < 0 or start_slice >= total_slice_number:
        raise ValueError(f'Start slice {start_slice} is out of bounds: [0, {total_slice_number - 1}].')

    if end_slice < 0 or end_slice >= total_slice_number:
        raise ValueError(f'End slice {end_slice} is out of bounds: [0, {total_slice_number - 1}].')

    if start_slice > end_slice:
        raise ValueError(f'Start slice {start_slice} must be less than end slice {end_slice}.')

    made_output_dir, output_dir = ensure_output_file_dir_existence(output_dicom_file)

    try:
        execute_cavass_cmd(f'mipg2dicom \'{input_file}\' \'{output_dicom_file}\' 0 {start_slice} {end_slice}')
    except Exception as e:
        if made_output_dir and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        raise e
