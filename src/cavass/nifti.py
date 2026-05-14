import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation, apply_orientation, inv_ornt_aff

from cavass._io import ensure_output_file_dir_existence


def save_nifti(output_file,
               data,
               voxel_spacing: float | list[float] | tuple[float, ...] | None = None,
               orientation='LPI'):
    """
    Save improc with nii format.

    Args:
        output_file (str):
        data (numpy.ndarray):
        voxel_spacing (sequence or None, optional, default=None): `tuple(x, y, z)`. Voxel spacing of each axis. If None,
            make `voxel_spacing` as `(1.0, 1.0, 1.0)`.
        orientation (str, optional, default='LPI'): 'LPI' | 'ARI'. LPI: Left-Posterior-Inferior;
            ARI: Anterior-Right-Inferior.

    Returns:

    """
    if voxel_spacing is None:
        voxel_spacing = (1.0, 1.0, 1.0)  # replace this with your desired voxel spacing in millimeters

    match orientation:
        case 'LPI':
            affine_matrix = np.diag(list(voxel_spacing) + [1.0])
        case 'ARI':
            # calculate the affine matrix based on the desired voxel spacing and ARI orientation
            affine_matrix = np.array([
                [0, -voxel_spacing[0], 0, 0],
                [-voxel_spacing[1], 0, 0, 0],
                [0, 0, voxel_spacing[2], 0],
                [0, 0, 0, 1]
            ])
        case _:
            raise ValueError(f'Unsupported orientation {orientation}.')

    # create a NIfTI improc object

    ensure_output_file_dir_existence(output_file)
    nii_img = nib.Nifti1Image(data, affine=affine_matrix)
    nib.save(nii_img, output_file)


def reorient(input_nifti_image, orientation: str):
    img = input_nifti_image.get_fdata()
    header = input_nifti_image.header

    affine = input_nifti_image.affine
    ornt = io_orientation(affine)
    target_ornt = axcodes2ornt((orientation[0], orientation[1], orientation[2]))
    transform = ornt_transform(ornt, target_ornt)
    img = apply_orientation(img, transform)

    affine = affine @ inv_ornt_aff(transform, img.shape)

    return nib.Nifti1Image(img, affine, header)