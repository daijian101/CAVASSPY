import numpy as np
from scipy import ndimage as ndi


def keep_largest_cc(mask: np.ndarray, connectivity: int | None = None):
    """
    Keep the largest connected component in a binary 2D or 3D mask.

    Args:
    mask (np.ndarray): ndarray (2D or 3D, bool or int). Binary input (0 = background, 1 = foreground).
    connectivity (int, optional, default=None): Connectivity for connected components.
    - For 2D: 1=4-connectivity, 2=8-connectivity (default=2).
    - For 3D: 1=6-connectivity, 2=18-connectivity, 3=26-connectivity (default=3).

    Returns:
        Binary mask with only the largest connected component kept.
    """
    mask = mask.astype(bool)

    if connectivity is None:
        connectivity = 2 if mask.ndim == 2 else 3

    structure = ndi.generate_binary_structure(mask.ndim, connectivity)
    labeled, num = ndi.label(mask, structure=structure)

    if num == 0:
        return np.zeros_like(mask)

    sizes = ndi.sum(mask, labeled, range(1, num + 1))
    largest_label = np.argmax(sizes) + 1
    cleaned = (labeled == largest_label)

    return cleaned
