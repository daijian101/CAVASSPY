import numpy as np


def windowing_transform(input_data, center, width, ymin=0, ymax=255, invert=False):
    """
    DICOM PS3.3 C.11.2.1.2:
    Pseudo-code:
        if (x <= c - 0.5 - (w-1) /2), then y = ymin
        else if (x > c - 0.5 + (w-1) /2), then y = ymax
        else y = ((x - (c - 0.5)) / (w-1) + 0.5) * (ymax- ymin) + ymin
    """
    width = max(float(width), 1.0)
    center = float(center)

    if width == 1.0:
        output = np.where(input_data <= center - 0.5, ymin, ymax)
    else:
        float_data = input_data.astype(np.float32)
        output = ((float_data - (center - 0.5)) / (width - 1.0) + 0.5) * (ymax - ymin) + ymin
        output = np.clip(output, ymin, ymax)

    output = output.astype(np.uint8)

    if invert:
        output = ymax - output

    return output


def cavass_soft_tissue_windowing(input_data):
    return windowing_transform(input_data, 1000, 500)


def cavass_bone_windowing(input_data):
    return windowing_transform(input_data, 2000, 4000)


def cavass_pet_windowing(input_data):
    return windowing_transform(input_data, 1200, 3500, invert=True)
