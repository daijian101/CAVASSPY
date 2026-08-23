import SimpleITK as sitk

from cavass import colors


def overlay(image, label_map, colormap: str | list[str | tuple[int, int, int]] | tuple[str | tuple[int, int, int], ...] |None = None, opacity=0.5):
    image = sitk.GetImageFromArray(image)
    label_map = sitk.GetImageFromArray(label_map)
    if colormap is not None:
        if isinstance(colormap, str):
            colormap = [colormap,]
        for i in range(len(colormap)):
            color = colormap[i]
            if isinstance(color, str):
                color_rgb = getattr(colors, color)
                if color_rgb is None:
                    raise ValueError(f"Color {color} is unavailable.")
                colormap[i] = color_rgb

        if len(colormap) == 1:
            itk_color_map = list(colormap[0])
        else:
            itk_color_map = list(colormap[-1])
            for i in colormap[:-1]:
                itk_color_map += list(i)

        overlaid_image = sitk.LabelOverlay(image, label_map, opacity=opacity, colormap=itk_color_map)
    else:
        overlaid_image = sitk.LabelOverlay(image, label_map, opacity=opacity)
    overlaid_image = sitk.GetArrayFromImage(overlaid_image)
    return overlaid_image
