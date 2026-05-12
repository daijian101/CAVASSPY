import SimpleITK as sitk


def overlay(image, label_map, colormap: list | tuple |None = None, opacity=0.5):
    image = sitk.GetImageFromArray(image)
    label_map = sitk.GetImageFromArray(label_map)
    if colormap:
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
