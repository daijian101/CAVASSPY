import datetime
import os

import numpy as np
import pydicom.uid
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid


def get_dicom_modality(modality: str):
    match modality.lower():
        case 'ct':
            return 'CT'
        case 'mri' | 'mr':
            return 'MR'
        case 'pet' | 'pt':
            return 'PT'
        case _:
            raise ValueError(f'Unknown modality: {modality}.')


def read_dicom_series(input_dir, replace_padding_value=True, dtype: np.dtype | None = None):
    files = [dcmread(os.path.join(input_dir, f))
             for f in os.listdir(input_dir) if f.endswith('.dcm')]
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    img_shape = list(files[0].pixel_array.shape) + [len(files)]
    img = np.zeros(img_shape, dtype=np.float32)

    for i, ds in enumerate(files):
        if dtype is None:
            pixel_array = ds.pixel_array
        else:
            pixel_data = ds.PixelData
            pixel_array = np.frombuffer(pixel_data, dtype=dtype).reshape((ds.Rows, ds.Columns))

        # replace padding value
        if replace_padding_value and hasattr(ds, 'PixelPaddingValue'):
            padding_value = ds.PixelPaddingValue
            min_val = pixel_array[pixel_array != padding_value].min()
            pixel_array[pixel_array == padding_value] = min_val

        pixel_array = pixel_array.astype(np.float32)
        # apply rescale
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

        img[:, :, i] = pixel_array

    return img


def init_dicom_dataset(**kwargs) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.ImplementationVersionName = kwargs.get('ImplementationVersionName', 'PYDICOM')

    ds = FileDataset(kwargs.get('dsFileName', ''), {}, file_meta=file_meta, preamble=b'\0' * 128)
    ds.StudyInstanceUID = kwargs.get('StudyInstanceUID', '')
    ds.SeriesInstanceUID = kwargs.get('SeriesInstanceUID', '')
    ds.FrameOfReferenceUID = kwargs.get('FrameOfReferenceUID', '')
    ds.SeriesNumber = kwargs.get('SeriesNumber', '1')
    ds.InstanceNumber = kwargs.get('InstanceNumber', '')
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.is_implicit_VR = kwargs.get('is_implicit_VR', False)
    ds.is_little_endian = kwargs.get('is_little_endian', True)

    ds.Modality = get_dicom_modality(kwargs['Modality'])
    match ds.Modality:
        case 'CT':
            file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
            ds.SamplesPerPixel = kwargs.get('SamplesPerPixel', 1)
            ds.PhotometricInterpretation = kwargs.get('PhotometricInterpretation', 'MONOCHROME2')
            ds.BitsAllocated = kwargs.get('BitsAllocated', 16)
            ds.BitsStored = kwargs.get('BitsStored', 16)
            ds.HighBit = ds.BitsStored - 1
        case 'PT':
            file_meta.MediaStorageSOPClassUID = pydicom.uid.PositronEmissionTomographyImageStorage
            ds.SamplesPerPixel = kwargs.get('SamplesPerPixel', 1)
            ds.PhotometricInterpretation = kwargs.get('PhotometricInterpretation', 'MONOCHROME2')
            ds.BitsAllocated = kwargs.get('BitsAllocated', 16)
            ds.BitsStored = kwargs.get('BitsStored', 16)
            ds.HighBit = ds.BitsStored - 1
        case _:
            raise ValueError(f'Unsupported modality {ds.Modality}.')

    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    dt = datetime.datetime.now()
    date_str = dt.strftime('%Y%m%d')
    time_str = dt.strftime('%H%M%S.%f')  # long format with micro seconds

    ds.ContentDate = date_str
    ds.ContentTime = time_str
    ds.StudyDate = date_str
    ds.StudyTime = time_str
    ds.SeriesDate = date_str
    ds.SeriesTime = time_str
    ds.AcquisitionDate = date_str
    ds.AcquisitionTime = time_str
    ds.InstanceCreationDate = date_str
    ds.InstanceCreationTime = time_str
    ds.ContentDate = date_str
    ds.ContentTime = time_str
    ds.InstanceCreationDate = date_str
    ds.InstanceCreationTime = time_str

    ds.PatientName = kwargs.get('PatientName', 'Patient^Name')
    ds.PatientID = kwargs.get('PatientID', '12345')
    ds.PatientSex = kwargs.get('PatientSex', '')
    ds.PatientBirthDate = kwargs.get('PatientBirthDate', '')
    ds.PatientAge = kwargs.get('PatientAge', '')
    ds.PatientWeight = kwargs.get('PatientWeight', '')

    ds.PixelRepresentation = kwargs['PixelRepresentation']
    ds.ImageType = kwargs.get('ImageType', '')

    ds.StudyDescription = kwargs.get('StudyDescription', '')
    ds.ReferringPhysicianName = kwargs.get('ReferringPhysicianName', '')

    ds.Manufacturer = kwargs.get('Manufacturer', '')
    ds.InstitutionName = kwargs.get('InstitutionName', 'INSTITUTION_NAME_UNDEFINED')
    ds.ManufacturerModelName = kwargs.get('ManufacturerModelName', '')
    ds.SoftwareVersions = kwargs.get('SoftwareVersions', '')

    ds.LossyImageCompression = kwargs.get('LossyImageCompression', '00')

    ds.AcquisitionNumber = kwargs.get('AcquisitionNumber', 1)
    ds.AcquisitionDate = date_str
    ds.AcquisitionTime = time_str

    ds.ImagePositionPatient = kwargs.get('ImagePositionPatient', '')
    ds.ImageOrientationPatient = kwargs.get('ImageOrientationPatient', '')
    ds.PixelSpacing = kwargs.get('PixelSpacing', '')
    ds.SliceThickness = kwargs.get('SliceThickness', '')
    ds.SliceLocation = kwargs.get('SliceLocation', '')
    ds.SpacingBetweenSlices = kwargs.get('SpacingBetweenSlices', '')
    ds.PatientOrientation = kwargs.get('PatientOrientation', '')
    ds.RescaleSlope = kwargs.get('RescaleSlope', 1.0)
    ds.RescaleIntercept = kwargs.get('RescaleIntercept', 0.0)

    ds.Rows = kwargs.get('Rows', '')
    ds.Columns = kwargs.get('Columns', '')
    ds.SmallestImagePixelValue = kwargs.get('SmallestImagePixelValue', '')
    ds.LargestImagePixelValue = kwargs.get('LargestImagePixelValue', '')

    ds.PixelData = kwargs.get('PixelData', '')

    ds.Units = kwargs.get('Units', '')
    return ds
