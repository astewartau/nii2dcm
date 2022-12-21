"""
Create 3D Fetal Brain MRI SVRTK single-frame DICOM dataset from NIfTI file
"""
import os
import nibabel as nib

import nii2dcm.dcm_writer
import nii2dcm.nii
import nii2dcm.svr


# Load Nifti file
niiInPath = r'/Users/tr17/data/DicomRecon/previous-recon/SVR-output.nii.gz'
nii = nib.load(niiInPath)

# Set output directory
dcmOutPath = r'/Users/tr17/code/nii2dcm/output'
if os.path.exists(dcmOutPath):
    if not os.path.isdir(dcmOutPath):
        raise ValueError('The DICOM output path must be a directory.')
else:
    os.makedirs(dcmOutPath)

# Get NIfTI parameters to transfer to DICOM
nii2dcm_parameters = nii2dcm.nii.Nifti.get_nii2dcm_parameters(nii)


# Write single DICOM

# Initialise
TestDicomMRISVR = nii2dcm.svr.DicomMRISVR('testDicomMriSVR.dcm')

# Transfer Series tags
nii2dcm.dcm_writer.transfer_nii_hdr_series_tags(TestDicomMRISVR, nii2dcm_parameters)

# Get NIfTI pixel data
# TODO: create method in Nifti class – need to think about -1 value treatment
nii_img = nii.get_fdata()
nii_img[nii_img == -1] = 0  # set background pixels = 0 (-1 in SVRTK)
nii_img = nii_img.astype("uint16")  # match DICOM datatype

# Set custom tags
# TODO: lines below for overriding and testing DICOM creation.
#  Need to decide how best to implement properly, e.g.: force TransferSyntaxUID depending on Dicom subclass,
#  or give user option to edit. Think former, with possibility of latter.
TestDicomMRISVR.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
TestDicomMRISVR.ds.BitsAllocated = 16

# Write DICOM Series, instance-by-instance
for instance_index in range(0, nii2dcm_parameters['NumberOfInstances']):

    # Transfer Instance tags
    nii2dcm.dcm_writer.transfer_nii_hdr_instance_tags(TestDicomMRISVR, nii2dcm_parameters, instance_index)

    # Write slice
    nii2dcm.dcm_writer.write_slice(TestDicomMRISVR, nii_img, instance_index, dcmOutPath)

print(TestDicomMRISVR.ds)