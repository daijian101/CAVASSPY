import numpy as np

from cavass import CAVASS
from ops import read_cavass_file

# MAC
im0_file = '/Users/j/tmp/data/New_Abdomen_CT_CAVASS/Abd-CT001/Abd-CT001.IM0'
bim_file = '/Users/j/tmp/data/New_Abdomen_CT_CAVASS/Abd-CT001/Abd-CT001_A-Ao.BIM'

# Ubuntu
# im0_file = '/home/ubuntu/sda/BCA/core/IM0/SKN177PC1/CT.IM0'
# bim_file = '/home/ubuntu/sda/BCA/core/IM0/SKN177PC1/SKN177PC1_OAM.BIM'

im0_data = CAVASS()
im0_data.read(im0_file)
im0_image = im0_data.get_data()

bim_data = CAVASS()
bim_data.read(bim_file)
bim_image = bim_data.get_data()

# Compare with original data
im0_image = np.transpose(im0_image, (1, 2, 0))
bim_image = np.transpose(bim_image, (1, 2, 0))

raw_im0_image = read_cavass_file(im0_file)
raw_bim_image = read_cavass_file(bim_file)
assert np.array_equal(im0_image, raw_im0_image)
assert np.array_equal(bim_image, raw_bim_image)


# Windows
# im0_file = r'D:\BCA_data\IM0\SKN101PC1\CT.IM0'
# bim_file = r'D:\BCA_data\IM0\SKN101PC1\SKN101PC1_Dphm.BIM'
#
# im0_data = CAVASS()
# im0_data.read(im0_file)
# im0_image = im0_data.get_data()
#
# bim_data = CAVASS()
# bim_data.read(bim_file)
# bim_image = bim_data.get_data()

# Read header

# header = CAVASS()
# header.read(im0_file, header_only=True)
# pass