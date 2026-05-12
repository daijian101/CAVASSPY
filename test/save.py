import numpy as np

from cavass import CAVASS
from cavass_data import from_template
from ops import read_cavass_file

# MAC
# im0_file = '/Users/j/tmp/data/New_Abdomen_CT_CAVASS/Abd-CT001/Abd-CT001.IM0'
# bim_file = '/Users/j/tmp/data/New_Abdomen_CT_CAVASS/Abd-CT001/Abd-CT001_A-Ao.BIM'
#
# im0_data = CAVASS()
# im0_data.read(im0_file)
# im0_image = im0_data.get_data()
#
# bim_data = CAVASS()
# bim_data.read(bim_file)
# bim_image = bim_data.get_data()
#
# # Read saved data
# dest_im0_file = '/Users/j/tmp/A.IM0'
# dest_bim_file = '/Users/j/tmp/B.BIM'
#
# im0_image[1:20] = 0
# im0_data.set_data(im0_image)
# im0_data.save(dest_im0_file)
#
# bim_image[1:10, 100:200, 400:500] = True
# new_bim_obj = from_template(im0_data)
#
# new_bim_obj.set_data(bim_image)
# new_bim_obj.save(dest_bim_file)
#
# im0_obj_1 = CAVASS()
# im0_obj_1.read(dest_im0_file)
#
# bim_obj_1 = CAVASS()
# bim_obj_1.read(dest_bim_file)
#
# assert np.array_equal(im0_obj_1.get_data(), im0_image)
# assert np.array_equal(bim_obj_1.get_data(), bim_image)

#Ubuntu
im0_file = '/home/ubuntu/sda/BCA/core/IM0/SKN177PC1/CT.IM0'
bim_file = '/home/ubuntu/sda/BCA/core/IM0/SKN177PC1/SKN177PC1_OAM.BIM'

im0_data = CAVASS()
im0_data.read(im0_file)
im0_image = im0_data.get_data()

bim_data = CAVASS()
bim_data.read(bim_file)
bim_image = bim_data.get_data()

# Read saved data
dest_im0_file = '/home/ubuntu/sdb/dj/tmp/A.IM0'
dest_bim_file = '/home/ubuntu/sdb/dj/tmp/B.BIM'

im0_image[1:20] = 0
im0_data.set_data(im0_image)
im0_data.save(dest_im0_file)

bim_image[1:10, 100:200, 400:500] = True
new_bim_obj = from_template(im0_data)

new_bim_obj.set_data(bim_image)
new_bim_obj.save(dest_bim_file)

im0_obj_1 = CAVASS()
im0_obj_1.read(dest_im0_file)

bim_obj_1 = CAVASS()
bim_obj_1.read(dest_bim_file)

assert np.array_equal(im0_obj_1.get_data(), im0_image)
assert np.array_equal(bim_obj_1.get_data(), bim_image)