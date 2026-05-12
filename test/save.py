import numpy as np

from cavass import CAVASS
from cavass_data import from_template
from ops import read_cavass_file

im0_file = '/Users/j/tmp/data/New_Abdomen_CT_CAVASS/Abd-CT001/Abd-CT001.IM0'
bim_file = '/Users/j/tmp/data/New_Abdomen_CT_CAVASS/Abd-CT001/Abd-CT001_A-Ao.BIM'

im0_data = CAVASS()
im0_data.read(im0_file)
im0_image = im0_data.get_data()

bim_data = CAVASS()
bim_data.read(bim_file)
bim_image = bim_data.get_data()

# Read saved data
dest_im0_file = '/Users/j/tmp/A.IM0'
dest_bim_file = '/Users/j/tmp/B.BIM'

new_bim_data = np.zeros_like(im0_image).astype(bool)
new_bim_data[1:10, 100:200, 400:500] = True
new_bim_obj = from_template(im0_data)

new_bim_obj.set_data(new_bim_data)
new_bim_obj.save(dest_bim_file)

im0_data.save(dest_im0_file)
