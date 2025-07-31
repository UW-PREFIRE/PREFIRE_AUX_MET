# This is an example driver script for generating batches of Anc-SimTruth base
# files. In practice, the year 2021 was divided into 8 time blocks and 8 driver
# scripts were run in parallel on longwave to generate the Anc-SimTruth base
# files for the 2021 full-year data simulation. This script produces files for
# the first of the 8 blocks.

# nohup usage:
#   nohup python AncSimTruth_base_datasim_full2021_driver[x].py > nohup[x].out&

import datetime as dt
import os
import glob

orbits_in_dir = '/data/users/mmm/datasim_orbits/1year_sim/2021/'
ancsim_out_dir = '/data/datasim/v01/2021/'
analysis_source = 'GEOSFPIT_equal_angle'
interp_method = 'ESMF'

# Divide into 8 temporal blocks during 2021 and run in parallel
# 365/8 = 45.625, so 5 blocks have 46 days and 3 blocks have 45 days
block_ndays = [46,46,46,46,46,45,45,45]
count_doy = 1
doys_end = []
for i, ndays in enumerate(block_ndays):
    doys_end.append(count_doy + block_ndays[i])
    count_doy += ndays

# Block 1
doys = range(1,doys_end[0],1)

# For subsequent blocks, modify script to start at the end doy of the previous block, 
# e.g. for block 2:
#   doys = range(doys_end[0],doys_end[1],1)

for doy in doys:
    doy_str = '{:03d}'.format(doy)
    doy_dir = ancsim_out_dir+doy_str+'/'
    if not os.path.exists(doy_dir):
        os.mkdir(doy_dir)
        
    orbit_fpaths = glob.glob(orbits_in_dir+doy_str+'/*1B-GEO*.nc')
    for orbit_fpath in orbit_fpaths:
        os.system('python -m Anc_SimTruth_base '+orbit_fpath+' '+doy_dir+' '+analysis_source+' '+interp_method)
        
        print('Finished Anc-SimTruth base file for '+orbit_fpath+' at '+
              dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
