# Download GEOS-IT data on cubed sphere grid for 2018 test period and sort into
# julian DOY directories.

# The following collections are downloaded for all available timesteps in 2018:
# - asm_inst_1hr_glo_C180x180x6_slv
# - asm_inst_3hr_glo_C180x180x6_v72
#       (test data only available for Jan-July 2018 as of 2022-03-04)
# - lnd_tavg_1hr_glo_C180x180x6_slv
# - ocn_tavg_1hr_glo_C180x180x6_slv

# Additionally, the single time-invariant asm_const_0hr_glo_C180x180x6_slv file
# is downloaded, since it is needed for surface elevation and land/water fraction
# calculations.

import os
import datetime as dt
from datetime import timedelta

parent_data_dir = '/data/users/k/GEOSIT_cubed_sphere_2018_testdata/'
os.chdir(parent_data_dir)

collections = [
               'asm_const_0hr_glo_C180x180x6_slv',
               'asm_inst_1hr_glo_C180x180x6_slv',
               'asm_inst_3hr_glo_C180x180x6_v72',
               'lnd_tavg_1hr_glo_C180x180x6_slv',
               'ocn_tavg_1hr_glo_C180x180x6_slv'
               ]

base_url = 'https://gmao.gsfc.nasa.gov/gmaoftp/ops/GEOSIT_sample/data_products/'

error_f = open('GEOSIT_cubed_sphere_2018_testdata_download_errors.txt','a')

for col in collections:
    if col == 'asm_const_0hr_glo_C180x180x6_slv':
        yr = 2018
        mth = 1
        dy = 1
        hr = 0
        mn = 0
        t = dt.datetime(yr, mth, dy, hr, mn)
        url = base_url+col+'/Y'+t.strftime('%Y')+'/M'+t.strftime('%m')+'/'+\
              'GEOS.it.asm.'+col+'.GEOS5271.'+t.strftime('%Y-%m-%dT%H%M')+\
              '.V01.nc4'
        try:
            os.system('wget '+url)
        except:
            print('** Error downloading file for col: '+col+', time: '+t.strftime('%Y-%m-%dT%H%M'))
            error_f.write('** Error downloading file for col: '+col+', time: '+t.strftime('%Y-%m-%dT%H%M'))
    
    else:
        if col == 'asm_inst_1hr_glo_C180x180x6_slv':
            t = dt.datetime(2018,1,1,0,0)
            end_dt = dt.datetime(2018,12,31,23,0)
            td_hrs = 1
        elif col == 'asm_inst_3hr_glo_C180x180x6_v72':
            t = dt.datetime(2018,1,1,0,0)
            end_dt = dt.datetime(2018,12,31,21,0)
            td_hrs = 3
        else:
            t = dt.datetime(2018,1,1,0,30)
            end_dt = dt.datetime(2018,12,31,23,30)
            td_hrs = 1
        
        while t <= end_dt:
            try:
                os.mkdir(parent_data_dir+t.strftime('%j')+'/')
            except FileExistsError:
                pass
            os.chdir(parent_data_dir+t.strftime('%j')+'/')
            
            url = base_url+col+'/Y'+t.strftime('%Y')+'/M'+t.strftime('%m')+'/'+\
                  'GEOS.it.asm.'+col+'.GEOS5271.'+t.strftime('%Y-%m-%dT%H%M')+\
                  '.V01.nc4'
            try:
                os.system('wget '+url)
            except:
                print('** Error downloading file for col: '+col+', time: '+t.strftime('%Y-%m-%dT%H%M'))
                error_f.write('** Error downloading file for col: '+col+', time: '+t.strftime('%Y-%m-%dT%H%M'))
            
            t += timedelta(hours=td_hrs)
            
error_f.close()
