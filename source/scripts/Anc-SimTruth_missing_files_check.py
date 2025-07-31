# During first attempt at generation of Anc-SimTruth base files for all of 2021,
# some GEOS FP-IT files were found to be missing, and the ESMF interpolation
# routine returned errors for another set of files. Use this script to make a
# list containing the file names of both types of errors, along with a list of
# Anc-SimTruth base files that are missing.

import numpy as np
import os
import glob
import datetime as dt
from datetime import timedelta

wdir = '/home/k/projects/PREFIRE_AUX_MET/Anc_SimTruth_base/'
geosfpit_dir = '/data/GEOS5_FP-IT/2021/'
orbits_dir = '/data/users/mmm/datasim_orbits/1year_sim/2021/'
ancsim_dir = '/data/datasim/v01/2021/'

base_dt = dt.datetime(2021,1,1,0,0)

# Determine which GEOS FP-IT files are missing
doys = np.arange(1,366,1)

hrs_inst = np.arange(0,24,3)
hrs_tavg = np.arange(0,24,1)

col_name_2d = 'inst3_2d_asm_Nx'
col_name_3d = 'inst3_3d_asm_Nv'
col_name_ocn = 'tavg1_2d_ocn_Nx'
col_name_lnd = 'tavg1_2d_lnd_Nx'

missing_geosfpit_files = []

for doy in doys:
    base_dt_doy = base_dt + timedelta(days=(int(doy)-1))
    
    existing_fpaths_2d = glob.glob(geosfpit_dir+'{:03d}'.format(doy)+'/*'+col_name_2d+'*.nc4')
    existing_fnames_2d = [os.path.basename(x) for x in existing_fpaths_2d]
    fnames_2d = []
    for hr in hrs_inst:
        t = dt.datetime(base_dt_doy.year, base_dt_doy.month, base_dt_doy.day, hr, 0)
        fnames_2d.append('GEOS.fpit.asm.'+col_name_2d+'.GEOS5124.'+t.strftime('%Y%m%d_%H%M')+'.V01.nc4')
    for fname in fnames_2d:
        if fname not in existing_fnames_2d:
            missing_geosfpit_files.append(fname)
        
    existing_fpaths_3d = glob.glob(geosfpit_dir+'{:03d}'.format(doy)+'/*'+col_name_3d+'*.nc4')
    existing_fnames_3d = [os.path.basename(x) for x in existing_fpaths_3d]
    fnames_3d = []
    for hr in hrs_inst:
        t = dt.datetime(base_dt_doy.year, base_dt_doy.month, base_dt_doy.day, hr, 0)
        fnames_3d.append('GEOS.fpit.asm.'+col_name_3d+'.GEOS5124.'+t.strftime('%Y%m%d_%H%M')+'.V01.nc4')
    for fname in fnames_3d:
        if fname not in existing_fnames_3d:
            missing_geosfpit_files.append(fname)

    existing_fpaths_ocn = glob.glob(geosfpit_dir+'{:03d}'.format(doy)+'/*'+col_name_ocn+'*.nc4')
    existing_fnames_ocn = [os.path.basename(x) for x in existing_fpaths_ocn]
    fnames_ocn = []
    for hr in hrs_tavg:
        t = dt.datetime(base_dt_doy.year, base_dt_doy.month, base_dt_doy.day, hr, 30)
        fnames_ocn.append('GEOS.fpit.asm.'+col_name_ocn+'.GEOS5124.'+t.strftime('%Y%m%d_%H%M')+'.V01.nc4')
    for fname in fnames_ocn:
        if fname not in existing_fnames_ocn:
            missing_geosfpit_files.append(fname)
        
    existing_fpaths_lnd = glob.glob(geosfpit_dir+'{:03d}'.format(doy)+'/*'+col_name_lnd+'*.nc4')
    existing_fnames_lnd = [os.path.basename(x) for x in existing_fpaths_lnd]
    fnames_lnd = []
    for hr in hrs_tavg:
        t = dt.datetime(base_dt_doy.year, base_dt_doy.month, base_dt_doy.day, hr, 30)
        fnames_lnd.append('GEOS.fpit.asm.'+col_name_lnd+'.GEOS5124.'+t.strftime('%Y%m%d_%H%M')+'.V01.nc4')
    for fname in fnames_lnd:
        if fname not in existing_fnames_lnd:
            missing_geosfpit_files.append(fname)


# # Parse file names that caused ESMF errors
# ESMF_error_files = []
# with open(wdir+'PET0.ESMF_LogFile') as infile:
#     log_info = infile.readlines()
# for l in log_info:
#     if 'GEOS.fpit.asm' in l:
#         ESMF_error_files.append(l.split('netCDF Error: ')[1].split(':')[0])


# # Save list of all files that are missing or that caused ESMF error
# missing_geosfpit_files.extend(set(ESMF_error_files))
# missing_geosfpit_files.sort()
# with open(wdir+'missing_GEOSFPIT_files.txt', 'w') as outfile:
#     outfile.write('\n'.join(missing_geosfpit_files))
    
    
# Determine which simulated orbits do not have a corresponding Anc-SimTruth base file.
missing_ancsim_files = []
for doy in doys:
    base_dt_doy = base_dt + timedelta(days=(int(doy)-1))
    orbit_fpaths = glob.glob(orbits_dir+'{:03d}'.format(doy)+'/*.nc')
    orbit_fnames = [os.path.basename(x) for x in orbit_fpaths]
    
    ancsim_fpaths = glob.glob(ancsim_dir+'{:03d}'.format(doy)+'/*.nc')
    ancsim_fnames_existing = [os.path.basename(x) for x in ancsim_fpaths]
    for fname in orbit_fnames:
        ancsim_fname_expected = fname.split('1B-GEO')[0]+'Anc-SimTruth'+fname.split('1B-GEO')[1]
        if ancsim_fname_expected not in ancsim_fnames_existing:
            missing_ancsim_files.append(ancsim_fname_expected)

# with open(wdir+'missing_Anc-SimTruth_files_1.txt', 'w') as outfile:
#     outfile.write('\n'.join(missing_ancsim_files))
