import netCDF4 as n
import xarray as xr
import glob
import os
import shutil

l1b_input_dir = '/data/datasim/S01_R00-clrsky/'
demtest_input_dir = '/home/mmm/projects/PREFIRE_geoloc_prototyping/test_output/'
output_dir = '/data/users/k/auxmet_sdps_test/inputs/'

demtest_fpaths = sorted(glob.glob(f'{demtest_input_dir}*.nc'))

addl_desc_str = ' **In this version, the file has been updated with '+\
    'realistic land fraction and surface elevation data from L1B DEM '+\
    'prototyping test files.**'

for demtest_fpath in demtest_fpaths:
    demtest_fname = os.path.basename(demtest_fpath)
    l1b_fname = \
        demtest_fname.split('DEMTEST_S00_R00')[0]+\
        '1B-RAD_S01_R00'+\
        demtest_fname.split('DEMTEST_S00_R00')[1]
    output_fname = \
        demtest_fname.split('DEMTEST_S00_R00')[0]+\
        '1B-RAD_S01_R00'+\
        demtest_fname.split('DEMTEST_S00_R00')[1]
        
    l1b_fpath = f'{l1b_input_dir}{l1b_fname}'
    output_fpath = f'{output_dir}{output_fname}'
    
    # Copy existing simulated L1B files in l1b_input_dir to output_dir
    shutil.copyfile(l1b_fpath, output_fpath)
        
    dem_ds = xr.open_dataset(demtest_fpath)
    # Modify variables in-place using 'r+' mode
    with n.Dataset(output_fpath, 'r+') as output_nc:
        # Get land fraction and elevation data from DEM test file
        output_nc['Geometry/land_fraction'][:] = dem_ds.lfrac.data
        output_nc['Geometry/elevation'][:] = dem_ds.altitude.data
        output_nc['Geometry/elevation_stdev'][:] = dem_ds.altitude_stdev.data
        
        output_nc.setncattr(
            'additional_file_description',
            output_nc.getncattr('additional_file_description')+addl_desc_str
            )
        output_nc.setncattr(
            'provenance',
            output_nc.getncattr('provenance')+' **and DEMTEST_S00_R00 granules**'
            )
        output_nc.setncattr(
            'input_product_files',
            output_nc.getncattr('input_product_files')+' **'+demtest_fname+'**'
            )
