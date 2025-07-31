#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Download 3-hourly CO2 and CH4 data from CAMS EGG4 reanalysis for full record
# (2003-2020)

# Must be run in an environment with a Climate Data Store (CDS) api key installed:
#   https://confluence.ecmwf.int/display/CKB/CAMS%3A+Reanalysis+data+documentation#CAMS:Reanalysisdatadocumentation-Dataorganisationandaccess
#   https://ads.atmosphere.copernicus.eu/api-how-to
#   https://confluence.ecmwf.int/pages/viewpage.action?pageId=177480803

# This script is a template. Downloading the actual data required breaking requests
# into a separate script for each year and took ~3 weeks for the download to
# complete, due to the slow speed of retrieval from Copernicus ADS MARS archive.
# - Scripts were run using nohup on longwave and output was stored in a file. E.g.: 
#   nohup ~/miniconda3/envs/cdsenv/bin/python CAMS_EGG4_CO2_CH4_dl_[year].py > nohup[year].out&

import cdsapi
import datetime as dt
import calendar

c = cdsapi.Client()

yrs = range(2003,2021,1)
mths = range(1,13,1)

data_dir = '/data/users/k/CAMS_EGG4_CO2_CH4/'

for yr in yrs:
    for mth in mths:
        end_day = calendar.monthrange(yr,mth)[1]
        beg_dt = dt.datetime(yr,mth,1)
        end_dt = dt.datetime(yr,mth,end_day)
        c.retrieve(
                   'cams-global-ghg-reanalysis-egg4',
                       {
                        'date': beg_dt.strftime('%Y-%m-%d')+'/'+end_dt.strftime('%Y-%m-%d'),
                        'format': 'netcdf',
                        'variable': [
                                     'co2_column_mean_molar_fraction',
                                     'ch4_column_mean_molar_fraction'
                                     ],
                        'step': ['0','3','6','9','12','15','18','21'],
                        'use': 'infrequent'
                        },
                   data_dir+beg_dt.strftime('%Y%m')+'.nc'
                   )
