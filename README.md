# PREFIRE_AUX_MET

Python package to produce the PREFIRE AUX-MET product (and the non-operations ANC-SimTruth product). This product provides meteorological fields from an analysis data source (e.g., GEOS-IT) interpolated in space and time to PREFIRE TIRS FOVs.

Currently, the only meteorological analysis data source supported for AUX-MET is the NASA Goddard Earth Observing System for Instrument Teams (GEOS-IT) analysis (https://gmao.gsfc.nasa.gov/GMAO_products/GEOS-IT/docs/). Both the simple-cylindrical gridded files and the cubed-sphere GEOS-IT files are supported. Support for other analysis sources such as ERA5 may be implemented in the future.

This code is released under the terms of this [LICENSE](LICENSE).  The version of this package can be found in [VERSION.txt](VERSION.txt).

# Installation

## Requirements

Python version 3.6+ is required, along with the following third-party Python
packages: netcdf4, scipy, esmpy, numpy, h5py, pyhdf, mpi4py, and cartopy.

The associated (Python-based) git repositories ['PREFIRE_PRD_GEN'](https://github.com/UW-PREFIRE/PREFIRE_PRD_GEN), ['PREFIRE_tools'](https://github.com/UW-PREFIRE/PREFIRE_tools), and ['PREFIRE_sim_tools'](https://github.com/UW-PREFIRE/) are also required for the proper operation of this package.

## Python Environment Setup

It is recommended to install the above Python packages in a dedicated conda environment (or something similar).  The packages used (and their versions) can be found in [conda_env.list](conda_env.list).

For example, using conda (and specifying Python 3.10.x from the conda-forge channel):

```
conda create --name for_PREFIRE_AUX_MET -c conda-forge python=3.10;
conda activate for_PREFIRE_AUX_MET;
conda install -c conda-forge netcdf4 scipy esmpy numpy h5py pyhdf mpi4py cartopy
;
```

The location of 'PREFIRE_PRD_GEN', 'PREFIRE_sim_tools', and 'PREFIRE_tools' depends on the value of the user's PYTHONPATH and/or sys.path -- for example, one could simply add each of those git repositories' local root Python source code directory to PYTHONPATH. Operationally, however, this package uses symbolic links to those git repositories' local root Python source code directories (or full copies of the same) in the source/ directory.

## Environment Variables

### Each job (executing this science algorithm package) is configured via information contained within environment variables.

### To specify that numpy, scipy, et cetera used by this algorithm should not use more than one thread or process, the below environment variables are expected to be set:

```
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

### Some environment variables are always required to be set (also see test/run_m*.sh or test/run_m*.ps1):

PACKAGE_TOP_DIR  :  the top-level directory (i.e., the one that contains dist/, test/, etc.) of this package

ANCILLARY_DATA_DIR  :  the package's ancillary data directory (should be an absolute path)

OUTPUT_DIR  :  the directory in which all meaningful output will be written (should be an absolute path)

TMP_DIR  :  the directory to which temporary, intermediate output files will be written (should be an absolute path)

ANALYSIS_SOURCE  :  source of meteorological analysis data (Valid values: GEOSIT_cubed_sphere, GEOSIT_equal_angle)

MET_AN_DIR  :  the root directory in which to find meteorological analysis data (should be an absolute path; e.g., "/data/GEOS-IT_5.29.14_cubedsphere/2021")

INTERP_METHOD  :  Interpolation library to use.  (Valid values: ESMF, internal)

ANCSIM_MODE  :  whether (and if so, which) ANC-SimTruth info should be written to the output file (valid values: NONE, ALLSKY, CLEARSKY, CLEARSKY_NOSEAICE, CLEARSKY_ALLSEAICE). If NONE (operational mode), the output file will be *_AUX-MET_*; otherwise, it will be *_ANC-SimTruth_*.

ATRACK_IDX_RANGE_0BI  :  coded frame (i.e., along-track segment) subset to process and output (for example, "ATRACK_IDXRANGE_0BASED_INCLUSIVE:2001:3100" => atrack dim indices, from 2001 through 3100; "ATRACK_IDXRANGE_0BASED_INCLUSIVE:0:END" => atrack dim indices, 0 through the last frame)

L1B_RAD_FILE  :  filepath of the "source" 1A-RAD product granule

### Some environment variables may not need to be set for operational use (instead, some have corresponding hard-coded default values that are "baked into" each operational algorithm delivery), but exist to enable efficient development and testing (also see test/run.sh).

PRODUCT_FULLVER  :  the full product processing/revision version string (e.g., "R01_P02").  Only increment 'Rxx' when the resulting products will be DAAC-ingested.

# Running the test script(s)

## Obtain and unpack ancillary and test data

None (for this version).

### Prepare the test input and output directories:

`cd test;`

On Linux/UNIX systems, possibly create a useful symbolic link to the test input data (if needed):

`ln -s WHEREEVER_THE_DATA_IS/inputs inputs;`

Prepare the output directory (Linux/UNIX example):

`mkdir -p outputs;`

_OR_ perhaps something like

`ln -s /data/users/myuser/data-PREFIRE_AUX_MET/outputs outputs;`

## Run the AUX_MET package

### A Linux/UNIX example

`cp run.sh my-run.sh;`

Edit `my-run.sh` as needed (e.g., change input file names)

`./my-run.sh`

The output file(s) will be in `test/outputs/`

## _The creation of this code was supported by NASA, as part of the PREFIRE (Polar Radiant Energy in the Far-InfraRed Experiment) CubeSat mission._
