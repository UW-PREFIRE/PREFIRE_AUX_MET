#!/usr/bin/env bash

# ** For this test script, set all paths as relative to the directory:
#  /home/k/projects/PREFIRE_AUX_MET/source/scripts/

#set -ve;  # Exit when the first non-zero exit status is encountered, and
           #  print out commands as we execute them
set -e;  # Exit when the first non-zero exit status is encountered

# Make sure to use an absolute path to the script as a basedir so we can call
# the script from any location:
readonly basedir=$(dirname $(realpath $0));

# Set input/output directories to be relative to the current working dir:
# - For tests on longwave, input, output and tmpfile directories are symbolic links
#   to directories in /data (to avoid placing large files in /home)

# Inputs
ANCILLARY_DATA_DIR=${PWD}/../../dist/ancillary;
#L1B_RAD_FILE=${PWD}/inputs/PREFIRE_SAT2_1B-GEOM_B02_R00_20210721000649_03049.nc;
#L1B_RAD_FILE=/data/users/ttt/data-reconstitute-old_simL1B/outputs/for_KM/PREFIRE_SAT2_1B-GEOM_S03_R00_20210101062206_00005.nc;

# Output and tmpfiles
OUTPUT_DIR=${PWD}/../../test/outputs;
TMP_DIR=${PWD}/../../test/tmpfiles;

# For this case, we assume inputs have been already copied, so bail if input
# files/dirs do not exist.
test -d "${ANCILLARY_DATA_DIR}" || { echo "Ancillary input directory does not exist: ${ANCILLARY_DATA_DIR}"; exit 1; }
#test -f "${L1B_RAD_FILE}" || { echo "Input file does not exist: ${L1B_RAD_FILE}"; exit 1; }

# Check if output and temporary file directories exist; if not, bail:
test -d "${OUTPUT_DIR}" || { echo "Output directory does not exist: ${OUTPUT_DIR}"; exit 1; }
test -d "${TMP_DIR}" || { echo "Temporary files directory does not exist: ${TMP_DIR}"; exit 1; }

# Set some necessary parameters:

#ANALYSIS_SOURCE="GEOSIT_equal_angle";
#INTERP_METHOD="internal";
#MET_AN_DIR="/data/GEOS-IT_test/2018_test";
#MET_AN_DIR="/data/users/k/GEOSIT_cubed_sphere_2018_testdata";

ANALYSIS_SOURCE="GEOSIT_cubed_sphere";
INTERP_METHOD="ESMF";
MET_AN_DIR="/data/GEOS-IT_5.29.14_cubedsphere/2021";
# ANCSIM_MODE can be "Y" or "N"
ANCSIM_MODE="Y";

ATRACK_IDX_RANGE_0BI="ATRACK_IDXRANGE_0BASED_INCLUSIVE:0:END";
#ATRACK_IDX_RANGE_0BI="ATRACK_IDXRANGE_0BASED_INCLUSIVE:0:8139";
#ATRACK_IDX_RANGE_0BI="ATRACK_IDXRANGE_0BASED_INCLUSIVE:0:1000";
#ATRACK_IDX_RANGE_0BI="ATRACK_IDXRANGE_0BASED_INCLUSIVE:3332:5433";

PRODUCT_FULLVER="B02_R00";

# First file for 2021-01-01 can't be processed because we don't have GEOS-IT
# data for 2020-12-31
#   - File is PREFIRE_SAT2_1B-GEOM_B02_R00_20210101002016_00006.nc 
for L1B_RAD_FILE in $(ls "${PWD}/../../test/inputs"/* | sed 1d);

do
# Make required environment vars available:
export ANCILLARY_DATA_DIR L1B_RAD_FILE OUTPUT_DIR TMP_DIR ANALYSIS_SOURCE;
export INTERP_METHOD MET_AN_DIR ANCSIM_MODE ATRACK_IDX_RANGE_0BI PRODUCT_FULLVER;

test -f "${L1B_RAD_FILE}" || { echo "Input file does not exist: ${L1B_RAD_FILE}"; exit 1; }
# Execute script that writes a new 'prdgit_version.txt', which contains product
#  moniker(s) and current (latest) git hash(es) that are part of the provenance
#  of this package's product(s).
# *** This step should not be done within the SDPS system, since that file is
#     created just before delivery to the SDPS system.
hn=`hostname -s`;
if [ "x$hn" = "xlongwave" ]; then
   python "${basedir}"/../../dist/determine_prdgit.py;
fi

# Run the process/calculation:

cd "${basedir}/../../dist";

python "${basedir}"/../../dist/produce_AUX.py;

echo "TEST completed successfully";

done
