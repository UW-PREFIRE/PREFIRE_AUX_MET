"""
Runs the AUX-MET algorithm in a way that divides a full granule up into a
number of chunks, producing a raw-*.nc output file for each (computed
serially).  This is chiefly intended to produce test data for the
PREFIRE_Product_Generator.

This program requires Python version 3.6 or later, and is importable as a
python module.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import netCDF4

# Example of usage:
#
#    python create_granule_parts.py GEOSIT_cubed_sphere ESMF NONE 4

def main(anchor_Path, args):
    """Main driver."""
    this_environ = os.environ.copy()

      # Specify that numpy, scipy, et cetera should not use more than one thread
      #  or process):
    this_environ["MKL_NUM_THREADS"] = '1'
    this_environ["NUMEXPR_NUM_THREADS"] = '1'
    this_environ["OMP_NUM_THREADS"] = '1'
    this_environ["VECLIB_MAXIMUM_THREADS"] = '1'
    this_environ["OPENBLAS_NUM_THREADS"] = '1'
  
    cwd_0 = Path(os.getcwd())  # Record the current working directory
    this_top_Path = Path(os.path.abspath(os.path.realpath(str(cwd_0 / ".."))))

    this_environ["PACKAGE_TOP_DIR"] = str(this_top_Path)
    this_environ["ANCILLARY_DATA_DIR"] = str(
                                          this_top_Path / "dist" / "ancillary")
    this_environ["OUTPUT_DIR"] = str(cwd_0 / "outputs")
    this_environ["TMP_DIR"] = str(cwd_0 / "tmpfiles")
    this_environ["PRODUCT_FULLVER"] = "R00_S06"
    this_environ["L1B_RAD_FILE"] = str(cwd_0 / "inputs" / \
#                       "PREFIRE_SAT1_1B-NLRAD_R00_S06_20211009122134_04256.nc")
                       "PREFIRE_SAT2_1B-NLRAD_R00_S06_20210211030714_00628.nc")

    # For this case, we assume inputs have been already copied, so bail if input
    # files/dirs do not exist.
    if not os.path.isdir(this_environ["ANCILLARY_DATA_DIR"]):
        print("ERROR: Ancillary input directory ("+ \
              this_environ["ANCILLARY_DATA_DIR"]+") does not exist.")
        sys.exit(1)
    if not os.path.isfile(this_environ["L1B_RAD_FILE"]):
        print("ERROR: Input file ("+ \
              this_environ["L1B_RAD_FILE"]+") does not exist.")
        sys.exit(1)

        # Check if output file directory exists; if not, bail:
    if not os.path.isdir(this_environ["OUTPUT_DIR"]):
        print("ERROR: Output directory ("+ \
              this_environ["OUTPUT_DIR"]+") does not exist.")
        sys.exit(1)

    this_environ["ANALYSIS_SOURCE"] = args.analysis_source
    this_environ["INTERP_METHOD"] = args.analysis_method

    if this_environ["ANALYSIS_SOURCE"] == "GEOSIT_cubed_sphere":
        this_environ["MET_AN_DIR"] = "/data/GEOS-IT_5.29.14_cubedsphere/2021"
    else:
        raise ValueError("ERROR: Invalid analysis source ("+ \
              this_environ["ANALYSIS_SOURCE"]+").")

    this_environ["ANCSIM_MODE"] = args.ancsim_mode

    # Determine a dimension value:
    ds = netCDF4.Dataset(this_environ["L1B_RAD_FILE"])
    n_atrack = ds.dimensions["atrack"].size
    ds.close()

    # Avoid any chunks consisting of a singular along-track frame, in order to
    #  evade issues due to NetCDF scalar <-> array(1) ambiguities:
    n_chunks = args.n_req_chunks
    nominal_stride = n_atrack//n_chunks
    if nominal_stride == 1:
        nominal_stride = 2
    inc_idx_rng = [(x, min(x+nominal_stride, n_atrack)-1) for x in \
                                            range(0, n_atrack, nominal_stride)]
    if inc_idx_rng[-1][1]-inc_idx_rng[-1][0] == 0:  # Singular along-track frame
        _ = inc_idx_rng.pop()  # Remove the singular chunk
        inc_idx_rng[-1] = (inc_idx_rng[-1][0],
                           inc_idx_rng[-1][1]+1)  # Enlarge last chunk by 1
    n_chunks = len(inc_idx_rng)

    os.chdir(str(anchor_Path / ".." / "dist"))
    for iir in inc_idx_rng:
        this_environ["ATRACK_IDX_RANGE_0BI"] = \
                       "ATRACK_IDXRANGE_0BASED_INCLUSIVE:"+f"{iir[0]}:{iir[1]}"
        print(this_environ["ATRACK_IDX_RANGE_0BI"])
               
        cmd = ["python", str(anchor_Path / ".." / "dist" / "produce_AUX.py")]
        subprocess.run(cmd, env=this_environ)


if __name__ == "__main__":
    # Determine fully-qualified filesystem location of this script:
    anchor_Path = Path(os.path.abspath(os.path.dirname(sys.argv[0])))

    # Process arguments:
    arg_description = "Run (potentially multiple) tests of the AUX_MET package."
    arg_parser = argparse.ArgumentParser(description=arg_description)
    arg_parser.add_argument("analysis_source")
    arg_parser.add_argument("analysis_method")
    arg_parser.add_argument("ancsim_mode")
    arg_parser.add_argument("n_req_chunks", type=int,
          help="Requested number of along-track processing chunks per granule.")

    args = arg_parser.parse_args()

    # Run driver:
    main(anchor_Path, args)
