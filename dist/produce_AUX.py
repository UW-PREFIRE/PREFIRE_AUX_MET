"""
Produce various auxiliary meteorology products from L1B input geometry/timing.

This program requires python version 3.6 or later, and is importable as a 
python module.
"""

  # From the Python standard library:
from pathlib import Path
import os
import sys
import argparse

  # From other external Python packages:

  # Custom utilities:


#--------------------------------------------------------------------------
def main():
    """Driver routine."""

    package_top_Path = Path(os.environ["PACKAGE_TOP_DIR"])

    sys.path.append(str(package_top_Path / "source"))
    from PREFIRE_AUX_MET.create_AUX_MET_product import create_AUX_MET_product

    ancillary_dir = os.environ["ANCILLARY_DATA_DIR"]
    input_L1B_fpath = os.environ["L1B_RAD_FILE"]
    output_dir = os.environ["OUTPUT_DIR"]
    tmpfiles_dir = os.environ["TMP_DIR"]
    met_an_dir = os.environ["MET_AN_DIR"]

    analysis_source = os.environ["ANALYSIS_SOURCE"]
    interp_method = os.environ["INTERP_METHOD"]

    atrack_idx_range_str = os.environ["ATRACK_IDX_RANGE_0BI"]
    tokens = atrack_idx_range_str.split(':')
    if tokens[2] == "END":
        atrack_np_idx_range = ("atrack", int(tokens[1]), None)  # Numpy indexing
    else:
        atrack_np_idx_range = ("atrack", int(tokens[1]),
                               int(tokens[2])+1)  # Numpy indexing

      # Default product_fullver:
    if "PRODUCT_FULLVER" not in os.environ:
        product_full_version = "R01_P00"
    elif len(os.environ["PRODUCT_FULLVER"].strip()) == 0:
        product_full_version = "R01_P00"
    else:
        product_full_version = os.environ["PRODUCT_FULLVER"]

    if os.environ["ANCSIM_MODE"].lower() == "none":
        ancsim_mode, ancsim_force_clearsky, ancsim_force_noseaice, \
                          ancsim_force_allseaice = (False, False, False, False)
    else:
        ancsim_mode = True
        tmp_str = os.environ["ANCSIM_MODE"].lower()
        ancsim_force_clearsky = ( "clearsky" in tmp_str )
        ancsim_force_noseaice = ( "noseaice" in tmp_str )
        ancsim_force_allseaice = ( "allseaice" in tmp_str )

    # Create the product data:
    create_AUX_MET_product(
        input_L1B_fpath, output_dir, tmpfiles_dir,
        analysis_source, interp_method,
        ancillary_dir, met_an_dir, product_full_version,
        ancsim_mode, ancsim_force_clearsky, ancsim_force_noseaice,
        ancsim_force_allseaice,
        atrack_range_to_process=atrack_np_idx_range
        )


if __name__ == "__main__":
    # Process arguments:
    arg_description = "Produce various auxiliary meteorology products from " \
                      "L1B input geometry/timing."
    arg_parser = argparse.ArgumentParser(description=arg_description)

    args = arg_parser.parse_args()

    # Run driver:
    main()
