import argparse
from pathlib import Path
from .create_AUX_MET_product import create_AUX_MET_product
from .paths import package_dir

arg_description = "Creates the PREFIRE AUX-MET product(s)."
arg_parser = argparse.ArgumentParser(description=arg_description)
arg_parser.add_argument("TIRS_L1B_fpath",
                        help="TIRS level-1B granule filepath.")
arg_parser.add_argument("AUX_MET_output_dir",
                        help="directory for AUX-MET output files.")
arg_parser.add_argument("met_analysis_source",
                        help="one of the following: 'GEOSIT_equal_angle', "
                             "'GEOSIT_cubed_sphere'.")
arg_parser.add_argument("interp_method",
                        help="one of the following: 'internal' {only with "
                             "'GEOSIT_equal_angle'}, 'ESMF'.")
arg_parser.add_argument("-t", "--tmp-dir", metavar="tmp_dir",
                        default=str(Path(package_dir) / "test" / "tmpfiles"),
                        help="directory in which temporary files are located "
                             "(default: %(default)s.")
arg_parser.add_argument("-a", "--ancillary-dir", metavar="ancillary_dir",
                        default=str(Path(package_dir) / "dist" / "ancillary"),
                        help="directory in which ancillary files are located "
                             "(default: %(default)s.")
arg_parser.add_argument("-m", "--met-analysis-dir", metavar="met_analysis_dir",
                        default=None,
                        help="directory in which the meteorological analysis "
                             "dataset is located.")
arg_parser.add_argument("-i", "--atrack-idx-range",
                        metavar="atrack_idx_range", default="0:END",
                        help="along-track index range (zero-based, inclusive) "
                             "to process within granule. Valid examples: "
                             "'0:8140', '456:1200', '0:END' "
                             "(default: %(default)s.")
arg_parser.add_argument("--year", metavar="YYYY", default=None,
                        help="For testing purposes, replace the actual year of "
                             "the level-1B granule with YYYY.")
args = arg_parser.parse_args()

if args.met_analysis_dir is None:
    if args.met_analysis_source == "GEOSIT_equal_angle":
        args.met_analysis_dir = "/data/GEOS-IT_test/2018_test"
    elif args.met_analysis_source == "GEOSIT_cubed_sphere":
        args.met_analysis_dir = \
                     "/data/users/k/GEOSIT_cubed_sphere_2018_testdata"

tokens = args.atrack_idx_range.split(':')
if tokens[1] == "END":
    atrack_idx_range_np = ("atrack", int(tokens[0]), None)  # Numpy indexing
else:
    atrack_idx_range_np = ("atrack", int(tokens[0]), int(tokens[1])+1)  # Numpy indexing

create_AUX_MET_product(args.TIRS_L1B_fpath, args.AUX_MET_output_dir,
                       args.tmp_dir,
                       args.met_analysis_source, args.interp_method,
                       args.ancillary_dir, args.met_analysis_dir,
                       atrack_range_to_process=atrack_idx_range_np,
                       substitute_year=args.year)
