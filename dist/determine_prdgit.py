"""
Determine product monikers and current (latest) git hashes that are part
of the provenance of this package's products ('AUX-MET'); write to file.

This program requires python version 3.6 or later, and is importable as a 
python module.
"""

  # From the Python standard library:
import os
import sys
import argparse
import subprocess
import importlib

  # From other external Python packages:

  # Custom utilities:


# *This* package must be first in the tuple of packages below:
Pypackages_to_query = ("PREFIRE_AUX_MET", "PREFIRE_sim_tools",
                       "PREFIRE_PRD_GEN", "PREFIRE_tools")


#--------------------------------------------------------------------------
def main(anchor_path):
    """Driver routine."""

    sys.path.append(os.path.join(anchor_path, "..", "source"))

    this_environ = os.environ.copy()

    product_moniker = "AUX-MET"

    # Build up string of product/algorithm monikers and git hash strings:
    git_cmd1 = ["git", "rev-parse", "--short=8", "--verify", "HEAD"]
    git_cmd2 = ["git", "diff", "--quiet"]
    beg_dir = os.getcwd()
    pg_pieces = [product_moniker, '(']
    initial_pass = True
    pkg = []
    for pkg_name in Pypackages_to_query:
        if not initial_pass:
            pg_pieces.append('+')

        # Import this package, save resulting object in list:
        pkg.append(importlib.import_module(pkg_name))

        # Read in algorithm moniker:
        with open(pkg[-1].filepaths.scipkg_version_fpath, 'r') as in_f:
            line = in_f.readline()
            pg_pieces.append(format(line.split()[0]))

        # Set output filepath (for *this* package):
        if initial_pass:
            try:
                output_fpath = pkg[-1].filepaths.scipkg_prdgitv_fpath
            except:
                output_fpath = pkg[-1].filepaths.scipkg_prdgitv_fpaths[0]
            initial_pass = False

        # Query git for latest commit hash:
        os.chdir(pkg[-1].filepaths.package_dir)
        try:
            cproc = subprocess.run(git_cmd1, stdout=subprocess.PIPE)
            commit_abbrev = cproc.stdout.decode().strip()
            cproc = subprocess.run(git_cmd2, stdout=subprocess.PIPE)
            if cproc.returncode != 0:
                modstr = "(modified)"
            else:
                modstr = ''
        except:  # git and/or .git/ not present/working
            commit_abbrev = "unknown"
            modstr = ''
        pg_pieces.append(commit_abbrev+modstr)
        os.chdir(beg_dir)

    # Assemble output string and write to a file:
    pg_pieces.append(')')
    with open(output_fpath, 'w') as out_f:
        text_to_write = ' '.join(pg_pieces) + '\n'
        out_f.write(text_to_write)


if __name__ == "__main__":
    # Determine fully-qualified filesystem location of this script:
    anchor_path = os.path.abspath(os.path.dirname(sys.argv[0]))

    # Process arguments:
    arg_description = ("Determine product monikers and current (latest) git "
                       "hashes that are part of the provenance of this "
                       "package's products ('AUX-MET'); write to file.")
    arg_parser = argparse.ArgumentParser(description=arg_description)

    args = arg_parser.parse_args()

    # Run driver:
    main(anchor_path)
