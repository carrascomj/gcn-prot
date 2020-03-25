"""Test fetch from PDB."""
import os

from os.path import join

from gcn_prot.data import fetch_PDB


def test_fetch(pdb_path):
    """Test fetch PDB file from RCSB by ATOM count comparison."""
    atoms_test = 0
    atoms_valid = 0

    fetch_PDB("1aa9.pdb", "1aa9")
    with open("1aa9.pdb") as f:
        for line in f:
            if "ATOM" in line:
                atoms_test += 1
    with open(join(pdb_path, "1aa9.pdb")) as f:
        for line in f:
            if "ATOM" in line:
                atoms_valid += 1

    assert atoms_test > 0
    assert atoms_test == atoms_valid
    os.remove("1aa9.pdb")
