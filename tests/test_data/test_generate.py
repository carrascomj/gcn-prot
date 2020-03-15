"""Test parser of PDB to generate a graph."""

from os.path import join

from gcn_prot.data import parse_pdb


def test_parse_pdb(pdb_path, graph_path):
    """Test correct parsing of PDB file into features."""
    prot_example = parse_pdb(join(pdb_path, "1agp.pdb"), chain="A")
    with open(join(graph_path, "1agp_a.txt")) as f:
        n_lines = 0
        for _ in f:
            n_lines += 1
    assert len(prot_example) == n_lines
    # number of features per line
    assert len(prot_example[0]) == 10
