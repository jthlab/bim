import pytest
from bim import tstats, sstats, bTree, bSFS, Colless, Neutrality_Tests
from bim.BIM import check_stats, main
import pandas as pd
import os
import tskit

test_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(test_dir, "data")

def test_check_stats():
    with pytest.raises(ValueError):
        check_stats(['not_a_valid_stat'])
    
    assert isinstance(check_stats(["all"]), list)
    assert sorted(check_stats(["all"])) == sorted(list(set(tstats + sstats)))
    assert check_stats(["Colless"]) == ["Colless"]
    assert sorted(check_stats(["Colless", "FayH"])) == sorted(["FayH", "Colless"])

def test_main():
    bim_df = pd.read_csv(os.path.join(data_dir, "bim.csv"), comment="#")
    bim_df = bim_df[sorted(bim_df.columns)]
    tree_paths = os.path.join(data_dir, "test.trees")
    print(tree_paths)

    df_res = main(
        tree_paths = [os.path.join(data_dir, "test.trees")],
        n = 30,
        stat=["btree","Colless"],
        tsz=1
    )
    df_res = df_res[sorted(df_res.columns)]

    assert isinstance(df_res, pd.DataFrame)
    pd.testing.assert_frame_equal(bim_df, df_res)

def test_newick():
    N = 30
    nwck = '((12,(9,((23,(10,11)),(((1,((6,(13,(17,18))),((25,(20,21)),(29,30)))),(16,26)),(15,19))))),(5,((24,(3,22)),((14,28),((2,4),(27,(7,8)))))));'
    btree = bTree(N = N)
    n, k = btree.newick_to_splits(nwck)['splits'] # Get the bifurcating tree split sizes from nwck representation
    bt = btree.predict(n, k, w = n-2) # This will return the optimization result
    cl = Colless(n, k) # This will return the Colless statistic for the same tree
    assert pytest.approx(bt.x[0], 0.001) == -8.062
    assert pytest.approx(cl, 0.001) == 0.256

def test_tskit():
    ts = tskit.load(os.path.join(data_dir, "test.trees"))
    N = ts.num_samples
    btree = bTree(N = N) # This will initialize the optimizer for a bifurcating tree with 30 nodes
    Tree = ts.first() # Get the first tree from the tree-sequence
    n, k = btree.tree_to_splits(Tree)['splits'] # Get the bifurcating tree split sizes from tskit.Tree (see the figure)
    bt = btree.predict(n, k, w = n-2) # This will return the optimization result
    cl = Colless(n, k) # This will return the Colless statistic for the same tree
    assert pytest.approx(bt.x[0], 0.001) == -8.062
    assert pytest.approx(cl, 0.001) == 0.256

def test_sfs():
    ts = tskit.load(os.path.join(data_dir, "test.trees"))
    N = ts.num_samples
    bsfs = bSFS(N = N)
    sfs = ts.allele_frequency_spectrum(polarised=True, span_normalise=False)[1:-1] # Calculate SFS
    bs = bsfs.predict(sfs)
    nt = Neutrality_Tests(N)

    assert pytest.approx(bs.x[0], 0.001) == 0.976
    assert pytest.approx(nt.TajD(sfs), 0.01) == 0.370
    assert pytest.approx(nt.FulD(sfs), 0.01) == -0.213
    assert pytest.approx(nt.FayH(sfs), 0.01) == 0.043
    assert pytest.approx(nt.ZngE(sfs), 0.01) == 0.081
    assert pytest.approx(nt.FerL(sfs), 0.01) == 0.130