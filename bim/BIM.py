import sys, os, json, re
import tskit

import numpy as np
import pandas as pd

from bim.utils import (
    Colless,
    Omega,
    Neutrality_Tests,
    intersect_with_weights,
    slidingSFS,
    tree_to_splits,
)
from bim import tstats, sstats


def check_stats(
    stat,
    supported_stats=tstats + sstats,
):
    """
    Check is user provided stats are supported

    Parameters
    ----------
    stat : list(str)
        user provided stats
    supported_stats : list(str)
        supported stats

    Returns
    -------
    list(str)
        valid user provided stats

    Raises
    ------
    ValueError
        if user provided stat is not supported

    """
    diff = set(stat).difference(set(supported_stats))
    if stat == ["all"]:
        return supported_stats
    elif len(diff) > 0:
        raise ValueError(f"{','.join(list(diff))} not in {','.join(supported_stats)}")
    else:
        return sorted(list(set(stat)))


class tree_neutrality:

    def __init__(
        self,
        N,
        Sstat,
        Tstat,
        wsz=None,
        tsz=None,
        ssz=None,
        eta_path=None,
        rho1_tree=0.0,
        rho2_tree=0.0,
        rho1_sfs=0.0,
        rho2_sfs=0.0,
        pop=None,
        log_pdf=None,
        weights="None",
        abeta=0.0,
    ):
        nt = Neutrality_Tests(N)

        # sfs based stats:
        Sstatf = {}
        for stat in Sstat:
            if stat == "bsfs":
                if eta_path is None:
                    eta = PConst(
                        a=[1.0], t=[0.0]
                    )  # id eta is not provided use constant popsize
                else:
                    if pop is None:
                        popid = (
                            0  # if pop is None, it will assume pop id in json file is 0
                        )
                    else:
                        popid = pop
                    with open(eta_path, "r") as fp:
                        eta = json.load(fp)[str(popid)]
                        eta = PConst(t=np.float32(eta["t"]), a=np.float32(eta["a"]))
                bsfs = bSFS(N, eta=eta, rho1=rho1_sfs, rho2=rho2_sfs)
                Sstatf[stat] = lambda sfs: float(bsfs.predict(sfs).x)
            elif stat == "TajD":
                Sstatf[stat] = lambda sfs: nt.TajD(sfs)
            elif stat == "FayH":
                Sstatf[stat] = lambda sfs: nt.FayH(sfs)
            elif stat == "ZngE":
                Sstatf[stat] = lambda sfs: nt.ZngE(sfs)
            elif stat == "FerL":
                Sstatf[stat] = lambda sfs: nt.FerL(sfs)
            elif stat == "FulD":
                Sstatf[stat] = lambda sfs: nt.FulD(sfs)

        def calcSstat(sfs):
            return {stat: Sstatf[stat](sfs) for stat in Sstat}

        # tree based stats
        Tstatf = {}
        for stat in Tstat:
            if stat == "btree":
                btree = bTree(
                    N, rho1=rho1_tree, rho2=rho2_tree, log_pdf=log_pdf, abeta=abeta
                )
                Tstatf[stat] = lambda n, k, w: float(
                    btree.predict(n, k, w).x
                )  # weighted likelihood
            if stat == "Colless":
                Tstatf[stat] = lambda n, k, w: Colless(n, k)
            if stat == "Omega":
                Tstatf[stat] = lambda n, k, w: Omega(n, k, top=4)

        def calcTstat(n, k, w):
            return {stat: Tstatf[stat](n, k, w) for stat in Tstat}

        self.calcSstat = calcSstat
        self.calcTstat = calcTstat
        self.Tstat = Tstat
        self.Sstat = Sstat
        self.wsz = wsz
        self.tsz = tsz
        self.ssz = ssz
        self.N = N
        self.pop = pop
        self.weights = weights
        self.abeta = abeta

    def predict(self, ts):

        calcSstat = self.calcSstat
        calcTstat = self.calcTstat
        wsz = self.wsz
        tsz = self.tsz
        ssz = self.ssz
        N = self.N
        pop = self.pop
        Tstat = self.Tstat
        Sstat = self.Sstat
        weights = self.weights

        if pop is not None:
            ts = ts.simplify(ts.samples(pop))

        Nt = ts.num_samples
        L = ts.get_sequence_length()

        if Nt < N:
            raise ValueError(
                "Number of samples in the tree(",
                Nt,
                ") is less than initilized sample size(",
                N,
                ")",
            )
        elif N < Nt:
            s1 = np.random.choice(ts.samples(), N, False)
            ts = ts.simplify(s1)
        else:
            pass

        # fit SFS based methods

        SFS, df, starts, ends = slidingSFS(ts, wsz=wsz, tsz=tsz, ssz=ssz)

        if len(Sstat) != 0:

            dfbsfs = []
            for i in range(SFS.shape[0]):
                sfs = SFS[i]
                SS = sfs.sum()

                start = df["start"].iloc[i]
                end = df["end"].iloc[i]
                ret = {"start": start, "end": end, "SS": SS}
                if SS == 0:
                    ret.update({stat: np.nan for stat in Sstat})
                else:
                    ret.update(calcSstat(sfs))
                dfbsfs.append(ret)
            dfbsfs = pd.DataFrame(dfbsfs)
            df = df.merge(dfbsfs, on=["start", "end"])
            del dfbsfs
            del SFS

        levels = np.arange(2, N + 1)
        if len(Tstat) != 0:

            dfbtree = []
            trees = ts.trees()
            ntrees = ts.num_trees
            for i in range(ntrees):
                Tree = next(trees)
                tst, ten = Tree.interval
                n, k = tree_to_splits(Tree, self.abeta)["splits"]

                if weights == "branch":
                    bls = np.array(
                        [
                            Tree.time(i)
                            for i in Tree.nodes(order="timedesc")
                            if Tree.is_internal(i)
                        ]
                        + [0]
                    )
                    bls = bls[:-1] - bls[1:]
                    bls = np.r_[bls, np.zeros(N - 1 - len(bls))]
                    ktk = np.flip(levels * bls)
                    w = ktk
                elif weights == "split":
                    w = n - 2
                else:
                    w = np.ones(N - 1)
                ret = {"start": tst, "end": ten}
                ret.update(calcTstat(n, k, w))
                dfbtree.append(ret)
            dfbtree = pd.DataFrame(dfbtree)

            # inersect both outs
            bs = dfbtree["start"].to_numpy()
            be = dfbtree["end"].to_numpy()
            y = dfbtree[Tstat].to_numpy()
            dfbtree = []
            for start, end in zip(starts, ends):
                dx = intersect_with_weights(bs, be, start, end, y)
                dx = pd.DataFrame(dx)
                for i in range(len(Tstat)):
                    stat = Tstat[i]
                    dx[stat] = dx["val"].apply(lambda x: x[i])
                dx = dx.drop(columns="val")
                dx = dx.groupby(["start", "end"])[Tstat].mean().reset_index()
                dfbtree.append(dx)
            dfbtree = pd.concat(dfbtree)
            df = df.merge(dfbtree, on=["start", "end"])
            del dfbtree

        df = df.sort_values(["start", "end"]).reset_index(drop=True)
        dflen = df.shape[0]
        df.insert(2, "N", N)

        return df.reset_index(drop=True)


def main(
    tree_paths,
    n,
    stat=['all'],
    wsz=None,
    ssz=None,
    tsz=None,
    pop=None,
    out="bim.csv",
    eta=None,
    r1t=0.0,
    r2t=0.0,
    r1s=0.0,
    r2s=0.0,
    log_pdf=None,
    weights="split",
    abeta=0.0,
):
    """
    BIM main CLI called function

    Parameters
    ----------

    tree_paths : tuple(str)
        Path to trees
    n : int
        sample size. If the sample size is less than the number of samples in the tree sequence file, BIM automatically subsamples
    stat : tuple(str)
        Statistics to compute
    wsz : int
        Window size for windowed statistic.
    ssz : int
        Stride size for windowed statistic.
    pop : int
        population_id, See tskit page
    out : str
        Output for csv file.
    eta : str
        Path for population size estimates. See the example.
    log_pdf : str
        log pdf of a splitting function.
    r1t : float
        l1 penalty for beta-Tree likelihood.
    r2t : float
        l2 penalty for beta-Tree likelihood.
    r1s : float
        l1 penalty for beta-SFS likelihood.
    r2s : float
        l2 penalty for beta-SFS likelihood.
    treew: str
        weighting method for beta-Tree likelihood. split, branch or None.

    Returns
    -------
        pd.DataFrame
    """

    stats = check_stats(stat)

    Tstat = list(set(tstats).intersection(stats))  # From tree splits
    Sstat = list(set(sstats).intersection(stats))  # From SFS

    if "bsfs" in Sstat:
        from bim.Bimbalance import bSFS, PConst

        global bSFS
        global PConst
    if "btree" in Tstat:
        from bim.Bimbalance import bTree, logfr, logfs

        global bTree
        if log_pdf == "logfs":
            log_pdf = logfs
        elif log_pdf == "logfr":
            log_pdf = logfr
        else:
            log_pdf = logfs

    tn = tree_neutrality(
        n,
        Sstat,
        Tstat,
        wsz=wsz,
        ssz=ssz,
        tsz=tsz,
        eta_path=eta,
        rho1_tree=r1t,
        rho2_tree=r2t,
        rho1_sfs=r1s,
        rho2_sfs=r2s,
        pop=pop,
        log_pdf=log_pdf,
        weights=weights,
        abeta=abeta,
    )

    df = []
    for path in tree_paths:
        ts = tskit.load(path)
        dfi = tn.predict(ts)
        _, treep = os.path.split(path)
        dfi["path"] = treep
        df.append(dfi)
    df = pd.concat(df).reset_index(drop=True).sort_values(["path", "start", "end"])

    f = open(out, "w+")
    f.write("#" + " ".join(sys.argv) + "\n")
    df.to_csv(f, index=False)
    f.close()
    return df


if __name__ == "__main__":
    main()
