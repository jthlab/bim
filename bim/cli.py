import click
from bim import tstats, sstats, __version__
from bim.BIM import main





@click.command()
@click.version_option(__version__)
@click.argument("tree_paths", nargs=-1, type=click.Path())
@click.argument("N", type=int)
@click.option(
    "--out",
    type=click.Path(writable=True),
    default="bim.csv",
    help="Out path",
    show_default=True
)
@click.option(
    "--stat",
    help="Statistics to compute. Multiple statistics can be selected using the --stat={stat1,stat2} syntax",
    type=click.Choice(['all'] + tstats + sstats , case_sensitive=False),
    multiple=True,
    default = ['all'],
    show_default = True
)
@click.option(
    "--wsz",
    type=int,
    help="Window size for windowed statistic."
)
@click.option(
    "--ssz",
    type=int,
    help="Stride size for windowed statistic."
)
@click.option(
    "--tsz",
    type=int,
    help="Tree-sequence size"
)
@click.option(
    "--pop",
    type=int,
    help="Population id, (see https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.samples)"

)
@click.option(
    "--eta",
    type=click.Path(exists=True),
    help='path for constant effective pop size function'
)
@click.option(
    "--r1t",
    type=float,
    default=0,
    help="l1 penalty for beta-tree",
    show_default=True
)
@click.option(
    "--r2t",
    type=float,
    default=0,
    help="l2 penalty for beta-tree",
    show_default=True
)
@click.option(
    "--r1s",
    type=float,
    default=0,
    help="l1 penalty for beta-sfs",
    show_default=True
)
@click.option(
    "--r2s",
    type=float,
    default=0,
    help="l1 penalty for beta-sfs",
    show_default=True
)
@click.option(
    "--log_pdf",
    type = click.Choice(['logfs', 'logfr'], case_sensitive=False), 
    help='log pdf of a splitting function',
    show_default=True, 
    default='logfs',
)
@click.option(
    "--weights",
    type=click.Choice(['branch','split'], case_sensitive=False),
    default='split',
    show_default=True
)
@click.option(
    "--abeta",
    type=float,
    default=0,
    show_default=True
)
def cli(**kwargs):
    """
    Software for β-Imbalance (BIM)
    Robust detection of natural selection using a probabilistic model of tree imbalance

    TREE_PATHS: paths to input trees. Multiple trees can be provided using the following syntax {first.trees,second.trees}
    
    N: sample size. If the sample size is less than the number of samples in the tree sequence file, BIM automatically subsamples
    """
    main(**kwargs)
