import typer

from pta import *

app = typer.Typer()

@app.command()
def build(kgw: str = 'ipoint', np: int = 20, n_ph: int = 10, n_th: int = 10, nt: int = 10, nf: int = 10, nb: int = 1):
    gw = GravitationalWave(key=kgw)
    pa = PulsarArray(np)
    grid = SkyMap(n_ph,n_th,nt,nf,nb)
    grid.generate_redshift(pa, gw)
    grid.plot_HD_curve(pa, key='obs', show=False)


@app.callback(invoke_without_command=True)
def context(ctx: typer.Context):
    """
    CLI running
    """
    if ctx.invoked_subcommand is None:
        print("Running a CLI...")