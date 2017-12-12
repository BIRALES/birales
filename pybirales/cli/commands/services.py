import click

from pybirales.birales import BiralesFacade, BiralesConfig


@click.group()
@click.pass_context
def services(ctx):
    pass


@services.command()
@click.argument('configuration', type=click.Path(exists=True), required=True)
@click.pass_context
def calibration(ctx, configuration):
    # Load the BIRALES configuration from file
    config = BiralesConfig(configuration, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    bf.calibrate()


@services.command()
@click.argument('configuration', type=click.Path(exists=True), required=True)
@click.pass_context
def run_server(ctx, configuration):
    # Load the BIRALES configuration from file
    config = BiralesConfig(configuration, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    bf.start_server()


@services.command()
def init_roach():
    pass


@services.command()
def best_pointing():
    pass
