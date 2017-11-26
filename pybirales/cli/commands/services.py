import click

from pybirales.app.app import configure_flask
from pybirales.birales import BiralesFacade, BiralesConfig


@click.group()
@click.pass_context
def services(ctx):
    pass


@services.command()
@click.argument('configuration', type=click.Path(exists=True))
@click.pass_context
def start_server(ctx, configuration):
    # Load the BIRALES configuration from file
    config = BiralesConfig(configuration, ctx.obj)

    # Initialise the Birales Facade (BOSS)
    bf = BiralesFacade(configuration=config)

    # Initialise Flask Application
    flask_app = configure_flask(configuration)

    # Start the Flask Application
    bf.start_server(flask_app)


@services.command()
def best_pointing():
    pass


@services.command()
def init_roach():
    pass
