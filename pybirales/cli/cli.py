import warnings

import click

from pybirales.cli.commands.pipelines import pipelines
from pybirales.cli.commands.scheduler import scheduler
from pybirales.cli.commands.services import services

warnings.filterwarnings("ignore")


@click.group()
@click.pass_context
def cli(ctx):
    return ctx


cli.add_command(scheduler)
cli.add_command(pipelines)
cli.add_command(services)

if __name__ == '__main__':
    cli(obj={})
