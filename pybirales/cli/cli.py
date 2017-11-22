import click

from commands.pipelines import pipelines
from commands.services import services


@click.group()
def cli():
    pass

cli.add_command(pipelines)
cli.add_command(services)


if __name__ == '__main__':
    cli()
