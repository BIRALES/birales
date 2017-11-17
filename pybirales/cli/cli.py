import click

from commands.pipelines import pipelines
from commands.services import services

cli = click.CommandCollection(sources=[pipelines, services])

if __name__ == '__main__':
    cli()
