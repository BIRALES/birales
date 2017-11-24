import click

from commands.pipelines import pipelines
from commands.services import services


@click.group()
@click.pass_context
def cli(ctx):
    return ctx

cli.add_command(pipelines)
cli.add_command(services)


if __name__ == '__main__':
    cli(obj={})
