import click

from commands.pipelines import pipelines
from commands.services import services


def update_config(config, section, key, value):
    if config:
        if section in config:
            config[section][key] = value
            return config
        config[section] = {}
        return update_config(config, section, key, value)
    return update_config({}, section, key, value)


@click.group()
@click.pass_context
def cli(ctx):
    return ctx

cli.add_command(pipelines)
cli.add_command(services)


if __name__ == '__main__':
    cli(obj={})
