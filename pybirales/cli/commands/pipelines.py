import click


@click.group()
@click.argument('configuration', type=click.Path(exists=True))
@click.option('--debug/--no-debug', default=False, help='Specify whether (or not) you\'d like to log debug messages.')
@click.pass_context
def pipelines():
    pass


@pipelines.command()
def standalone_test(configuration, debug):
    pass


@pipelines.command()
def test_receiver(configuration, debug):
    pass


@pipelines.command()
@click.option('--save-raw/--no-save-raw', default=False, help='Save raw data')
def birales_pipeline(configuration, debug, save_raw):
    pass


@pipelines.command()
@click.option('--save-raw/--no-save-raw', default=False, help='Save raw data?')
def correlator_pipeline(configuration, debug, save_raw):
    pass


@pipelines.command()
def save_raw_data(configuration, debug):
    pass


@pipelines.command()
def offline_birales_pipeline(configuration, debug):
    pass


@pipelines.command()
def offline_correlator(configuration, debug):
    pass


@pipelines.command()
def multi_pipeline(configuration, debug):
    pass


@pipelines.command()
@click.option('--save-raw/--no-save-raw', default=False, help='Save raw data')
@click.option('--save-beam/--no-save-beam', default=False, help='Save beam data')
def detection_pipeline(configuration, debug, save_raw, save_beam):
    pass