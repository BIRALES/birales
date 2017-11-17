import click


@click.group()
@click.pass_context
def services():
    pass

@services.command()
def calibrate():
    pass


@services.command()
def best_pointing():
    pass


@services.command()
def init_roach():
    pass
