import typer

app = typer.Typer(help="Awesome CLI user manager.")

# Positional / required arguments
    parser.add_argument(
        "database",
        help="database path",
    )
    parser.add_argument(
        "output",
        help="output directory name",
    )

    # Optional arguments
    parser.add_argument(
        "--pulsemap", "-p",
        help="pulsemap to be extracted",
        default='SRTTWOfflinePulsesDC',
    )

@app.command()
def database(database: str):
    """
    Define a database path.
    """
    typer.echo(f"database: {database}")


@app.command()
def output(
    output: str,
    force: bool = typer.Option(
        ...,
        prompt="Are you sure you want to delete the user?",
        help="Force deletion without confirmation.",
    ),
):
    """
    Delete a user with USERNAME.

    If --force is not used, will ask for confirmation.
    """
    if force:
        typer.echo(f"Deleting user: {username}")
    else:
        typer.echo("Operation cancelled")


@app.command()
def delete_all(
    force: bool = typer.Option(
        ...,
        prompt="Are you sure you want to delete ALL users?",
        help="Force deletion without confirmation.",
    )
):
    """
    Delete ALL users in the database.

    If --force is not used, will ask for confirmation.
    """
    if force:
        typer.echo("Deleting all users")
    else:
        typer.echo("Operation cancelled")


@app.command()
def init():
    """
    Initialize the users database.
    """
    typer.echo("Initializing user database")


if __name__ == "__main__":
    app()