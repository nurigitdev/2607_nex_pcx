from app.main import create_app


def test_create_app_metadata() -> None:
    app = create_app()

    assert app.title == "NeX_PCX"
    assert app.version == "0.1.0"
