"""Run from the downloaded GraphNN repository: python janus_cabbage_graph.py."""
if __name__ == "__main__":
    try:
        from graphnn.gui import main
    except ModuleNotFoundError as error:
        raise SystemExit(f"Missing dependency: {error.name}. Run: python -m pip install -r requirements.txt\nKeep this launcher inside the GraphNN folder.") from error
    main()
