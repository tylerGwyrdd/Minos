# Documentation

Use these commands from the repository root to build and serve the MkDocs site.

## Generate docs (build static site)

```bash
poetry run mkdocs build
```

Generated output is written to `site/`.

## Serve docs locally (with live reload)

```bash
poetry run mkdocs serve
```

Then open `http://127.0.0.1:8000/`.

## Optional: clean build + serve

```bash
poetry run mkdocs build --clean
poetry run mkdocs serve
```
