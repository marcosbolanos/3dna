# Repo tools

- use `uv` for adding packages and running commands
- Dev dependencies are added with `uv add --dev`
- Project is installed as editable for testing
- Typecheck with `uv run pyright`
- Lint with `uv run ruff`

# Conventions and rules
- never use dynamic imports. Every library used must be explicitly in the project's dependencies. Dev dependencies are separate.
