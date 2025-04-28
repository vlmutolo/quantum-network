run: create-result-dir
    cargo run --release
    uv run python/plot.py

create-result-dir:
    @mkdir -p "{{justfile_directory()}}/results"
