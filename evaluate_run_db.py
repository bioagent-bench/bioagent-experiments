import duckdb
from pathlib import Path

df = duckdb.read_json([
    str(p) for p in Path('~/run_logs').expanduser().glob('**.json')
]).to_df()