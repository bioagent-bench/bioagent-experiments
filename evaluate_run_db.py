import duckdb
from pathlib import Path

df = duckdb.read_json([
    str(p) for p in Path('./run-logs').glob('**/run_metadata.json')
]).to_df()