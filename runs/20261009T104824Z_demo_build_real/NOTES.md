# Crashed — no results, do not cite

This folder holds a `config.json` and no `results.json` because the run
died partway: `sample_readings` caught `urllib.error.URLError` but not a
bare `ConnectionResetError`, which Google's edge threw on line 4 of 16.
Three lines had been read (15 calls) and were lost with it.

Kept rather than deleted so the day's quota arithmetic adds up: roughly
17 reading calls on `gemini-3.5-flash` were spent here and produced
nothing.

Fixed in the same commit: `OSError` is now caught and retried on another
key, and readings are cached per crop under
`data/demo_cache/readings/`, so a crash or a rerun no longer costs
quota. The replacement run is
`20261009T105809Z_demo_build_real` — see RESULTS.md row 13.
