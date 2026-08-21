# External data

## cricsheet_player_styles.csv

Player batting/bowling style and playing-role metadata, keyed by the same
player-name convention Cricsheet ball-by-ball data uses (`unique_name`).

- **Source**: bundled CSV from https://github.com/mavaali/cricket-mcp
  (`data/player_meta.csv`), itself sourced from the `cricketdata` R package
  (https://github.com/ropenscilabs/cricketdata), an open-source package on
  CRAN maintained by Rob Hyndman et al.
- **Fetched**: 2026-08-20
- **Coverage** (against this project's `data/real_overs.csv` players):
  84.5% of batsmen have a real `batting_style`, 79.4% of bowlers have a
  real `bowling_style`. Uncovered players fall back to "unknown", same as
  before this data was added.
- **Used by**: `backend/app/ml/player_style_registry.py` normalizes the
  free-text style strings into a small set of cricket-meaningful buckets
  (arm + pace/spin type), consumed by both `src/features.py`-style
  training-data augmentation and `backend/app/ml/historical_feature_store.py`
  at serving time, so the two stay consistent.
