# NFL Parlay Helper (Dual Probabilities, 2025)

Two probability sections:
1) **Historical** — based on the player's last **N** games (regularized mean & std).  
2) **Context‑Adjusted** — modifies the estimate using *injuries, weather, pace proxy, usage trend, opponent defensive injuries,* and optionally blends toward the **market's implied Over probability** (de‑vig).

## Run (local or Streamlit Cloud)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Data at runtime
- **Stats, injuries, depth charts, schedules:** via `nflreadpy` & nflverse (no keys).  
- **Weather:** Open‑Meteo hourly forecast by stadium coordinates.  
- **Market odds:** enter American odds for Over & Under to incorporate vig.

## Notes
- Pace is a **rough proxy** based on team appearance counts per week; you can replace with official pace/seconds-per-play data later.
- Usage trend boosts/dampens the mean if the last 3 games outperform/underperform the prior 3.
- Defensive injuries: counts OUT/IR/Doubt for DB/LB/DL/EDGE positions to adjust generosity.
- All multipliers are mild and clamped to keep estimates conservative.
- This is **not betting advice**; purely informational.
