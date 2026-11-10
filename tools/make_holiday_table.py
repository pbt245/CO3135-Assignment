"""Build data/external/holidays_es.csv from the `holidays` package (run once, locally).

Why a pre-built CSV: the `holidays` package is not guaranteed on Kaggle (offline
sessions), and a committed table makes the holiday features reproducible regardless
of the package version.

Columns
- hol_nat:   1 if a national holiday in Spain.
- hol_share: share of the 17 autonomous communities that have a public holiday that
             day (national days -> 1.0; e.g. Holy Thursday -> ~0.9; a single-region day
             -> ~0.06). A cheap proxy for regional holidays; population weights would be
             better but need external data.
- names:     holiday names, for eyeballing.
"""
from pathlib import Path

import holidays
import pandas as pd

YEARS = range(2014, 2020)  # one year of margin each side for lag features
# The 17 autonomous communities (Ceuta, Melilla and the city-level subdivision excluded).
REGIONS = ["AN", "AR", "AS", "CB", "CL", "CM", "CN", "CT", "EX", "GA",
           "IB", "MC", "MD", "NC", "PV", "RI", "VC"]


def main() -> None:
    nat = holidays.Spain(years=YEARS)
    reg = {r: holidays.Spain(subdiv=r, years=YEARS) for r in REGIONS}
    days = pd.date_range(f"{min(YEARS)}-01-01", f"{max(YEARS)}-12-31", freq="D")
    rows = []
    for d in days:
        dd = d.date()
        n_reg = sum(dd in reg[r] for r in REGIONS)
        if dd in nat or n_reg:
            names = set()
            if dd in nat:
                names.add(nat.get(dd))
            for r in REGIONS:
                if dd in reg[r]:
                    names.add(reg[r].get(dd))
            rows.append({"date": dd, "hol_nat": int(dd in nat),
                         "hol_share": round(n_reg / len(REGIONS), 4),
                         "names": " | ".join(sorted(names))})
    out = Path(__file__).resolve().parents[1] / "data" / "external" / "holidays_es.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out} (holidays {holidays.__version__})")


if __name__ == "__main__":
    main()
