# Data

## `scimagojr_2025.csv`

SCImago Journal Rank (SJR) scores, used by the `journal_impact` scoring
criterion to weight articles by the standing of the journal that published
them.

`agents/setup_bigquery.py` loads it into `<dataset>.journal_impact`, keeping the
rows with a usable SJR value (31,765 of 32,194).

**Format:** semicolon-delimited, comma decimal separators (`104,065` means
104.065). The loader handles both.

**Updating:** exports are downloaded from
[scimagojr.com/journalrank.php](https://www.scimagojr.com/journalrank.php) —
set the year, then use the CSV download link. The site is behind a bot check,
so this is a manual browser download. Point the loader at any year's export:

```bash
python agents/setup_bigquery.py --csv data/scimagojr_2026.csv --force-journals
```

Column layout has changed between years (2025 added `Open Access` and
`Open Access Diamond`). The loader selects columns by name and ignores the
rest, so newer exports load without changes.

**Citation**, per SCImago's terms for non-commercial use:

> SCImago, (n.d.). SJR — SCImago Journal & Country Rank [Portal].
> Retrieved 2026-07-27, from https://www.scimagojr.com
