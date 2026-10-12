# Gazetteer (historical municipality register)

This register is supplied by the project owner. Without an entry here **no** municipality can
be resolved automatically, and therefore **no** result can be published
(issue codes `ENTITY_NOT_FOUND` / `REFERENCE_DATA_MISSING`).

Format: one or more UTF-8 CSV files (`*.csv`) in this folder, comma-separated, with a header
line. Required columns: `unit_id,name,level`. All columns:

| Column | Required | Meaning |
|---|---|---|
| unit_id | yes | stable, unique id (e.g. official key + validity period) |
| name | yes | official name at the time of validity, spelled exactly as in the official register |
| level | yes | e.g. `GEMEINDE`, `LANDKREIS`, `STADTKREIS`, `REGIERUNGSBEZIRK`, `LAND` (must match `section_kinds` / `unit_level_default` of the profiles) |
| parent_unit_id | no | parent unit (municipality -> county -> region -> state); must itself be listed |
| official_key | no | official municipality key (AGS / Gemeindekennziffer) |
| valid_from / valid_to | no | ISO date `YYYY-MM-DD`; empty = unbounded |
| aliases | no | historical or alternative spellings, `;`-separated (real variants only, never OCR errors) |
| verified | no | `true` once the entry has been checked; unverified entries block publication |
| source | no | source, e.g. "Amtliches Gemeindeverzeichnis 1950" |

Example (see `gazetteer_template.csv.example`):

```csv
unit_id,name,level,parent_unit_id,official_key,valid_from,valid_to,aliases,verified,source
he_lk_alsfeld_1953,Alsfeld,LANDKREIS,he_land,,,,,true,Gemeindeverzeichnis Hessen 1950
he_g_alsfeld_1953,"Alsfeld, Stadt",GEMEINDE,he_lk_alsfeld_1953,,,,Alsfeld,true,Gemeindeverzeichnis Hessen 1950
```

Matching rules (see `semantics/entities.py`): automatic only on an exact (normalised) match of
name or alias, valid on election day, inside the recognised county context, and unique. Fuzzy
matches are only shown as candidates for review - a new unit is **never** created.
