# Design decisions

Short architecture decision records. Every rule below is enforced in code and covered by tests;
the code references them as `docs/decisions.md, Dn`.

The guiding principle: **a wrong published value is worse than a missing one.** Historical
election results are used as reference data by others; an error that looks plausible is never
noticed again. Every decision therefore trades recall for precision and auditability.

---

## D1 - No majority vote between OCR engines

**Decision.** A cell value is accepted only if at least two *independent* engine groups
(Tesseract, EasyOCR, PDF text layer) read exactly the same value under a strict number grammar
**and no group reads anything else**. Two against one is a disagreement and goes to review.

**Why.** Majority voting and "majority + passing arithmetic" were both evaluated and rejected:
two engines can share a failure mode (e.g. a broken `3` read as `8` by both), and a
transposition error can satisfy a sum by accident. Arithmetic is shown to the reviewer as a
*hint* (`review show`), never used to resolve a value.

**Where.** `ocr/resolver.py`, `tests/test_resolver.py`.

## D2 - No LLM / VLM in the extraction path

**Decision.** Values are read only by OCR engines whose output can be traced per glyph token;
ML is used for table detection/structure (as evidence, see D4) and OCR recognition only.
Generative recognisers (e.g. Surya >= 0.22, which is a VLM) were rejected.

**Why.** A generative model can produce a fluent, plausible but invented number with no
evidence trail. Every stored value must point to a token crop and the raw engine readings.

## D3 - Domain knowledge is data, never code

**Decision.** Parties, elections, column layouts, header aliases, section keywords and number
formats live in `reference/` (YAML, validated by pydantic). Document profiles are bound to one
source file by SHA-256. Units and parties are never created from OCR output; unknown names go
to review with fuzzy candidates.

**Why.** Every document differs; hard-coded switches do not generalise and hide assumptions.
A profile is reviewable, diffable and verifiable on its own (`verified: false` blocks
publication until a human has checked it).

**Trade-off.** A new document needs a new profile (200-480 lines of YAML for the volumes so
far). This is the main scaling cost of the approach; see README, "Limitations".

## D4 - Column structure needs two deterministic evidences; TATR is recorded only

**Decision.** A column partition is accepted only when two deterministic evidences agree:
(1) printed ruling lines or whitespace gutters in the body and (2) the printed column headers
or column numbers - and no glyph crosses a column boundary. The Table Transformer (TATR)
structure is stored as evidence (`ML_STRUCTURE_DISAGREEMENT`, INFO), never decisive.

**Why.** TATR disagreed with the visually verified structure on 36 % (Hessen 1953) and 18 %
(BW 1949) of the pages and fails on the rule-less line-printer tables of 1961-1969. A shifted
column silently assigns correct numbers to the wrong party - worse than any OCR error.

**Where.** `layout/column_strategies.py` (three strategies: `body_rules`, `header_rules`,
`column_numbers`), `layout/structure.py`.

## D5 - One unreadable header may be inferred from its neighbours

**Decision.** A column whose header text is unreadable counts as confirmed only if the column
count is exact, all its neighbours are confirmed directly, and nothing contradicts it (no
better alias of another column, no neighbour's column number read there). INFO issue
`HEADER_COLUMN_INFERRED`. Two adjacent unconfirmed columns always block.

**Why.** With an exact count and confirmed neighbours a single column cannot be shifted
without a contradiction; two adjacent ones could swap silently.

## D6 - An empty party cell is 0 only with an exact party-sum proof

**Decision.** In the 1965/1969 line-printer tables parties without votes are printed as blank
cells. A blank becomes 0 only if the profile allows it **and** the sum of all other (agreed)
party values equals the valid votes exactly; otherwise `MISSING_VALUE` (blocking).

**Why.** A blank can also be a lost glyph (faint print, scan defect). The exact sum is the
only evidence that nothing is missing.

## D7 - Printed shares and comparison lines are cross-checks, never published

**Decision.** Percent rows, difference and change rows, and comparison lines (results of the
previous election recomputed for new constituency boundaries) are read and used as
*blocking* cross-checks of the absolute values; they are never published.

**Why.** They are independent printed evidence for the absolute numbers. Publishing them would
duplicate derivable data and, for comparison lines, mix boundary definitions.

## D8 - Document differences are detected from the data (definition probes)

**Decision.** Whether a printed quantity follows the checked definition (e.g. "turnout =
voters / eligible", or shares rounded so that each row sums to exactly 100) is decided per run
and table profile from the rows whose values are all agreed: >= 90 % matching rows -> the
definition holds (deviations block); <= 50 % -> the publication uses another definition (check
replaced by an INFO note, value stored as printed); in between, or too few rows -> undecided,
the check stays blocking.

**Why.** Per-document switches were rejected in favour of auditable, data-driven decisions.
Evidence: BW 1949 prints a turnout that matches the computation in only 19 % of the rows
(postal voters counted elsewhere); 1957/1961 constituency tables adjust share rounding to 100.

**Where.** `validation/rules.py` (`DefinitionProbe`), `pipeline/document_service.py`.

## D9 - Structural row markers need one matching reading and no contradiction

**Decision.** Line markers (`1`/`2`/`V`, `E`/`Z`) are verified when at least one engine reads
the expected marker and no engine reads another vocabulary marker. Data values keep D1.

**Why.** The marker only confirms the line grouping the geometry already found; its
vocabulary is tiny and fixed, so a single matching reading without contradiction suffices.

## D10 - Nothing is published without verified reference data

**Decision.** Rows whose election, party, profile or municipality is not marked `verified`
are blocked (`REFERENCE_UNVERIFIED` / `REFERENCE_DATA_MISSING`). The publication gate
re-checks all conditions itself (defence in depth).

**Consequence.** Until the historical municipality gazetteer is supplied and the reference data
are verified by a person, the project publishes nothing. Machine-verified rows are available
as an explicitly *unpublished* provisional export (`export provisional`) and are measured in
the validation report.

## D11 - OCR fine-tuning: evaluated, not enabled by default

**Decision.** An EasyOCR recogniser fine-tuned on number tokens (labels: D1 consensus, or a
single dissent proven by the exact party sum) is available but not enabled; a fine-tuned
Tesseract model was rejected.

**Why.** EasyOCR: held-out errors 16 -> 3 on 20,000 cells, +1 to +7 percentage points agreed
cells, 0 changed agreed values on 42 held-out pages. But its labels come from the pipeline's own
consensus, so the held-out set under-represents exactly the hard cells; enabling it needs a
hand-labelled evaluation set first. Tesseract: four runs, all increased wrong readings
(68 -> 218..722; inserted digits, 5 -> 0/9/3).

## D12 - Parallel extraction, sequential interpretation

**Decision.** Rendering, image analysis and OCR of a page run in worker processes
(`spawn`, one set of models per worker, CPU threads split evenly); line grouping and the section
context run in the main process strictly in page order.

**Why.** Extraction depends only on the page; interpretation carries the county/section context
from page to page. `scripts/regression_snapshot.py` proves that a parallel run stores
byte-identical evidence to a sequential run on all eleven table layouts, and the complete 1961
volume (501,317 records) is identical with 1 and 9 workers.

**Measured.** 1961 volume, Apple M3 Pro: 12.8 s/page sequential, 6.1 s/page with 5 workers (2.1×),
7.2 s/page with 9 workers. The default is half the cores; beyond that the per-worker model memory
and the split torch threads cost more than they gain.
