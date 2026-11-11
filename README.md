# Analysis pipeline

This repository contains the analysis pipeline for a two-week diary study
(N=9) on how listeners notice, interpret and respond to problematic
recommender-system behaviours (PBs) on music streaming platforms. It
supports a manuscript submitted to the *International Journal of
Human-Computer Interaction* (IJHCI).

Three PB categories, plus an uncertain option, structure the analysis.

| Code | Construct |
|------|-----------|
| SB   | Societal Bias |
| CI   | Commercial Imperatives |
| LD   | Limited Discovery |
| O    | Other / unclassified |

## What the pipeline does

Three scripts run in sequence.

1. **`src/analyze_events.py`** is the event-level analysis. It loads the
   diary events, maps the selected PB statements to the SB, CI, LD and O
   items, reports scale reliability (Cronbach's α with Feldt intervals),
   computes the conditional probabilities P(PB | context) with
   Clopper-Pearson intervals, aggregates them to category level, and
   correlates the PB items with the three Likert appraisal items
   (ALGO_CAUSE, FUTURE_USE, ARTIST_HARM). It writes the tables and figures
   and an enriched event-level CSV for the next script.
2. **`src/analyze_participants.py`** is the participant-level analysis. It
   reads the enriched CSV and writes one row per participant (PB rates and
   counts, context shares, Likert means).
3. **`src/analyze_supplementary.py`** computes the platform shares, the
   events per day, the weekly values of Table A4 and the leave-one-out
   ranges of Appendix B.3. It prints a comparison of each value with the
   number quoted in the paper.

## Data

The diary data is included in `data/events_20260324.csv`. The scripts
read it from that location, relative to the repository root. If the file
is missing, they stop with a message that names the expected path.

The diary data file and the participant ID mapping (`PID_MAP` in
`src/config.py`) are included in this repository by documented decision.

The facsimiles of the diary forms and the framework analysis matrix that
the manuscript links to are in `artefacts/`. No script reads or writes
them.

## Licence

The MIT licence in `LICENSE` covers the code. The Creative Commons
Attribution 4.0 International licence (CC BY 4.0) covers the data in
`data/`, as stated in [`data/LICENSE-DATA.txt`](data/LICENSE-DATA.txt).
The data licence applies to the data from the commit that adds
`data/LICENSE-DATA.txt`.

## Setup

Install the pinned package versions in a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate        # Windows uses venv\Scripts\activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` lists the exact versions with which every output
was reproduced, using Python 3.12.3 on Linux. `requirements.txt` gives
minimum versions only.

### Font

The figures use Crimson Pro as a stand-in until the journal fonts are
used. `src/figstyle.py` holds the intended targets, Minion Pro
(`SERIF_FAMILY`) and Myriad Pro (`SANS_FAMILY`), and the fallback in use,
Crimson Pro (`FALLBACK_SERIF_FAMILY`). `FONT_SOURCE` selects the fallback
(the default) or a target. Setting the environment variable
`FIGSTYLE_FONT_SOURCE=target` (with `FIGSTYLE_FONT=serif` or `sans`)
selects a target without editing code.

Crimson Pro is a static TrueType serif under the SIL Open Font License.
It is part of TeX Live as the `crimsonpro` package, which is also
available from CTAN. The scripts need its Regular, Italic, Bold and
BoldItalic `.ttf` files. They look for them in this order.

1. The folder named by the environment variable `FIGSTYLE_FONT_DIR`, or
   by `FALLBACK_FONT_DIR` in `src/figstyle.py`. A relative path is read
   from the repository root.
2. Otherwise, TeX Live's `kpsewhich`, which finds the `crimsonpro`
   package wherever TeX Live is installed.

The files are registered in place. Nothing is copied or installed. If
they are not found, the scripts stop with a `FileNotFoundError` instead of
drawing in a substitute font.

Fonts are embedded in PDF output as TrueType (`pdf.fonttype` and
`ps.fonttype` are 42). Mathematical text uses the text font, with STIX for
glyphs it lacks. With the fallback font, italic mathematics such as the ρ
in fig06 uses STIXGeneral Italic.

## Running the pipeline

Run the scripts from `src/`, in this order. `analyze_participants.py`
reads the enriched CSV written by `analyze_events.py`.

```bash
cd src
export MPLBACKEND=Agg            # headless runs, no figure windows
python3 analyze_events.py        # step 1, event-level analysis
python3 analyze_participants.py  # step 2, participant-level analysis
python3 analyze_supplementary.py # step 3, platform shares, events per day, Table A4, B.3
```

Each `main()` function accepts path overrides, for example
`main(input_path="path/to/events.csv")`.

### Weeks and events per day

The week split follows documented decisions in `src/config.py`. Week 1
runs from 1 December to the end of 7 December 2025 and week 2 from
8 December to the end of 14 December. Events logged before 1 December
count as week 1 and, for events per day, as events of the first day.
Three events from one participant logged on 15 December are counted in
week 2. Events per day use a fixed period of 14 days.

## Running the tests

Run each test file with Python from the repository root.

```bash
python3 tests/test_rounding.py
python3 tests/test_pb_parsing.py
python3 tests/test_round_once.py
```

`test_round_once.py` runs the event-level pipeline into a temporary folder
and checks every reported number against an independent full-precision
calculation.

## Steps that cannot be reproduced from this repository

Two steps happened before this pipeline.

1. **Event exclusion.** Participants registered 275 events. Eleven were
   removed by hand before `data/events_20260324.csv` was created. These
   were incomplete registrations, events unrelated to music listening,
   and events that participants flagged as not problematic. The file
   with all 275 events is not in the repository, so the exclusion and
   the week 1 and week 2 totals of 141 and 134 cannot be rerun.
2. **Participant characteristics.** Table A1 and the figures in Section
   5.1 of the paper (gender, age, education, platforms, Gold-MSI, AMCA
   and self-assessed knowledge) come from questionnaire data that is not
   in the repository. No script produces them.

## Project structure

```
EUAA_DiaryStudy/
├── artefacts/                       # diary form facsimiles and framework matrix
├── data/
│   ├── events_20260324.csv          # diary events (input)
│   └── LICENSE-DATA.txt             # CC BY 4.0 for the data
├── outputs/                         # generated, not tracked
│   ├── figures/events/              # figures from analyze_events.py
│   └── tables/                      # CSV outputs and notes
├── src/
│   ├── config.py                    # paths, PB taxonomy, week rule, colours
│   ├── figstyle.py                  # fonts, mathematics font, embedding
│   ├── plotting.py                  # save_and_show() helper
│   ├── rounding.py                  # half-up rounding for reported numbers
│   ├── analyze_events.py            # step 1
│   ├── analyze_participants.py      # step 2
│   └── analyze_supplementary.py     # step 3
├── tests/                           # run each file with python
├── requirements.txt
├── requirements-lock.txt
└── LICENSE
```

Everything under `outputs/` is regenerated by the scripts and is not
tracked. `outputs/figures/participants/` stays empty because
`analyze_participants.py` draws no figures.

## Outputs

**`outputs/tables/`**

| File | Produced by | Contents |
|------|-------------|----------|
| `events_enriched.csv` | analyze_events.py | Diary events with binary PB item columns |
| `construct_probs_{INTENT,TEMP,TRIGGERS}.csv` | analyze_events.py | Category-level P(PB \| context), the unweighted mean of the item-level probabilities (no interval) |
| `conditional_probs_{INTENT,TEMP,TRIGGERS}.csv` | analyze_events.py | Item-level P(PB \| context) with Clopper-Pearson intervals, and an `exploratory_effect_size` column that is not reported in the paper (see `conditional_probs_NOTE.txt`) |
| `sensitivity_construct_probs_{INTENT,TEMP,TRIGGERS}.csv` | analyze_events.py | Sensitivity check, not reported in the paper (see `sensitivity_construct_probs_NOTE.txt`) |
| `likert_pb_correlations.csv` | analyze_events.py | Participant-level Spearman ρ between Likert items and PB items |
| `participant_summary.csv` | analyze_participants.py | One row per participant with PB rates and counts, context shares and Likert means |
| `platform_shares.csv`, `events_per_day.csv`, `table_A4_weekly.csv`, `loo_spearman_B3.csv`, `supplementary_comparison.csv` | analyze_supplementary.py | Values for Section 5.2, Table A4 and Appendix B.3, and their comparison with the paper |

**`outputs/figures/events/`**

Each figure is saved as a 300 dpi PNG and as a PDF of the same name. PDFs
carry no creation date, so repeated runs give identical files. There is
no `fig04`.

## Notes on the statistics

- **N=9.** All results are exploratory. Spearman ρ is reported as a
  descriptive summary. The rank-biserial r and Kendall's W are
  exploratory and are not reported in the paper. No significance test and
  no correction for multiple comparisons is applied.
- **Rounding.** Every reported number is rounded once, from its
  full-precision value, with the half-up rule (`decimal.ROUND_HALF_UP`
  applied to the shortest decimal representation of the float). Halves
  are rounded away from zero, so 3.25 becomes 3.3 and 0.125 becomes 0.13.
  Python's default rounding of halves to even is not used. CSV files that
  are not shown in the paper as such (for example `participant_summary.csv`
  and `table_A4_weekly.csv`) keep full precision.
  `tests/test_round_once.py` fails if any reported value is rounded twice.
- **Reproducibility.** No statistic uses resampling or a random seed. The
  only random element is the display jitter in `fig00`, seeded by
  `JITTER_SEED` in `src/config.py`.
- **Measurement model.** SB4 was reclassified as CI5.
  `analyze_events.py` reports reliability for both the original and the
  revised item sets.
- **O2 answers.** Free-text answers to the "Other" option are excluded
  from every analysis.

## Acknowledgment

The study design, methodology, statistical approach, and all analytical
decisions in this pipeline were developed by the research team. Claude 
Sonnet 5 (Anthropic) was used to refactor the codebase: cleaning up
structure, fixing portability issues, adding consistent documentation, 
and ensuring all figures are saved to disk.
