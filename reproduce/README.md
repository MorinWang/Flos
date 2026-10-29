# Reproducing the paper

This folder holds the code behind every number, table and figure of the paper and its supplement: the benchmark
generator, the research implementation of FLoS, the baselines, the experiment scripts with the exact arguments used,
the result files, and the scripts that build the tables and figures from those files. Source files keep their
experiment IDs (`A17_...`, `A39_...`); the tables below map them to the paper. Paths and the torch device are read
from `src/paths.py`.

## Setup

```bash
pip install -r requirements.txt       # the versions the reported results were produced with
```

Data is read from `./data`, results go to `./outputs` and graph eigendecompositions are cached in `./cache`; set
`FLOS_DATA`, `FLOS_OUT`, `FLOS_CACHE` and `FLOS_EXTERNAL` to use other folders, and `FLOS_DEVICE` (`cuda` or `cpu`)
to choose the device.

## Data

| Reanalysis | Variables | Script | Notes |
|---|---|---|---|
| NCEP/NCAR R1 | air, slp, pres, pottmp, rhum, omega, lftx | `data_prep/fetch_ncep.sh` | 73 x 144, 1948-01 to 2026-02 |
| GPCP v2.3 | precipitation | `data_prep/fetch_ncep.sh` | 1979-01 to 2026-06, listed with the NCEP/NCAR group |
| ERA5 | air, slp, pres, pottmp, rhum, omega, cape | `data_prep/fetch_era5.py`, then `convert_era5.py` | needs a Copernicus CDS API key; 2.5 degrees, aligned month by month with NCEP |
| 20CRv3 | air, slp, pres, pottmp, rhum, omega, cape | `data_prep/fetch_convert_20cr.py` | NOAA PSL OPeNDAP, 1948-01 to 2015-12, bilinear to the NCEP mesh |

Run `fetch_ncep.sh` first: the ERA5 and 20CRv3 converters take their mesh and time axis from the NCEP air temperature
file. `fetch_ncep.sh` cuts the files to the months used in the paper, so the benchmark is regenerated exactly even
after NOAA appends new months.

## Baselines from third parties

`scripts/00_fetch_baselines.sh` clones the two baselines that are run from their official code, at the commits used:
ScaledGD (https://github.com/hdong920/Tensor_RPCA_ScaledGD, `a0f97aa`) and LR-STSS
(https://github.com/Rachita-Mondal/Anomaly_Detection, `ac79109`). Both are imported without modification. GLOSS and
TV-RPCA are line-by-line ports of the authors' MATLAB code, included in `src/`.

## Experiments

Run the scripts in `scripts/` in order; each writes its CSVs to `$FLOS_OUT`.

| Script | Code | Used for |
|---|---|---|
| `01_benchmark.sh` | `A5_leaderboard.py --step eig` | eigendecomposition cache |
| `02_flos_dev.sh` | `A15_lam_sweep.py` | selecting lambda on the development variables |
| `03_flos_heldout.sh` | `A17_holdout.py` | Table II (FLoS, NCEP and ERA5), Fig. 8 |
| `04_pcp.sh` | `A18_pcp_sweep.py` | Table II (PCP) |
| `05_eof_gloss_direct.sh` | `A21_opp_sweep.py` | Table II (EOF, GLOSS, magnitude ranking) |
| `06_scaledgd.sh` | `A24_scaledgd.py` | Table II (ScaledGD) |
| `07_lrstss.sh` | `A23_mondal.py` | Table II (LR-STSS) |
| `08_20crv3.sh` | `A25_third.py` | Table II (20CRv3) |
| `09_ablation.sh` | `A28_ablate_20cr.py`, `A29_sweep_20cr.py`, `A22_ablate_sparse.py` | Section VI, Table S6, Fig. 8 |
| `10_spectrum.sh` | `A28_spectrum.py` | spectra of the fields (Section V-B) |
| `11_tables_figures.sh` | `main_table.py`, `sec54_numbers.py`, `A29_figure.py` | main-table CSVs |
| `12_cpu_timing.sh` | `A5_leaderboard.py` | CPU run times (optional) |
| `13_convex_core.sh` | `A30_core.py`, `A30b_core_sweep.py` | FLoS-C and high-pass baselines: Table III, Table S8 |
| `14_exponent_frpcag.sh` | `A31_exponent.py`, `A32_frpcag.py` | exponent (Fig. 9b, Table S7), FRPCAG (Table II, Table S9) |
| `15_contrast_taper.sh` | `A33_contrast.py`, `A36_taper.py`, `A36b_retune.py` | roughness contrast (Fig. 5), tapered insertion (Fig. 6, Table S5) |
| `16_fairness.sh` | `A37_fair.py`, `A37_gloss.py`, `A37_rpcag.py` | EOF ranks, high-pass baselines, rounds (Fig. 9a), GLOSS (Fig. 4) |
| `17_truncation.sh` | `A39_notrunc.py` | FLoS without truncation (Table III, Section V-D) |
| `18_examples.sh` | `A35_example.py`, `A35b_example_taper.py`, `A38_*` | Figs. 2, 3 and 7 |
| `19_paper_tables_figures.sh` | `analysis/*.py` | every table and figure, from `results/` |

FLoS keeps alpha = 0.85, s = 0.25, p = 1.25 and lambda = 0.72, at most 500 rounds and tolerance 1e-5. The FLoS runs of
the paper used a CUDA GPU in float32; GPU single precision and CPU double precision agree to 1e-4 in AUPRC.

## Result files

`results/` contains the outputs of all runs, grouped as the analysis scripts expect them. To rebuild the tables from
your own runs, copy the CSVs from `$FLOS_OUT` into the matching folder:

| Folder | Files |
|---|---|
| `results/main/` | `main_table.csv`, `main_table_long.csv` (scripts 03 to 08, 11), `A25_*.csv` |
| `results/heldout/` | `A15_*.csv`, `A17_holdout_*.csv`, `A22_ablate_*.csv` |
| `results/ablation/` | `A28_*.csv`, `A29_*.csv` |
| `results/tpami/` | `A30_*` to `A39_*` (scripts 13 to 18) |

`analysis/make_readme_assets.py` draws the animations of the top-level README from the same files;
`results/tpami/flos_iterations.npz` holds the rounds recorded by `examples/record_iterations.py`.
