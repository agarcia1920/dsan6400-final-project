# SoulCycle Analog Rider Network Simulation

Boutique fitness companies frequently describe themselves as communities rather than simply collections of exercise classes. However, much of the social interaction that may occur within these environments is analog: riders repeatedly attend the same studios, instructors, and class times without necessarily interacting through an explicit digital friendship network. This project constructs a synthetic SoulCycle environment to examine whether repeated in-person co-attendance is sufficient to generate meaningful social-network structure over time.

The model simulates studios, instructors, recurring class schedules, weekly schedule changes, rider attendance behavior, shared class participation, familiarity, active social ties, and coordinated attendance. Rather than attempting to recreate the behavior of any individual rider, the project focuses on the broader structural question of whether recurring participation within a geographically constrained fitness system can produce localized social relationships.

The calibrated 52-week implementation is frozen under the Git tag `v1.0-calibrated`.

## Research Questions

This project focuses on the following primary research question:

> Can repeated analog co-attendance in a geographically constrained boutique-fitness system produce meaningful social structure without an explicit digital social network?

To address this broader question, the analysis considers three supporting questions:

1. Does repeated co-attendance generate nontrivial familiarity and active social ties?
2. To what extent do geography, studio preferences, and recurring attendance opportunities shape the resulting network?
3. Does coordinated attendance expand riders' social networks, or does it reinforce relationships that have already formed?

## Model overview

The simulation consists of two interacting systems: the company environment and the rider environment.

Once riders are assigned to classes, the model records repeated co-attendance between rider pairs. These repeated encounters form the basis of three related network definitions:

- **Co-attendance network:** an edge represents at least one shared class.
- **Familiarity network:** an edge forms after at least three shared classes.
- **Active social network:** an edge requires at least six shared classes and an active tie strength of at least 1.0 after decay.

Tie strength decays over time when riders do not continue attending together, while coordinated attendance allows established relationships to influence future class selection.

Generated instructor names are fictitious.

## Methodology (analysis)

Simulation outputs under `outputs/` are treated as the analytical dataset. Longitudinal checkpoints are weeks **1, 4, 8, 13, 26, 39, 52**. Course-style methods live in `src/soulcycle_network/analysis/`: graph summaries, GCC metrics, degree distributions, centrality on the GCC, categorical assortativity, Louvain communities, attendance-shuffle null models, and ER / preferential attachment / attractiveness / fitness comparisons.

## Assumptions (v1.0-calibrated)

Operational parameters match `src/soulcycle_network/config.py` and the frozen tag `v1.0-calibrated`. Exported CSVs are assumed to come from the same config. Graph summaries distinguish **population-wide** metrics (all 10k nodes when isolates are included) from **edge-induced / GCC** metrics where noted in the notebooks.

## Repository structure

```text
analysis/      Three notebooks (validation, network analysis, experiments/results)
data/          Studio and instructor inputs (unchanged filenames)
outputs/       Full experiment exports (gitignored locally)
results/       Flat compact CSVs for notebooks and submission
scripts/       run_simulation, run_experiment, run_analysis
src/           Simulation package + analysis/ subpackage
tests/         conftest + test_model_rules + test_pipeline
```

## Commands

```bash
pytest
python scripts/run_simulation.py --seed 6400 --scenario baseline
python scripts/run_experiment.py
python scripts/run_analysis.py --task all
```

`run_analysis.py --task all` runs calibration, longitudinal rebuild, null models, model comparisons, and compact export to `results/`.

Then open `analysis/01_model_validation.ipynb`, `02_network_analysis.ipynb`, and `03_experiments_and_results.ipynb`.
