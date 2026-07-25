# SoulCycle Analog Rider Network Simulation

This project simulates repeated rider co-attendance across a synthetic SoulCycle studio network and studies how familiarity, social ties, and coordinated attendance emerge over time.

## Current components

- Studio and market data loading
- Persistent studio schedule generation
- Recurring baseline class-slot generation
- Synthetic instructor population generation
- Market-tier behavioral parameter estimation
- Studio capacity vs instructor-demand diagnostics
- Baseline instructor-to-slot assignment
- Weekly class schedule realization (off weeks and substitutions)

## Data

The repository contains:

- `data/studios.csv`
- `data/active_instructors_final.csv`
- `data/instructors_sample.csv`

Real instructor names are used only to preserve aggregate active-population counts. Generated simulation instructors use fictitious names.

## Running the project

```bash
pip install -r requirements.txt
pytest
```

## Package layout

```text
src/soulcycle_network/
├── baseline_class_slot.py
├── baseline_instructor_schedule.py
├── class_slot_builder.py
├── config.py
├── instructor.py
├── instructor_assignment.py
├── instructor_generator.py
├── instructor_parameters.py
├── studio.py
├── studio_loader.py
├── studio_schedule.py
├── weekly_class_session.py
└── weekly_schedule.py
```

## Modeling notes

- Home-cluster allocation uses weekly bike supply (`rides_per_week * bikes_per_ride`) rather than slot counts.
- `CLASS_LOAD_STUDIO_EFFECT` in `config.py` is a fixed modeling assumption linking baseline class load to regular studio count.
- Instructor baseline loads are calibrated to total recurring class supply (1,615 slots) and studio allocations respect per-studio capacity.
