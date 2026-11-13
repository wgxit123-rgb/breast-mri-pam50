# Breast MRI PAM50 development study

Python source for a retrospective study of breast MRI prediction of expression-defined PAM50 subtypes. The repository documents the implemented methods for preprocessing, classification, development comparisons, aggregate plotting and conditional molecular-score analysis.

## Browse the implementation

| Task | Source |
|---|---|
| Model and preprocessing | [pam50_v2_core.py](archive/workflow/scripts/pam50_v2_core.py) |
| Baseline model fitting and feature extraction | [pam50_v2_train.py](archive/workflow/scripts/pam50_v2_train.py) |
| Nominal-phase control | [nominal_phase_control.py](archive/workflow/scripts/nominal_phase_control.py) |
| Full-field window handling | [window_patient_candidate.py](archive/workflow/scripts/window_patient_candidate.py) |
| Signed enhancement and class-balanced sampling | [model_improvement_train_20261027.py](archive/workflow/scripts/model_improvement_train_20261027.py) |
| Conditional molecular associations | [molecular_association_v2_20261019.py](archive/workflow/scripts/molecular_association_v2_20261019.py) |
| Plotting saved aggregate results | [plot_aggregates.py](tools/plot_aggregates.py) |

## Check the download

From the repository root, use Python 3.10 or later:

```bash
python -B tools/verify_release.py
```

The command uses only the standard library. It verifies the manifest hashes and parses Python source without importing the research scripts.

## Scope and dependencies

This is a source snapshot for inspecting the implementation. The historical workflows depend on original study inputs, fold assignments, runtime bindings and local job controls. Those inputs and any patient-level data, images, weights, outputs and result tables are outside this repository. Some archived runners use Windows process supervision and have incomplete historical dependencies. The archived plotting and aggregate integrity utilities require their corresponding separately supplied aggregate inputs.

The dependency inventories are [requirements-research.txt](requirements-research.txt) and [requirements-figures.txt](requirements-figures.txt). [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) maps the files to the study and describes the changes made to this snapshot. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for rights information.
