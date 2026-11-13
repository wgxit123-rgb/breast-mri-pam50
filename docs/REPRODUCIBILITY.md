# Source and reproducibility

The associated study is “Breast MRI prediction of PAM50 subtypes: a retrospective analysis of development performance and input limitations.” The archived source shows the implemented research methods. The original training and evaluation workflows require study-specific inputs and environment settings. Some historical runners expect the original Windows workspace and protocol hashes. Inspect historical scripts before importing or running them.

- The source for the original 743-case orchestration is omitted because it contains fixed patient identifiers. Related continuation and scheduling files remain available for inspection and depend on the omitted file.
- The balanced follow-up analysis uses the current interpreter by default, or the interpreter set in `BREAST_MRI_ANALYSIS_PYTHON`.
- In the molecular association script, the approved identifier for the duplicate-platform case must be supplied through `BREAST_MRI_DUPLICATE_PLATFORM_JOIN_ID` in a controlled analysis environment. The existing exclusion assertion still checks against that explicitly supplied identifier. No identifier is supplied in this repository. This version was syntax-checked only.
- The plotting scripts read saved aggregate outputs from their expected input paths. The aggregate data and table files are outside this source package. `tools/verify_integrity.py` pertains to the separate historical aggregate package.

`MANIFEST.json` records the snapshot provenance and file hashes. Cite the actual repository link and commit or release version when referencing this code. Any article DOI or software DOI should be cited once assigned and verified.
