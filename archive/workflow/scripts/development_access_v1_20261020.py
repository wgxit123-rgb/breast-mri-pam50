"""Prospective development-target interface;no whole-cohort label source access."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports/development_access_v1_20261020"
LABEL_FIELDS = ("pid", "label", "outer", "inner_0", "inner_1", "inner_2")
SCORE_FIELDS = ("pid", "Module11_Prolif_score", "Module5_TcellBcell_score", "Mod10_ECM")
CLASSES = frozenset(("Basal", "Her2", "LumA", "LumB"))
SOURCE_FOLDS = "reports/pam50_v2/fold_manifest.csv"
SOURCE_SCORES = "reports/molecular_association_v2_20261019/development_association_input.csv"
SOURCE_PROTOCOL = "reports/pam50_v2/protocol.json"


@dataclass(frozen=True)
class LabelTarget:
    """Research target and partition ownership;never an imaging predictor."""

    pid: str
    label: str
    outer: int
    inner: tuple[int, int, int]


@dataclass(frozen=True)
class MolecularTarget:
    """Three observational molecular outcomes with explicit missing values."""

    pid: str
    scores: tuple[float | None, float | None, float | None]


@dataclass(frozen=True)
class DevelopmentTargets:
    """Separate typed targets from a content-bound development-only bundle."""

    labels: tuple[LabelTarget, ...]
    molecular: tuple[MolecularTarget, ...]


def sha(path: Path) -> str:
    """Return SHA256 of bounded local input bytes."""
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Target interface file exceeds64MiB")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(root: Path, relative: str) -> Path:
    """Reject absolute paths,traversal and symlinks escaping the approved root."""
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Relative contained target path required")
    resolved = (root / candidate).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def read_rows(path: Path, fields: tuple[str, ...], exact: bool) -> list[dict[str, str]]:
    """Read an allowlisted schema;reject surplus or malformed derived columns."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(fields).issubset(columns):
            raise ValueError("Missing or duplicate columns")
        if exact and tuple(columns) != fields:
            raise ValueError("Unexpected derived columns or order")
        rows = []
        for index, row in enumerate(reader):
            if index >= 2000 or None in row or any(row.get(key) is None for key in fields):
                raise ValueError("Malformed or excessive target rows")
            rows.append({key: row[key] for key in fields})
        return rows


def validate_rows(labels: list[dict[str, str]], scores: list[dict[str, str]],
                  expected_count: int) -> DevelopmentTargets:
    """Validate membership,partition ownership and target schemas without fitting."""
    if len(labels) != expected_count or len(scores) != expected_count:
        raise ValueError("Unexpected target denominator")
    ids = [row["pid"] for row in labels]
    score_ids = [row["pid"] for row in scores]
    if len(set(ids)) != expected_count or len(set(score_ids)) != expected_count:
        raise ValueError("Duplicate target identifiers")
    if ids != score_ids:
        raise ValueError("Foreign,target-missing or reordered patient identifiers")
    targets = []
    for row in labels:
        if row["label"] not in CLASSES:
            raise ValueError("Unknown or excluded primary label")
        outer = int(row["outer"])
        inner = tuple(int(row["inner_%d" % index]) for index in range(3))
        if outer not in range(3) or any(value not in (-1, 0, 1) for value in inner):
            raise ValueError("Invalid fold")
        if inner[outer] != -1 or any(inner[index] not in (0, 1) for index in range(3) if index != outer):
            raise ValueError("Invalid fitting/evaluation ownership")
        targets.append(LabelTarget(row["pid"], row["label"], outer, (inner[0], inner[1], inner[2])))
    molecular = []
    for row in scores:
        values = []
        for key in SCORE_FIELDS[1:]:
            value = None if row[key] == "" else float(row[key])
            if value is not None and not math.isfinite(value):
                raise ValueError("Nonfinite molecular score is not an imputed value")
            values.append(value)
        molecular.append(MolecularTarget(row["pid"], (values[0], values[1], values[2])))
    return DevelopmentTargets(tuple(targets), tuple(molecular))


def load_targets(root: Path, contract_relative: str, expected_contract_sha256: str) -> DevelopmentTargets:
    """Load only hash-bound derived targets;caller must supply the frozen contract SHA."""
    contract_path = contained(root, contract_relative)
    if sha(contract_path) != expected_contract_sha256:
        raise ValueError("Development contract SHA mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["schema"] != "development_target_access_v1" or contract["scope"] != "original_training_only":
        raise ValueError("Unrecognized target scope")
    files = contract["derived_files"]
    labels_path = contained(root, files["labels"]["path"])
    scores_path = contained(root, files["scores"]["path"])
    if sha(labels_path) != files["labels"]["sha256"] or sha(scores_path) != files["scores"]["sha256"]:
        raise ValueError("Target CSV SHA mismatch")
    labels = read_rows(labels_path, LABEL_FIELDS, exact=True)
    scores = read_rows(scores_path, SCORE_FIELDS, exact=True)
    return validate_rows(labels, scores, contract["patients"])


def write_json_once(path: Path, value: dict) -> None:
    """Preserve first records;never replace a receipt or contract."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def export() -> None:
    """Export once from already-development-only artifacts,without model work."""
    DEST.mkdir(exist_ok=False)
    source_names = (SOURCE_FOLDS, SOURCE_SCORES, SOURCE_PROTOCOL)
    sources = {}
    for relative in source_names:
        source = contained(ROOT, relative)
        expected = sha(source)
        backup = DEST / "preimplementation" / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup)
        if sha(backup) != expected:
            raise ValueError("Snapshot SHA mismatch")
        sources[relative] = {"sha256": expected, "backup": backup.relative_to(ROOT).as_posix()}
    write_json_once(DEST / "PREIMPLEMENTATION_SHA256.json", {"source_files": sources,
                     "script_sha256": sha(Path(__file__)), "new_experiment": False})
    protocol = json.loads((ROOT / SOURCE_PROTOCOL).read_text(encoding="utf-8"))
    if sources[SOURCE_FOLDS]["sha256"] != protocol["fold_manifest_sha256"]:
        raise ValueError("Folds differ from frozen V2protocol")
    labels = read_rows(ROOT / SOURCE_FOLDS, LABEL_FIELDS, exact=True)
    source_scores = read_rows(ROOT / SOURCE_SCORES, ("pid", "label", "outer") + SCORE_FIELDS[1:], exact=False)
    by_id = {row["pid"]: row for row in source_scores}
    if len(by_id) != len(source_scores):
        raise ValueError("Duplicate source molecular identifiers")
    scores = []
    for row in labels:
        scored = by_id[row["pid"]]
        if row["label"] != scored["label"] or row["outer"] != scored["outer"]:
            raise ValueError("Source label/fold mismatch")
        scores.append({key: scored[key] for key in SCORE_FIELDS})
    bundle = validate_rows(labels, scores, 748)
    if len(source_scores) != len(bundle.labels):
        raise ValueError("Foreign source identifiers")
    for name, fields, rows in (("labels.csv", LABEL_FIELDS, labels), ("molecular_targets.csv", SCORE_FIELDS, scores)):
        with (DEST / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    contract = {"schema": "development_target_access_v1", "scope": "original_training_only",
                "patients": 748, "created_utc": datetime.now(timezone.utc).isoformat(),
                "historical_whole_table_access_not_undone": True, "future_training_adopted": False,
                "OS_level_isolation": False, "molecular_targets_are_image_predictors": False,
                "source_files": sources, "derived_files": {
                    key: {"path": (DEST / name).relative_to(ROOT).as_posix(), "sha256": sha(DEST / name)}
                    for key, name in (("labels", "labels.csv"), ("scores", "molecular_targets.csv"))}}
    write_json_once(DEST / "contract.json", contract)
    contract_sha = sha(DEST / "contract.json")
    loaded = load_targets(ROOT, (DEST / "contract.json").relative_to(ROOT).as_posix(), contract_sha)
    if loaded != bundle:
        raise ValueError("Export/readback differs")
    for relative, row in sources.items():
        if sha(ROOT / relative) != row["sha256"]:
            raise ValueError("Original source changed during export")
    write_json_once(DEST / "EXPORT_READBACK.json", {"status": "derived_target_interface_checked_not_adopted",
                     "patients": len(loaded.labels), "molecular_target_records": len(loaded.molecular),
                     "source_to_view_records_exact": True, "readback_exact": True,
                     "contract_sha256": contract_sha, "original_sources_unchanged": True,
                     "whole_cohort_source_tables_opened_this_export": False,
                     "training_or_prediction": False, "new_statistics": False,
                     "historical_label_access_resolved": False, "all_release_gates_unchanged": True})
    print(json.dumps({"status": "prospective_interface_checked_not_adopted", "patients": 748,
                      "contract_sha256": contract_sha}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["export"])
    parser.parse_args()
    export()
