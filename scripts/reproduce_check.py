"""Comparer deux exécutions."""
import argparse
import hashlib
import json
from pathlib import Path

from .data import ROOT, now

FILES = ["partitions.csv", "selected_models.json", "semi_supervised_cv.csv",
         "pseudo_labels_audit.csv", "test_predictions.csv", "test_metrics.csv"]


def fingerprints():
    return {name: hashlib.sha256((ROOT / "output" / name).read_bytes()).hexdigest() for name in FILES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Conserver les empreintes avant de rejouer l'étude.")
    parser.add_argument("--reference", type=Path, help="Comparer à une capture précédente.")
    args = parser.parse_args()
    # Comparer les fichiers stables, pas les durées de calcul.
    actual = fingerprints()
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(actual, indent=2), encoding="utf-8")
        print(f"Empreintes conservées dans {args.snapshot}")
    elif args.reference:
        expected = json.loads(args.reference.read_text(encoding="utf-8"))
        compared = {name: {"expected": expected[name], "actual": actual[name], "identical": actual[name] == expected[name]}
                    for name in FILES}
        result = {"checked_at": now(), "compared_files": compared, "all_identical": all(r["identical"] for r in compared.values()),
                  "exclusions": "Les durées d'entraînement et horodatages sont variables et exclus."}
        (ROOT / "output/reproduction_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if not result["all_identical"]:
            raise SystemExit("Différence entre exécutions : consulter output/reproduction_check.json")
        print(f"{len(FILES)} exports stables identiques entre les deux exécutions.")
    else:
        parser.error("Indiquer --snapshot ou --reference.")
