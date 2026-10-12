"""Contrôle des résultats et des partitions."""
from __future__ import annotations
import hashlib
import json

import numpy as np
import pandas as pd

from .data import ROOT, INPUTS, NAMES, TARGET, load_data, make_groups, now
from .study import metrics, observed_labels, masked_training, save_json


def main():
    out = ROOT / "output"
    frame = load_data()
    audit = pd.read_csv(out / "partitions.csv", index_col="row_id")
    predictions = pd.read_csv(out / "test_predictions.csv", index_col="row_id")
    scores = pd.read_csv(out / "test_metrics.csv").set_index("model")
    cv = pd.read_csv(out / "supervised_cv.csv")
    ssl = pd.read_csv(out / "semi_supervised_cv.csv")
    pseudo = pd.read_csv(out / "pseudo_labels_audit.csv")
    pca = json.loads((out / "pca_summary.json").read_text(encoding="utf-8"))
    final = json.loads((out / "test_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    checks = []

    def check(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    check(len(audit) == len(frame) and audit.index.is_unique, "Toutes les lignes ont une partition unique")
    check(set(INPUTS).isdisjoint(NAMES[30:]), "Aucun résultat mécanique ou microstructural parmi les entrées")
    groups = make_groups(frame)
    check(np.array_equal(groups.to_numpy(), audit.group_id.to_numpy()), "Groupes reproductibles")
    dev = audit.index[audit.partition == "development"]
    test = audit.index[audit.partition == "test"]
    check(not set(groups.loc[dev]) & set(groups.loc[test]), "Aucun groupe commun développement/test")
    check(set(predictions.index) == set(frame.loc[test].dropna(subset=[TARGET]).index),
          "Seuls les vrais labels des groupes de test sont évalués")
    threshold = float(frame.loc[dev, TARGET].median())
    check(threshold == final["threshold"], "Seuil final calculé sur le développement uniquement")
    check(np.array_equal(predictions.truth, observed_labels(frame.loc[predictions.index, TARGET], threshold)),
          "Vérités du test cohérentes avec le seuil figé")
    # Recalculer les scores à partir des prédictions exportées.
    for name, score in scores.iterrows():
        calculated = metrics(predictions.truth, predictions[name])
        check(all(np.isclose(score[key], value) for key, value in calculated.items()),
              f"Métriques de test recomputées : {name}")
    # Vérifier séparément chaque partition de validation.
    for fold in sorted(audit.loc[dev].validation_fold.unique()):
        train = audit.index[(audit.partition == "development") & (audit.validation_fold != fold)]
        valid = audit.index[(audit.partition == "development") & (audit.validation_fold == fold)]
        check(not set(groups.loc[train]) & set(groups.loc[valid]), f"Aucun groupe commun au pli {fold}")
        threshold = float(frame.loc[train, TARGET].median())
        records = cv[cv.fold == fold]
        check(np.allclose(records.threshold, threshold), f"Seuil CV du pli {fold} appris sans validation")
        check((records.n_train == frame.loc[train, TARGET].notna().sum()).all(), f"Effectifs train du pli {fold}")
        for _, entry in pseudo[pseudo.fold == fold].iterrows():
            visible, masked_threshold, labels = masked_training(frame, train, entry.fraction, int(entry.seed))
            check(masked_threshold == entry.threshold, f"Seuil SSL pli {fold}, fraction {entry.fraction}, graine {entry.seed}")
            check(len(visible) == entry.n_visible and (labels == -1).sum() == entry.n_unlabeled,
                  f"Masquage SSL pli {fold}, fraction {entry.fraction}, graine {entry.seed}")
    check(len(ssl) == 2 * len(pseudo), "Deux modèles comparés pour chaque essai semi-supervisé")
    check((pseudo.n_pseudo <= pseudo.n_unlabeled).all(), "Pseudo-labels limités aux données non labélisées d'apprentissage")
    check(set(pca["features"]).issubset(INPUTS) and pca["n"] == len(dev), "ACP sans sorties ni test")
    check(manifest["data_sha256"] == hashlib.sha256((ROOT / "data/raw/welddb.tar").read_bytes()).hexdigest(),
          "Empreinte de l'archive vérifiée")
    check(manifest["source_sha256"] == hashlib.sha256((ROOT / "scripts/study.py").read_bytes()).hexdigest(),
          "Empreinte de la source de l'étude vérifiée")
    record = {"verified_at": now(), "check_count": len(checks), "checks": checks,
              "test_predictions_sha256": hashlib.sha256((out / "test_predictions.csv").read_bytes()).hexdigest(),
              "scope": "Cohérence des artefacts et protocole ; aucune garantie industrielle."}
    save_json(out / "verification.json", record)
    print(f"{len(checks)} vérifications des artefacts réussies.")


if __name__ == "__main__":
    main()
