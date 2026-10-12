"""ACP, modèles supervisés et self-training."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

from .data import ROOT, INPUTS, NUMERIC, CATEGORICAL, NAMES, TARGET, canonical_id, dictionary, load_data, make_groups, now

OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
SEED = 42


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_figure(name):
    plt.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()


class FeatureFilter(BaseEstimator, TransformerMixin):
    """Retirer les colonnes constantes ou trop incomplètes."""
    def __init__(self, max_missing=0.60):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        self.columns_ = [c for c in X if X[c].isna().mean() <= self.max_missing
                         and X[c].nunique(dropna=True) > 1]
        if not self.columns_:
            raise ValueError("Aucune entrée exploitable dans l'apprentissage.")
        self.removed_ = [c for c in X if c not in self.columns_]
        return self

    def transform(self, X):
        return X[self.columns_].copy()


def numeric_columns(X):
    return [c for c in X if c in NUMERIC]


def categorical_columns(X):
    return [c for c in X if c in CATEGORICAL]


def preprocessing(scale=False):
    # Médiane pour les nombres, one-hot pour les catégories.
    numerical_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numerical_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="manquant")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("filter", FeatureFilter()),
        ("transform", ColumnTransformer([
            ("num", Pipeline(numerical_steps), numeric_columns),
            ("cat", categorical, categorical_columns),
        ], sparse_threshold=0)),
    ])


def make_model(name, params=None):
    params = params or {}
    if name == "majoritaire":
        model = DummyClassifier(strategy="most_frequent")
    elif name == "knn":
        model = KNeighborsClassifier(**params)
    elif name == "arbre":
        model = DecisionTreeClassifier(random_state=SEED, **params)
    elif name == "foret":
        model = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                       random_state=SEED, n_jobs=2, **params)
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", preprocessing(scale=name == "knn")), ("model", model)])


def observed_labels(values, threshold):
    """Une cible absente reste à -1."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isnan(values), -1, (values >= threshold).astype(int))


def metrics(y, prediction):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=[0, 1], zero_division=0)
    return {"f1_macro": float(f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, prediction)),
            **{f"{metric}_{c}": float(values[c]) for metric, values in
               [("precision", precision), ("recall", recall), ("f1", f1), ("support", support)]
               for c in [0, 1]}}


def assert_separate(groups, train, valid):
    assert not set(groups.loc[train]) & set(groups.loc[valid]), "Fuite de groupe"


def masked_training(frame, train_ids, fraction=1.0, seed=42):
    """Seuls les labels visibles servent à calculer le seuil."""
    known_ids = frame.loc[train_ids].index[frame.loc[train_ids, TARGET].notna()]
    if fraction == 1.0:
        visible_ids = known_ids.to_numpy()
    else:

        rng = np.random.default_rng(seed)
        count = max(4, int(np.ceil(fraction * len(known_ids))))
        visible_ids = np.sort(rng.choice(known_ids, size=count, replace=False))
    # Ne pas utiliser les valeurs des labels masqués pour fixer le seuil.
    threshold = float(frame.loc[visible_ids, TARGET].median())
    y = pd.Series(-1, index=train_ids, dtype=int)
    y.loc[visible_ids] = observed_labels(frame.loc[visible_ids, TARGET], threshold)
    assert set(y[y >= 0]) == {0, 1}
    return visible_ids, threshold, y


def prepare_partitions(frame, groups):
    # Séparer aussi les lignes sans cible.
    dev_pos, test_pos = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED)
                              .split(frame, groups=groups))
    dev, test = frame.index[dev_pos], frame.index[test_pos]
    assert_separate(groups, dev, test)
    # Réduire le nombre de plis seulement si les classes sont mal réparties.
    for n_splits in [5, 3]:
        splits = []
        for train_pos, valid_pos in GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(
                frame.loc[dev], groups=groups.loc[dev]):
            train_ids, valid_ids = dev[train_pos], dev[valid_pos]
            known_train = frame.loc[train_ids].index[frame.loc[train_ids, TARGET].notna()]
            known_valid = frame.loc[valid_ids].index[frame.loc[valid_ids, TARGET].notna()]
            threshold = float(frame.loc[known_train, TARGET].median())
            if len(known_valid) == 0 or set(observed_labels(frame.loc[known_valid, TARGET], threshold)) != {0, 1}:
                break
            assert_separate(groups, train_ids, valid_ids)
            splits.append((train_ids, valid_ids))
        if len(splits) == n_splits:
            break
    else:
        raise ValueError("Les groupes ne permettent pas trois plis avec les deux classes.")
    audit = frame[["weld_id", TARGET]].copy()
    audit["canonical_id"] = audit.weld_id.map(canonical_id)
    audit["group_id"] = groups
    audit["partition"] = "development"
    audit.loc[test, "partition"] = "test"
    audit["validation_fold"] = -1
    for fold, (_, valid) in enumerate(splits):
        audit.loc[valid, "validation_fold"] = fold
    audit.to_csv(OUTPUT / "partitions.csv", encoding="utf-8", index=True)
    return dev, test, splits


def describe(frame, groups, dev):
    pd.DataFrame(frame.attrs["conversion_audit"]).to_csv(OUTPUT / "conversion_audit.csv", index=False)
    dictionary().to_csv(OUTPUT / "dictionary.csv", index=False)
    missing = frame.isna().mean().rename("missing_fraction")
    pd.concat([missing, frame.notna().sum().rename("observed")], axis=1).to_csv(OUTPUT / "missing.csv")
    frame.select_dtypes("number").describe().T.to_csv(OUTPUT / "descriptive_statistics.csv")
    summary = {"rows": len(frame), "columns": len(frame.columns), "labeled": int(frame[TARGET].notna().sum()),
               "unlabeled": int(frame[TARGET].isna().sum()), "groups": int(groups.nunique()),
               "unique_ids": int(frame.weld_id.nunique()), "unique_profiles": len(frame[INPUTS].drop_duplicates()),
               "full_duplicates": int(frame.duplicated().sum()),
               "converted_cells": len(frame.attrs["conversion_audit"]),
               "censored_input_cells": sum(r["action"].startswith("censored") and r["feature"] in INPUTS
                                           for r in frame.attrs["conversion_audit"]),
               "target_min": float(frame[TARGET].min()), "target_max": float(frame[TARGET].max()),
               "target_median_descriptive": float(frame[TARGET].median())}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), gridspec_kw={"width_ratios": [1.4, 1]})
    missing[INPUTS].sort_values().mul(100).plot.bar(ax=axes[0], color="#497e9a")
    axes[0].axhline(60, color="#ae483e", ls="--", lw=1)
    axes[0].set_ylabel("Valeurs absentes (%)")
    axes[0].tick_params(axis="x", labelsize=7)
    frame[TARGET].dropna().plot.hist(bins=25, ax=axes[1], color="#497e9a", edgecolor="white")
    axes[1].set_xlabel("Résistance à la traction (MPa)")
    axes[1].set_ylabel("Nombre d'observations")
    fig.tight_layout()
    save_figure("data_overview")

    labeled = frame[TARGET].notna()
    differences = []
    for col in NUMERIC:
        differences.append({"feature": col, "median_labeled": frame.loc[labeled, col].median(),
                            "median_unlabeled": frame.loc[~labeled, col].median(),
                            "missing_labeled": frame.loc[labeled, col].isna().mean(),
                            "missing_unlabeled": frame.loc[~labeled, col].isna().mean()})
    pd.DataFrame(differences).to_csv(OUTPUT / "labeled_unlabeled.csv", index=False)
    pd.crosstab(frame.weld_type, labeled, normalize="columns").to_csv(OUTPUT / "process_labeled_unlabeled.csv")
    retained = FeatureFilter().fit(frame.loc[dev, NUMERIC]).columns_
    corr = frame.loc[dev, retained].corr()
    corr.to_csv(OUTPUT / "correlations.csv")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1)
    plt.title("Corrélations des entrées numériques - développement")
    save_figure("correlations")
    save_json(OUTPUT / "data_summary.json", summary)
    return summary


def pca_analysis(frame, dev):
    X = frame.loc[dev, NUMERIC]
    selector = FeatureFilter().fit(X)
    cols = selector.columns_
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    # L'ACP porte sur les entrées numériques du développement.
    Z = prep.fit_transform(X[cols])
    pca = PCA().fit(Z)
    coordinates = pca.transform(Z)
    assert np.isfinite(coordinates).all()

    loadings = np.corrcoef(Z.T, coordinates[:, :2].T)[:len(cols), len(cols):]
    pd.DataFrame(loadings, index=cols, columns=["PC1", "PC2"]).to_csv(OUTPUT / "pca_loadings.csv")
    pd.DataFrame(coordinates[:, :2], index=dev, columns=["PC1", "PC2"]).to_csv(OUTPUT / "pca_scores.csv")
    cumulative = pca.explained_variance_ratio_.cumsum()
    threshold = float(frame.loc[dev, TARGET].median())
    labels = observed_labels(frame.loc[dev, TARGET], threshold)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    axes[0].plot(np.arange(1, len(cols)+1), cumulative * 100, marker=".", color="#315e73")
    axes[0].axhline(80, ls="--", color="#ae483e", lw=1)
    axes[0].set(xlabel="Nombre de composantes", ylabel="Variance cumulée (%)", ylim=(0, 103))
    for label, color, text in [(-1, "#aaaaaa", "Sans mesure"), (0, "#ae483e", "Faible"), (1, "#315e73", "Élevée")]:
        mask = labels == label
        axes[1].scatter(coordinates[mask, 0], coordinates[mask, 1], s=9, alpha=.55, c=color, label=text)
    axes[1].set(xlabel="PC1", ylabel="PC2")
    axes[1].legend(fontsize=7)
    for name, (x, y) in zip(cols, loadings):
        axes[2].arrow(0, 0, x, y, color="#497e9a", alpha=.75, width=.006, head_width=.035, length_includes_head=True)
        axes[2].text(x * 1.08, y * 1.08, name.replace("temperature", "temp."), fontsize=6)
    axes[2].add_patch(plt.Circle((0, 0), 1, fill=False, color="#bbbbbb"))
    axes[2].set(xlabel="Corrélation avec PC1", ylabel="Corrélation avec PC2", xlim=(-1.35, 1.35), ylim=(-1.3, 1.3))
    axes[2].set_aspect("equal")
    fig.tight_layout()
    save_figure("pca")
    info = {"features": cols, "removed": selector.removed_, "n": len(dev),
            "pc1": float(pca.explained_variance_ratio_[0]), "pc2": float(pca.explained_variance_ratio_[1]),
            "n_80_percent": int(np.searchsorted(cumulative, .80) + 1),
            "top_pc1": pd.Series(abs(loadings[:, 0]), index=cols).nlargest(4).index.tolist(),
            "top_pc2": pd.Series(abs(loadings[:, 1]), index=cols).nlargest(4).index.tolist()}
    save_json(OUTPUT / "pca_summary.json", info)
    return info


def choose_simpler(candidates):
    """Choisir le réglage le plus simple à 0,01 du meilleur F1."""
    best = max(c["f1_mean"] for c in candidates)
    return next(c for c in candidates if c["f1_mean"] >= best - .01)


def supervised(frame, groups, splits):
    grids = {
        "majoritaire": [{}],
        "knn": [{"n_neighbors": k} for k in [21, 11, 5, 3]],
        "arbre": [{"max_depth": depth, "min_samples_leaf": leaf} for depth in [3, 5, 10] for leaf in [10, 5]],
        "foret": [{"max_depth": depth, "min_samples_leaf": leaf} for depth in [5, 10, None] for leaf in [5, 2]],
    }
    rows, choices = [], {}
    # Comparer les réglages sur les mêmes plis.
    for name, settings in grids.items():
        candidates = []
        for params in settings:
            scores = []
            for fold, (train_ids, valid_ids) in enumerate(splits):
                train = frame.loc[train_ids].index[frame.loc[train_ids, TARGET].notna()]
                valid = frame.loc[valid_ids].index[frame.loc[valid_ids, TARGET].notna()]
                assert_separate(groups, train_ids, valid_ids)
                threshold = float(frame.loc[train, TARGET].median())
                y_train = observed_labels(frame.loc[train, TARGET], threshold)
                y_valid = observed_labels(frame.loc[valid, TARGET], threshold)
                if name == "knn" and params["n_neighbors"] > len(train):
                    continue
                model = make_model(name, params)
                started = time.perf_counter()
                model.fit(frame.loc[train, INPUTS], y_train)
                prediction = model.predict(frame.loc[valid, INPUTS])
                row = {"model": name, "params": json.dumps(params), "fold": fold,
                       "threshold": threshold, "n_train": len(train), "n_valid": len(valid),
                       "seconds": time.perf_counter()-started, **metrics(y_valid, prediction)}
                rows.append(row)
                scores.append(row)
            if len(scores) != len(splits):
                continue
            candidates.append({"model": name, "params": params,
                               "f1_mean": float(np.mean([s["f1_macro"] for s in scores])),
                               "f1_std": float(np.std([s["f1_macro"] for s in scores], ddof=1)),
                               "accuracy_mean": float(np.mean([s["accuracy"] for s in scores]))})
        choices[name] = choose_simpler(candidates)
        print(f"CV {name}: F1={choices[name]['f1_mean']:.3f} {choices[name]['params']}", flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "supervised_cv.csv", index=False)

    winner = choose_simpler([choices[name] for name in ["arbre", "knn", "foret"]])
    save_json(OUTPUT / "selected_models.json", {"models": choices, "winner": winner})
    return choices, winner


def fit_pair(frame, train_ids, valid_ids, params, fraction, seed):
    visible, threshold, y_all = masked_training(frame, train_ids, fraction, seed)
    valid = frame.loc[valid_ids].index[frame.loc[valid_ids, TARGET].notna()]
    truth = observed_labels(frame.loc[valid, TARGET], threshold)
    if set(truth) != {0, 1}:
        raise ValueError("Validation à une seule classe.")
    # Même préparation pour comparer les deux forêts.

    prep = preprocessing().fit(frame.loc[visible, INPUTS])
    X_all = prep.transform(frame.loc[train_ids, INPUTS])
    X_visible = prep.transform(frame.loc[visible, INPUTS])
    X_valid = prep.transform(frame.loc[valid, INPUTS])
    assert np.isfinite(X_all).all() and np.isfinite(X_valid).all()
    forest = make_model("foret", params).named_steps["model"]
    forest.fit(X_visible, y_all.loc[visible].to_numpy())
    base_pred = forest.predict(X_valid)
    ssl = SelfTrainingClassifier(estimator=make_model("foret", params).named_steps["model"],
                                 threshold=.90, max_iter=10)
    # Les lignes à -1 reçoivent éventuellement un pseudo-label.
    ssl.fit(X_all, y_all.to_numpy())
    ssl_pred = ssl.predict(X_valid)
    originally_unlabeled = y_all.to_numpy() == -1
    accepted = originally_unlabeled & (ssl.transduction_ != -1)
    # Vérifier les pseudo-labels masqués après l'apprentissage.
    hidden_truth = frame.loc[train_ids, TARGET].notna().to_numpy() & originally_unlabeled
    checked = accepted & hidden_truth
    pseudo_accuracy = (float(accuracy_score(observed_labels(frame.loc[train_ids, TARGET], threshold)[checked],
                                           ssl.transduction_[checked])) if checked.any() else None)
    details = {"fraction": fraction, "seed": seed, "threshold": threshold, "n_visible": len(visible),
               "n_unlabeled": int(originally_unlabeled.sum()), "n_pseudo": int(accepted.sum()),
               "pseudo_0": int((ssl.transduction_[accepted] == 0).sum()),
               "pseudo_1": int((ssl.transduction_[accepted] == 1).sum()),
               "iterations": int(ssl.n_iter_), "termination": str(ssl.termination_condition_),
               "checked_pseudo": int(checked.sum()), "pseudo_accuracy_masked": pseudo_accuracy}
    predictions = pd.DataFrame({"truth": truth, "foret": base_pred, "self_training": ssl_pred}, index=valid)
    return [(method, metrics(truth, pred)) for method, pred in [("foret", base_pred), ("self_training", ssl_pred)]], details, predictions


def semi_supervised(frame, groups, splits, params):
    rows, audit = [], []
    for fraction in [1.0, .25, .50]:
        seeds = [42] if fraction == 1 else [42, 43, 44]
        for seed in seeds:
            for fold, (train, valid) in enumerate(splits):
                assert_separate(groups, train, valid)
                results, detail, predictions = fit_pair(frame, train, valid, params, fraction, seed)
                detail["fold"] = fold
                audit.append(detail)
                for name, score in results:
                    rows.append({"model": name, "fraction": fraction, "seed": seed, "fold": fold, **score})
        print(f"Semi-supervisé : fraction={fraction:.0%} terminée", flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "semi_supervised_cv.csv", index=False)
    pd.DataFrame(audit).to_csv(OUTPUT / "pseudo_labels_audit.csv", index=False)
    result = pd.DataFrame(rows)
    summary = result.groupby(["fraction", "model"]).f1_macro.agg(["mean", "std"]).reset_index()
    summary.to_csv(OUTPUT / "semi_supervised_summary.csv", index=False)
    # Les essais partagent des données : écart-type descriptif.
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for method, label, color in [("foret", "Forêt supervisée", "#315e73"), ("self_training", "Self-training", "#ae483e")]:
        subset = summary[summary.model == method]
        ax.errorbar(subset.fraction * 100, subset["mean"], yerr=subset["std"], marker="o", capsize=4, label=label, color=color)
    ax.set(xlabel="Labels d'apprentissage conservés (%)", ylabel="F1 macro en validation", ylim=(0, 1.03))
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure("semi_supervised")
    return summary


def interpret(frame, splits, choices):
    rows = []
    for fold, (train_ids, valid_ids) in enumerate(splits):
        train = frame.loc[train_ids].index[frame.loc[train_ids, TARGET].notna()]
        valid = frame.loc[valid_ids].index[frame.loc[valid_ids, TARGET].notna()]
        threshold = float(frame.loc[train, TARGET].median())
        forest = make_model("foret", choices["foret"]["params"])
        forest.fit(frame.loc[train, INPUTS], observed_labels(frame.loc[train, TARGET], threshold))
        # Permuter une entrée et mesurer la baisse de F1 sur la validation.
        permutation = permutation_importance(forest, frame.loc[valid, INPUTS],
                                             observed_labels(frame.loc[valid, TARGET], threshold),
                                             scoring="f1_macro", n_repeats=5, random_state=42, n_jobs=1)
        for col, mean, std in zip(INPUTS, permutation.importances_mean, permutation.importances_std):
            rows.append({"fold": fold, "feature": col, "importance": mean, "repeat_std": std})
    pd.DataFrame(rows).to_csv(OUTPUT / "importance_folds.csv", index=False)
    summary = pd.DataFrame(rows).groupby("feature").importance.agg(["mean", "std"]).sort_values("mean", ascending=False)
    summary.to_csv(OUTPUT / "importance_summary.csv")

    dev = np.unique(np.concatenate([valid for _, valid in splits]))
    top = summary.head(3).index.tolist()
    associations = []
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
    for col, ax in zip(top, axes):
        observed = frame.loc[dev, [col, TARGET]].dropna()
        if col in NUMERIC and observed[col].nunique() > 4:
            bins = pd.qcut(observed[col], 4, duplicates="drop")
            trends = observed.groupby(bins, observed=False)[TARGET].agg(["median", "count"])
            trends.to_csv(OUTPUT / f"trend_{col}.csv")
            trends["median"].plot.bar(ax=ax, color="#497e9a")
            ax.tick_params(axis="x", labelsize=6, rotation=25)
            rho = observed[col].corr(observed[TARGET], method="spearman")
            associations.append({"feature": col, "spearman": float(rho), "n": len(observed)})
        else:
            observed.groupby(col)[TARGET].median().plot.bar(ax=ax, color="#497e9a")
            ax.tick_params(axis="x", labelsize=6, rotation=25)
            associations.append({"feature": col, "spearman": None, "n": len(observed)})
        ax.set(xlabel=col, ylabel="Traction médiane (MPa)")
    fig.tight_layout()
    save_figure("associations")
    save_json(OUTPUT / "associations.json", associations)
    return summary


def censoring_sensitivity(groups, splits, winner):
    """Comparer les conventions de censure sur les mêmes plis."""
    rows = []
    for policy in ["half", "upper", "zero"]:
        frame = load_data(censored=policy)
        for fold, (train_ids, valid_ids) in enumerate(splits):
            train = frame.loc[train_ids].index[frame.loc[train_ids, TARGET].notna()]
            valid = frame.loc[valid_ids].index[frame.loc[valid_ids, TARGET].notna()]
            threshold = float(frame.loc[train, TARGET].median())
            model = make_model(winner["model"], winner["params"])
            model.fit(frame.loc[train, INPUTS], observed_labels(frame.loc[train, TARGET], threshold))
            rows.append({"policy": policy, "fold": fold, **metrics(
                observed_labels(frame.loc[valid, TARGET], threshold), model.predict(frame.loc[valid, INPUTS]))})
    pd.DataFrame(rows).to_csv(OUTPUT / "censoring_sensitivity.csv", index=False)


def final_evaluation(frame, groups, dev, test, choices, winner):
    assert_separate(groups, dev, test)
    train = frame.loc[dev].index[frame.loc[dev, TARGET].notna()]
    valid = frame.loc[test].index[frame.loc[test, TARGET].notna()]
    threshold = float(frame.loc[train, TARGET].median())
    # Le seuil du test est celui du développement complet.
    truth = observed_labels(frame.loc[valid, TARGET], threshold)
    assert set(truth) == {0, 1}
    predictions = pd.DataFrame({"truth": truth}, index=valid)
    rows = []
    for name in ["majoritaire", winner["model"]]:
        model = make_model(name, choices[name]["params"])
        model.fit(frame.loc[train, INPUTS], observed_labels(frame.loc[train, TARGET], threshold))
        pred = model.predict(frame.loc[valid, INPUTS])
        predictions[name] = pred
        rows.append({"model": name, **metrics(truth, pred)})
        if name != "majoritaire":
            processed = model.named_steps["preprocess"].transform(frame.loc[valid, INPUTS])
            assert np.isfinite(processed).all()
    pair, detail, pair_predictions = fit_pair(frame, dev, test, choices["foret"]["params"], 1.0, 42)

    for name, score in pair:
        if name not in predictions:
            predictions[name] = pair_predictions[name]
            rows.append({"model": name, **score})
    predictions.to_csv(OUTPUT / "test_predictions.csv")
    pd.DataFrame(rows).to_csv(OUTPUT / "test_metrics.csv", index=False)
    prep = preprocessing().fit(frame.loc[train, INPUTS])
    features = prep.named_steps["transform"].get_feature_names_out()
    small_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    small_tree.fit(prep.transform(frame.loc[train, INPUTS]), observed_labels(frame.loc[train, TARGET], threshold))
    (OUTPUT / "tree_rules.txt").write_text(export_text(small_tree, feature_names=list(features)), encoding="utf-8")
    info = {"threshold": threshold, "development_rows": len(dev), "test_rows": len(test),
            "development_groups": int(groups.loc[dev].nunique()), "test_groups": int(groups.loc[test].nunique()),
            "development_labeled": len(train), "test_labeled": len(valid),
            "test_class_0": int((truth == 0).sum()), "test_class_1": int((truth == 1).sum()),
            "supervised_winner": winner["model"], "ssl": detail}
    save_json(OUTPUT / "test_summary.json", info)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    displayed = ["majoritaire", winner["model"], "self_training"]
    for ax, name in zip(axes, displayed):
        matrix = confusion_matrix(truth, predictions[name], labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set(title=name, xlabel="Classe prédite", ylabel="Classe observée")
    fig.tight_layout()
    save_figure("confusion")
    return pd.DataFrame(rows), info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Rejouer volontairement toute l'étude, sans changer la sélection.")
    args = parser.parse_args()
    marker = OUTPUT / "run_manifest.json"
    if marker.exists() and not args.overwrite:
        raise SystemExit("Une étude existe déjà. Utiliser --overwrite pour la reproduire explicitement.")
    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=.85)
    frame = load_data()
    assert frame.shape == (1652, 44)
    assert set(INPUTS).isdisjoint(NAMES[30:])
    groups = make_groups(frame)
    dev, test, splits = prepare_partitions(frame, groups)
    summary = describe(frame, groups, dev)
    pca = pca_analysis(frame, dev)
    choices, winner = supervised(frame, groups, splits)
    ssl = semi_supervised(frame, groups, splits, choices["foret"]["params"])
    importance = interpret(frame, splits, choices)
    censoring_sensitivity(groups, splits, winner)
    print("Choix figés. Évaluation finale sur les groupes de test.", flush=True)
    final, info = final_evaluation(frame, groups, dev, test, choices, winner)
    # Enregistrer les versions et la source utilisée pour cette exécution.
    manifest = {"completed_at": now(), "python": platform.python_version(), "seed": SEED,
                "cv_folds": len(splits), "confidence": .90, "max_ssl_iterations": 10,
                "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "reportlab"]},
                "data_sha256": hashlib.sha256((ROOT / "data/raw/welddb.tar").read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256((ROOT / "scripts/study.py").read_bytes()).hexdigest(),
                "protocol": "grouped_holdout_train_median_foldwise_cv_self_training"}
    save_json(marker, manifest)
    print(final[["model", "f1_macro", "accuracy"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
