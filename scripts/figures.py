"""Figures de synthèse."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ROOT, INPUTS, TARGET, load_data

SHORT = {"current": "Intensité", "voltage": "Tension", "heat_input": "Apport thermique",
         "interpass_temperature": "T interpasses", "pwht_temperature": "T traitement",
         "pwht_time": "Durée traitement", "weld_type": "Procédé", "current_type": "AC/DC",
         "polarity": "Polarité"}


def refresh_figures():
    out = ROOT / "output"
    figures = out / "figures"
    frame = load_data()
    # Refaire les figures à partir des résultats enregistrés.
    missing = pd.read_csv(out / "missing.csv", index_col=0).missing_fraction[INPUTS].sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.25), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].bar(np.arange(len(missing)), missing*100, color="#497e9a")
    axes[0].set_xticks(np.arange(len(missing)), [SHORT.get(c, c) for c in missing.index], rotation=65, ha="right", fontsize=7)
    axes[0].axhline(60, color="#ae483e", ls="--", lw=1)
    axes[0].set_ylabel("Valeurs absentes (%)")
    axes[1].hist(frame[TARGET].dropna(), bins=25, color="#497e9a", edgecolor="white")
    axes[1].set(xlabel="Résistance à la traction (MPa)", ylabel="Nombre d'observations")
    fig.tight_layout()
    fig.savefig(figures / "data_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    # Les matrices utilisent les prédictions du test, sans réentraîner les modèles.
    predictions = pd.read_csv(out / "test_predictions.csv", index_col=0)
    import json
    winner = json.loads((out / "test_summary.json").read_text(encoding="utf-8"))["supervised_winner"]
    labels_display = {"majoritaire": "Classe majoritaire", "knn": "k-NN", "arbre": "Arbre CART",
                      "foret": "Forêt aléatoire", "self_training": "Self-training"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for name, ax in zip(["majoritaire", winner, "self_training"], axes):
        matrix = confusion_matrix(predictions.truth, predictions[name], labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    vmin=0, vmax=max((predictions.truth == 0).sum(), (predictions.truth == 1).sum()))
        ax.set(title=labels_display[name], xlabel="Classe prédite", ylabel="Classe observée")
    fig.tight_layout()
    fig.savefig(figures / "confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    scores = pd.read_csv(out / "pca_scores.csv", index_col=0)
    loadings = pd.read_csv(out / "pca_loadings.csv", index_col=0)


    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    # Reprendre la préparation utilisée pour l'ACP.
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(
        frame.loc[scores.index, loadings.index]))
    pca = PCA().fit(Z)
    threshold = float(frame.loc[scores.index, TARGET].median())
    from .study import observed_labels
    labels = observed_labels(frame.loc[scores.index, TARGET], threshold)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1, 1.1, 1.3]})
    axes[0].plot(np.arange(1, Z.shape[1]+1), pca.explained_variance_ratio_.cumsum()*100,
                 marker=".", color="#315e73")
    axes[0].axhline(80, ls="--", color="#ae483e", lw=1)
    axes[0].set(xlabel="Composantes", ylabel="Variance cumulée (%)", ylim=(0, 103))
    for label, color, title in [(-1, "#aaa", "Sans mesure"), (0, "#ae483e", "Faible"), (1, "#315e73", "Élevée")]:
        mask = labels == label
        axes[1].scatter(scores.PC1[mask], scores.PC2[mask], s=9, alpha=.5, c=color, label=title)
    axes[1].set(xlabel="PC1", ylabel="PC2")
    axes[1].legend(fontsize=7)
    # Montrer les huit corrélations les plus fortes.
    selected = loadings.loc[(loadings**2).sum(axis=1).nlargest(8).index].iloc[::-1]
    positions = np.arange(len(selected))
    axes[2].barh(positions-.16, selected.PC1, height=.30, label="PC1", color="#315e73")
    axes[2].barh(positions+.16, selected.PC2, height=.30, label="PC2", color="#ae483e")
    axes[2].set_yticks(positions, [SHORT.get(c, c) for c in selected.index], fontsize=8)
    axes[2].axvline(0, color="#777", lw=.8)
    axes[2].set(xlabel="Corrélation avec les axes", xlim=(-1.05, 1.05))
    axes[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    refresh_figures()
