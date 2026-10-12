# Projet soudures

On prédit un niveau de résistance à la traction avec k-NN, un arbre et une forêt
aléatoire. On teste aussi le self-training avec les données sans label.
La cible reste un indicateur de résistance, pas une mesure complète de qualité.

## Lancer le projet

Python 3.12, avec uv. Depuis ce dossier :

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'tmp/matplotlib'
uv sync --locked
uv run python -m scripts.data --download
uv run python -m scripts.study --overwrite
uv run python -m scripts.figures
uv run python -m scripts.verify
```

Les données sont déjà dans `data/raw`. `--overwrite` permet de rejouer l'étude.
Le rapport s'édite directement dans `rapport_soudures.tex`. Les noms et la date
sont en haut du fichier. Les images sont incluses depuis
`output/figures` avec `\includegraphics`.

## Fichiers générés

`output` contient les fichiers générés. Le `.tex` à la racine est édité à la main.

| Script | Sorties |
| --- | --- |
| `study.py` | statistiques, ACP, modèles, scores, prédictions et figures |
| `verify.py` | `verification.json` |
| Compilateur LaTeX | `output/pdf/rapport_soudures.pdf` et son log |
| `reproduce_check.py` | `reproduction_check.json`, après comparaison de deux exécutions |
| `figures.py` | version de synthèse de certains graphiques |

Les CSV et figures se régénèrent en relançant les scripts d'analyse.

## Quelques choix

- `N` signifie manquant ; les zéros sont conservés. Les valeurs `<x` deviennent `x/2`.
- La classe élevée correspond à une traction supérieure ou égale à la médiane de l'apprentissage.
- Les essais liés restent dans la même partition. Test séparé et CV à cinq plis.
- Imputation et normalisation sont apprises sur l'apprentissage seulement.
- Self-training : confiance >0,90 et dix itérations maximum.

## Vérifier

```powershell
uv run python -m pytest -q
```

Pour comparer deux exécutions :

```powershell
uv run python -m scripts.reproduce_check --snapshot tmp/reference.json
uv run python -m scripts.study --overwrite
uv run python -m scripts.reproduce_check --reference tmp/reference.json
```

Les références bibliographiques sont dans le rapport.
