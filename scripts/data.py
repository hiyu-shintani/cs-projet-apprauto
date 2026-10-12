"""Lecture et préparation des données."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://www.phase-trans.msm.cam.ac.uk/map/data/materials/welddb-b.html"
ARCHIVE_URL = "https://www.phase-trans.msm.cam.ac.uk/map/data/tar/welddb.tar"
NAMES = [
    "C", "Si", "Mn", "S", "P", "Ni", "Cr", "Mo", "V", "Cu", "Co", "W",
    "O", "Ti", "N", "Al", "B", "Nb", "Sn", "As", "Sb", "current", "voltage",
    "current_type", "polarity", "heat_input", "interpass_temperature", "weld_type",
    "pwht_temperature", "pwht_time", "yield_strength", "tensile_strength",
    "elongation", "reduction_area", "charpy_temperature", "charpy_toughness",
    "hardness", "fatt_50", "primary_ferrite", "ferrite_second_phase",
    "acicular_ferrite", "martensite", "ferrite_carbide", "weld_id",
]
INPUTS = NAMES[:30]
CATEGORICAL = ["current_type", "polarity", "weld_type"]
NUMERIC = [n for n in INPUTS if n not in CATEGORICAL]
TARGET = "tensile_strength"


def now() -> str:
    return datetime.now(ZoneInfo("Europe/Paris")).isoformat(timespec="seconds")


def download() -> dict:
    """Télécharger les données si elles ne sont pas déjà présentes."""
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "welddb.tar"
    if not archive.exists():
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
            payload = response.read()

        with tarfile.open(fileobj=io.BytesIO(payload)) as source:
            assert {"welddb.data", "welddb.info", "welddb.tex"}.issubset(source.getnames())
        archive.write_bytes(payload)
    # Garder une copie brute et vérifier son empreinte.
    payload = archive.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(payload)) as source:
        for name in ["welddb.data", "welddb.info", "welddb.tex"]:
            member = source.extractfile(name)
            if member is None:
                raise ValueError(f"Fichier absent : {name}")
            (raw / name).write_bytes(member.read())
    digest = hashlib.sha256(payload).hexdigest()
    manifest = raw / "provenance.json"
    if manifest.exists():
        provenance = json.loads(manifest.read_text(encoding="utf-8"))
        if provenance["sha256"] != digest:
            raise ValueError("L'archive a changé ; vérifier explicitement la provenance.")
    else:
        provenance = {"source_url": SOURCE_URL, "archive_url": ARCHIVE_URL,
                      "retrieved_at": now(), "sha256": digest, "bytes": len(payload)}
        manifest.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance


def parse_numeric(value, censored="half"):
    """Lire les valeurs numériques et les formats spéciaux."""
    if pd.isna(value):
        return np.nan, "missing"
    value = str(value)
    # Une limite de détection est approximée par la moitié de la borne.
    if value.startswith("<"):
        limit = float(value[1:])
        factors = {"half": .5, "upper": 1., "zero": 0.}
        return limit * factors[censored], f"censored_{censored}"
    interval = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", value)
    if interval:
        return (float(interval[1]) + float(interval[2])) / 2, "interval_midpoint"
    nitrogen = re.fullmatch(r"(\d+(?:\.\d+)?)tot(?:\d+(?:\.\d+)?|nd)res", value)
    if nitrogen:
        return float(nitrogen[1]), "nitrogen_total_residual_not_used"
    hardness = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\(Hv\d+\)|Hv\d+)", value)
    if hardness:
        return float(hardness[1]), "hardness_annotation_retained_in_audit_excluded_from_models"
    return float(value), "numeric"


def load_data(censored="half") -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "welddb.data"
    widths = [len(line.split()) for line in path.read_text().splitlines() if line.strip()]
    if not widths or set(widths) != {44}:
        raise ValueError("Chaque ligne doit contenir exactement 44 colonnes.")
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=NAMES, na_values=["N"],
                        keep_default_na=False, dtype=str)
    # Conserver les conversions pour pouvoir les retrouver.
    audit = []
    for name in NAMES:
        if name not in CATEGORICAL + ["weld_id"]:
            parsed = []
            for row, value in frame[name].items():
                numeric, action = parse_numeric(value, censored)
                parsed.append(numeric)
                if action not in ["numeric", "missing"]:
                    audit.append({"row_id": row, "feature": name, "original": value,
                                  "converted": numeric, "action": action})
            frame[name] = parsed
    frame.index.name = "row_id"
    frame.attrs["conversion_audit"] = audit
    return frame


def canonical_id(identifier: str) -> str:
    """Retirer les suffixes aw, ht, ch et mic pour regrouper les essais liés."""
    prefix, separator, sample = identifier.rpartition("-")
    if not separator:
        return identifier

    base = re.sub(r"(?:aw|ht|mic|ch)(?:\d+)?(?:[-+]\d+(?:\.\d+)?)?$", "", identifier,
                  flags=re.IGNORECASE).rstrip("-")

    prefix2, separator2, sample2 = base.rpartition("-")
    if separator2:
        sample2 = re.sub(r"(?:aw|ht|mic|ch)\d*$", "", sample2, flags=re.IGNORECASE)
        base = prefix2 + "-" + sample2
    return base.rstrip("-") or identifier


def make_groups(frame: pd.DataFrame) -> pd.Series:
    """Regrouper les essais liés ou les entrées identiques."""
    parent = list(range(len(frame)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        parent[find(b)] = find(a)

    # Un même essai peut être lié par son identifiant ou par ses paramètres.
    seen_ids, seen_inputs = {}, {}
    for i, (_, row) in enumerate(frame.iterrows()):
        key_id = canonical_id(str(row["weld_id"]))
        key_x = tuple(None if pd.isna(v) else v for v in row[INPUTS])
        if key_id in seen_ids:
            union(i, seen_ids[key_id])
        if key_x in seen_inputs:
            union(i, seen_inputs[key_x])
        seen_ids[key_id], seen_inputs[key_x] = i, i
    labels = [find(i) for i in range(len(frame))]
    return pd.Series(pd.factorize(np.asarray(labels))[0], index=frame.index, name="group_id")


def dictionary() -> pd.DataFrame:
    descriptions = [
        "Carbone", "Silicium", "Manganèse", "Soufre", "Phosphore", "Nickel", "Chrome",
        "Molybdène", "Vanadium", "Cuivre", "Cobalt", "Tungstène", "Oxygène", "Titane",
        "Azote", "Aluminium", "Bore", "Niobium", "Étain", "Arsenic", "Antimoine",
        "Intensité", "Tension", "Courant AC/DC", "Polarité de l'électrode",
        "Apport thermique", "Température entre passes", "Procédé de soudage",
        "Température de traitement après soudage", "Durée de traitement après soudage",
        "Limite d'élasticité", "Résistance à la traction", "Allongement à rupture",
        "Réduction de section", "Température de l'essai Charpy", "Énergie absorbée Charpy",
        "Dureté", "Température de transition 50 % FATT", "Ferrite primaire",
        "Ferrite avec seconde phase", "Ferrite aciculaire", "Martensite",
        "Ferrite avec agrégats de carbures", "Identifiant de soudure",
    ]
    units = (["% massique"] * 12 + ["ppm massique"] * 9 +
             ["A", "V", "catégorie", "catégorie", "kJ/mm", "°C", "catégorie", "°C", "h",
              "MPa", "MPa", "%", "%", "°C", "J", "kg/mm²", "non précisée (température)"] +
             ["%"] * 5 + ["identifiant"])
    roles = ["entrée" if n in INPUTS else "cible" if n == TARGET else
             "groupement" if n == "weld_id" else "exclue des entrées" for n in NAMES]
    return pd.DataFrame({"column": range(1, 45), "name": NAMES, "description": descriptions,
                         "unit": units, "role": roles})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        print(json.dumps(download(), ensure_ascii=False, indent=2))
    else:
        print(load_data().shape)
