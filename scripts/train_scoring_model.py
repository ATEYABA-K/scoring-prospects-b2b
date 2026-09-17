"""
Scoring de propension à la souscription — cas d'usage IA pour la priorisation
d'un effort commercial (analogue à un scoring de leads B2B/Pro-PME).

Dataset : UCI "Bank Marketing" (bank-additional-full.csv), campagnes
télémarketing d'une banque portugaise, cible = souscription à un dépôt à terme.
Choisi car sa structure (contacts commerciaux + résultat converti/non converti)
est directement transposable à un scoring de prospects B2B.

Ce script : charge les données, retire la fuite de données connue de ce
dataset (voir README), entraîne un modèle de classification, puis regarde
si le score qu'il donne à chaque prospect concentre vraiment les bonnes
conversions (analyse par décile).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "bank-additional-full.csv"

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
NUMERIC = ["age", "campaign", "pdays", "previous", "emp.var.rate",
           "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_FILE, sep=";")

    # "duration" (durée de l'appel) n'est connue qu'APRES avoir passé l'appel :
    # l'utiliser reviendrait à predire le résultat avec une info qui n'existe
    # pas encore au moment où on voudrait prioriser un prospect. Fuite de
    # données classique sur ce dataset — on l'exclut volontairement.
    df = df.drop(columns=["duration"])

    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])
    return X, y


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("prep", preprocessor), ("model", model)])


def decile_lift_chart(proba: np.ndarray, y_test: pd.Series, out_path: Path) -> pd.DataFrame:
    df = pd.DataFrame({"proba": proba, "y": y_test.values})
    df["decile"] = pd.qcut(df["proba"], 10, labels=False, duplicates="drop")
    df["decile"] = 9 - df["decile"]  # decile 0 = score le plus haut
    summary = df.groupby("decile").agg(taux_conversion=("y", "mean"), n=("y", "size")).reset_index()
    baseline = df["y"].mean()
    summary["lift"] = summary["taux_conversion"] / baseline

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["decile"], summary["taux_conversion"] * 100, color="#4C78A8")
    ax.axhline(baseline * 100, color="#B00020", linestyle="--", label=f"Taux moyen ({baseline*100:.1f} %)")
    ax.set_xlabel("Décile de score (0 = prospects les mieux notés)")
    ax.set_ylabel("Taux de conversion réel (%)")
    ax.set_title("Le score concentre-t-il vraiment les bons prospects ?\n(décile par décile, sur données jamais vues à l'entraînement)")
    ax.set_xticks(range(10))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return summary


def main() -> None:
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    pipe = build_pipeline(HistGradientBoostingClassifier(random_state=42))
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    lift_summary = decile_lift_chart(proba, y_test, ROOT / "chart_lift_deciles.png")
    print(lift_summary.to_string(index=False))

    lift_summary.to_csv(ROOT / "data" / "resultats_deciles.csv", index=False)
    print("\nTerminé — graphique et résultats écrits à la racine du projet.")


if __name__ == "__main__":
    main()
