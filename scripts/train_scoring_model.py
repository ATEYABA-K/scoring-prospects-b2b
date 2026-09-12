"""
Scoring de propension à la souscription — cas d'usage IA pour la priorisation
d'un effort commercial (analogue à un scoring de leads B2B/Pro-PME).

Dataset : UCI "Bank Marketing" (bank-additional-full.csv), campagnes
télémarketing d'une banque portugaise, cible = souscription à un dépôt à terme.
Choisi car sa structure (contacts commerciaux + résultat converti/non converti)
est directement transposable à un scoring de prospects B2B.

Ce script : charge les données, retire la fuite de données connue de ce
dataset (voir README), entraîne 2 modèles, compare leurs performances,
produit une analyse par décile (lift) et une analyse d'importance des
variables au niveau métier (pas au niveau des colonnes one-hot).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_auc_score, average_precision_score
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


def evaluate(pipe: Pipeline, X_test, y_test, name: str) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "name": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "proba": proba,
    }


def plot_curves(results: list[dict], y_test, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for r in results:
        RocCurveDisplay.from_predictions(y_test, r["proba"], name=r["name"], ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, r["proba"], name=r["name"], ax=axes[1])
    axes[0].set_title("Courbe ROC")
    axes[1].set_title("Courbe précision-rappel\n(taux de base = 11 % de conversions)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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


def plot_feature_importance(pipe: Pipeline, X_test, y_test, out_path: Path) -> None:
    # Permutation importance calculée sur les colonnes ORIGINALES (avant
    # encodage) : on permute "job", pas "job_admin."/"job_blue-collar"/...
    # séparément — sinon l'importance d'une variable catégorielle se
    # retrouve artificiellement diluée entre ses modalités.
    result = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                     n_repeats=10, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=True)
    importances = importances[importances > 0].tail(12)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index, importances.values, color="#54A24B")
    ax.set_xlabel("Baisse du ROC-AUC quand la variable est mélangée aléatoirement")
    ax.set_title("Quelles variables pèsent le plus dans le score ?")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    models = {
        "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    }

    results = []
    best_pipe, best_name, best_auc = None, None, -1
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        res = evaluate(pipe, X_test, y_test, name)
        results.append(res)
        print(f"{name:25s} ROC-AUC={res['roc_auc']:.3f}  PR-AUC={res['pr_auc']:.3f}")
        if res["roc_auc"] > best_auc:
            best_pipe, best_name, best_auc = pipe, name, res["roc_auc"]

    plot_curves(results, y_test, ROOT / "chart_courbes_roc_pr.png")

    best_proba = [r["proba"] for r in results if r["name"] == best_name][0]
    lift_summary = decile_lift_chart(best_proba, y_test, ROOT / "chart_lift_deciles.png")
    print(f"\nMeilleur modèle : {best_name}")
    print(lift_summary.to_string(index=False))

    plot_feature_importance(best_pipe, X_test, y_test, ROOT / "chart_importance_variables.png")

    lift_summary.to_csv(ROOT / "data" / "resultats_deciles.csv", index=False)
    print("\nTerminé — graphiques et résultats écrits à la racine du projet.")


if __name__ == "__main__":
    main()
