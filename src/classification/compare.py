"""
Comparaison finale des 3 approches de classification.

Génère :
  1. Tableau comparatif (console + texte)
  2. Graphique de comparaison (barres groupées)
  3. Synthèse JSON pour le mémoire

Usage :
    python -m src.classification.compare
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = ROOT / "outputs" / "classification"
OUTPUT = OUTPUT_BASE / "comparison"


def load_results():
    """Charger les résultats des 3 approches."""
    results = {}

    # SVM
    svm_report = OUTPUT_BASE / "svm" / "classification_report.txt"
    if svm_report.exists():
        text = svm_report.read_text(encoding="utf-8")
        results["SVM (Features manuelles)"] = _parse_report(text)

    # CNN
    cnn_json = OUTPUT_BASE / "cnn" / "results.json"
    if cnn_json.exists():
        with open(cnn_json) as f:
            data = json.load(f)
        results["CNN (EfficientNet-B0)"] = {
            "accuracy": data["accuracy"],
            "f1_macro": data["f1_macro"],
        }

    # Hybrid
    hybrid_json = OUTPUT_BASE / "hybrid" / "results.json"
    if hybrid_json.exists():
        with open(hybrid_json) as f:
            data = json.load(f)
        results["Hybride (CNN+SVM)"] = {
            "accuracy": data["accuracy"],
            "f1_macro": data["f1_macro"],
        }

    return results


def _parse_report(text):
    """Extraire accuracy et F1 depuis le rapport texte."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Accuracy"):
            parts = line.split(":")
            if len(parts) == 2:
                result["accuracy"] = float(parts[1].strip())
        if line.startswith("F1-macro"):
            parts = line.split(":")
            if len(parts) == 2:
                result["f1_macro"] = float(parts[1].strip())
    return result


def compare():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Comparaison — 3 Approches de Classification")
    print("=" * 70)

    results = load_results()

    if not results:
        print("\n  [ERREUR] Aucun résultat trouvé.")
        print("  Exécutez d'abord les 3 approches :")
        print("    python -m src.classification.train_svm")
        print("    python -m src.classification.train_cnn")
        print("    python -m src.classification.train_hybrid")
        return

    # ── Tableau console ───────────────────────────────────────────────
    print(f"\n{'Approche':<30} | {'Accuracy':>10} | {'F1-macro':>10}")
    print("-" * 58)
    for name, metrics in results.items():
        acc = metrics.get("accuracy", "—")
        f1 = metrics.get("f1_macro", "—")
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else acc
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
        print(f"{name:<30} | {acc_str:>10} | {f1_str:>10}")
    print("-" * 58)

    # ── Graphique barres groupées ─────────────────────────────────────
    print("\n[INFO] Génération du graphique comparatif...")

    approaches = list(results.keys())
    accuracies = [results[a].get("accuracy", 0) for a in approaches]
    f1_scores = [results[a].get("f1_macro", 0) for a in approaches]

    x = np.arange(len(approaches))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy",
                   color="#3498db", edgecolor="white")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1-macro",
                   color="#e74c3c", edgecolor="white")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparaison des approches de classification — Qualité de combustion",
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Annoter les barres
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT / "comparison_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Synthèse JSON ─────────────────────────────────────────────────
    summary = {
        "task": "Classification qualité de combustion",
        "classes": ["bonne", "moyenne", "mauvaise"],
        "results": results,
    }
    with open(OUTPUT / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Rapport texte ─────────────────────────────────────────────────
    with open(OUTPUT / "comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("Comparaison des approches de classification\n")
        f.write("=" * 58 + "\n\n")
        f.write(f"{'Approche':<30} | {'Accuracy':>10} | {'F1-macro':>10}\n")
        f.write("-" * 58 + "\n")
        for name, metrics in results.items():
            acc = metrics.get("accuracy", 0)
            f1m = metrics.get("f1_macro", 0)
            f.write(f"{name:<30} | {acc:>10.4f} | {f1m:>10.4f}\n")
        f.write("-" * 58 + "\n")

    print(f"\n  Graphique : {OUTPUT / 'comparison_chart.png'}")
    print(f"  JSON      : {OUTPUT / 'comparison_summary.json'}")
    print(f"  Rapport   : {OUTPUT / 'comparison_report.txt'}")

    print("\n" + "=" * 70)
    print("  [DONE] Comparaison terminée")
    print("=" * 70)

    # Identifier le meilleur
    if results:
        best = max(results.items(), key=lambda x: x[1].get("f1_macro", 0))
        print(f"\n  🏆 Meilleure approche : {best[0]}")
        print(f"     F1-macro = {best[1].get('f1_macro', 0):.4f}")


if __name__ == "__main__":
    compare()
