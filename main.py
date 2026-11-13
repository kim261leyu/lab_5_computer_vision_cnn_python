import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")  # без вікон: графіки лише зберігаються у файли
import matplotlib.pyplot as plt

from src.dataset import MeterDataset
from src.cnn_model import MultiDigitCNN, CRNN
from src.trainer import train, train_crnn, ctc_decode

DATA_DIR = "data/meter_images"
REPORT_DIR = "report"
EPOCHS = 10
BATCH_SIZE = 64
SEED = 0


def model_size_kb(model):
    """Розмір параметрів моделі в КБ (кількість параметрів * байт на параметр)."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0


def collect_predictions(model, loader, is_crnn):
    """Повертає справжні та передбачені послідовності цифр для всього loader."""
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            if is_crnn:
                seqs = [ctc_decode(s.tolist()) for s in out.argmax(dim=2)]
            else:
                seqs = out.argmax(dim=2).tolist()
            trues += y.tolist()
            preds += seqs
    return trues, preds


def edit_distance(a, b):
    """Відстань Левенштейна між двома послідовностями."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def compute_metrics(trues, preds):
    """Матриця плутанини 10x10, F1 по кожному класу та CER."""
    cm = np.zeros((10, 10), dtype=int)
    cer_sum = 0.0
    for t, p in zip(trues, preds):
        for a, b in zip(t, p):      # порівняння позиція в позицію
            cm[a, b] += 1
        cer_sum += edit_distance(p, t) / len(t)

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(10), where=denom > 0)
    return cm, f1, cer_sum / len(trues)


def evaluate_model(name, model, trainer, is_crnn, train_loader, val_loader):
    res = trainer(model, train_loader, val_loader, epochs=EPOCHS)

    trues, preds = collect_predictions(model, val_loader, is_crnn)
    cm, f1, cer = compute_metrics(trues, preds)

    return {
        "name": name,
        "digit_acc": res["digit_acc"] * 100.0,
        "full_acc": res["full_acc"] * 100.0,
        "f1_macro": float(f1.mean()),
        "f1_per_class": f1,
        "cer": cer,
        "train_ms": res["train_ms_per_epoch"],
        "infer_ms": res["infer_ms"],
        "size_kb": model_size_kb(model),
        "params": sum(p.numel() for p in model.parameters()),
        "losses": res.get("losses", []),
        "cm": cm,
    }


def plot_losses(results, path):
    plt.figure(figsize=(7, 4))
    for r in results:
        if r["losses"]:
            plt.plot(range(1, len(r["losses"]) + 1), r["losses"], marker="o", label=r["name"])
    plt.yscale("log")
    plt.xlabel("Епоха")
    plt.ylabel("Втрати (loss), логарифмічна шкала")
    plt.title("Криві втрат на навчальній вибірці")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.log1p(cm), cmap="Blues")  # колір у логарифмічній шкалі, щоб помилки були видні
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Передбачена цифра")
    ax.set_ylabel("Справжня цифра")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                color = "white" if np.log1p(cm[i, j]) > np.log1p(cm.max()) / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=7)
    fig.colorbar(im, label="ln(1 + кількість)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_result(r):
    print(f"Модель: {r['name']}")
    print(f"Точність по цифрах (Accuracy): {r['digit_acc']:.2f}%")
    print(f"Точність по числах (exact match): {r['full_acc']:.2f}%")
    print(f"F1-Score (macro, 10 цифр): {r['f1_macro']:.4f}")
    print(f"CER: {r['cer']:.4f}")
    print(f"Час навчання: {r['train_ms']:.2f} мс/епоху")
    print(f"Час інференсу: {r['infer_ms']:.3f} мс/зображення")
    print(f"Параметрів: {r['params']}, розмір моделі: {r['size_kb']:.1f} КБ\n")


def main():
    print("=" * 90)
    print("  ЛАБОРАТОРНА РОБОТА №5: СВЕРТОЧНІ НЕЙРОННІ МЕРЕЖІ (PYTHON 3.10+ / PYTORCH)")
    print("  ВАРІАНТ 4: Оптичне розпізнавання лічильників та приладів (OCR)")
    print("=" * 90 + "\n")

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Один і той самий поділ даних для обох моделей
    ds = MeterDataset(DATA_DIR)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = random_split(
        ds, [n_train, len(ds) - n_train],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f"Зображень: {len(ds)} (навчання: {len(train_ds)}, валідація: {len(val_ds)})\n")

    results = []

    torch.manual_seed(SEED)
    results.append(evaluate_model("CNN Classifier", MultiDigitCNN(), train, False,
                                  train_loader, val_loader))
    print_result(results[-1])

    torch.manual_seed(SEED)
    results.append(evaluate_model("CRNN (CTC)", CRNN(), train_crnn, True,
                                  train_loader, val_loader))
    print_result(results[-1])

    # Порівняльна таблиця
    table = pd.DataFrame([{
        "Архітектура": r["name"],
        "Accuracy цифр (%)": round(r["digit_acc"], 2),
        "Accuracy чисел (%)": round(r["full_acc"], 2),
        "F1 (macro)": round(r["f1_macro"], 4),
        "CER": round(r["cer"], 4),
        "Час навчання (мс/епоха)": round(r["train_ms"], 1),
        "Інференс (мс/зобр.)": round(r["infer_ms"], 3),
        "Розмір моделі (КБ)": round(r["size_kb"], 1),
    } for r in results])
    print(table.to_string(index=False))
    table.to_csv(os.path.join(REPORT_DIR, "results.csv"), index=False)

    f1_table = pd.DataFrame({r["name"]: r["f1_per_class"].round(4) for r in results})
    f1_table.index.name = "цифра"
    print("\nF1 по кожній цифрі:")
    print(f1_table.to_string())

    # Графіки для README
    plot_losses(results, os.path.join(REPORT_DIR, "loss_curves.png"))
    plot_confusion(results[0]["cm"], "Матриця плутанини: CNN Classifier",
                   os.path.join(REPORT_DIR, "confusion_cnn.png"))
    plot_confusion(results[1]["cm"], "Матриця плутанини: CRNN (CTC)",
                   os.path.join(REPORT_DIR, "confusion_crnn.png"))

    print("[УСПІШНО] Навчання та тестування CNN і CRNN завершені.")


if __name__ == "__main__":
    main()