"""
Configurable classic sentiment classifier (CN/EN) with TF-IDF + SVM/LR/RandomForest.
Trains on labeled data, optional validation, predicts test set,
evaluates with labels (if present), and writes submission files. Chinese uses only char-level TF-IDF (word removed).

Usage:
    python run_all.py --team YourTeam --run 1 [--algo lr] [--val_ratio 0.0]

Outputs:
    YourTeam_1_CN.txt, YourTeam_1_EN.txt (test predictions in required format)

Dependencies:
    pip install scikit-learn jieba numpy
"""
import os
import re
import random
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt

try:
    import jieba  # type: ignore
    JIEBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    JIEBA_AVAILABLE = False

# ---------------------------- Data Loading ----------------------------

def read_reviews(file_path: str, label: int) -> List[Tuple[str, int]]:
    """Parse review blocks and attach a label (1=pos, 0=neg)."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = re.findall(r"<review id=\"\d+\">(.*?)</review>", text, flags=re.S)
    cleaned = []
    for b in blocks:
        t = b.replace("\r", "\n")
        t = re.sub(r"\n+", " ", t).strip()
        if t:
            cleaned.append((t, label))
    return cleaned


def load_language_set(root: str, lang: str) -> List[Tuple[str, int]]:
    """Load one language's pos/neg sets and shuffle."""
    pos_path = os.path.join(root, f"{lang}_sample_data", "sample.positive.txt")
    neg_path = os.path.join(root, f"{lang}_sample_data", "sample.negative.txt")
    data = []
    data.extend(read_reviews(pos_path, 1))
    data.extend(read_reviews(neg_path, 0))
    random.shuffle(data)
    return data


# ---------------------------- Tokenization ----------------------------

def tokenize_for_tfidf(text: str, lang: str) -> List[str]:
    """Return a list of tokens for TF-IDF. CN uses jieba if available; fallback char-level."""
    if lang == "cn":
        if JIEBA_AVAILABLE:
            tokens = jieba.lcut(text)
        else:
            tokens = list(text)
    else:
        tokens = re.findall(r"[A-Za-z']+|\d+", text.lower())
    return tokens


# ---------------------------- Model ----------------------------

def build_pipeline(lang: str, algo: str = "svm", cn_max_df: float = 0.995, en_max_df: float = 0.98,
                   C: float = 3.0, lr_max_iter: int = 5000,
                   rf_n_estimators: int = 300, rf_max_depth: int = 30):
    """Build a Pipeline = TF-IDF + Classifier.

    - CN analyzer: char 1-2gram（仅保留）
    - EN: word 1-2gram
    - algo: svm | lr | rf
    """
    if lang == "cn":
        tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 2),
            min_df=2,
            max_df=cn_max_df,
            sublinear_tf=True,
        )
    else:
        tfidf = TfidfVectorizer(
            tokenizer=lambda x: tokenize_for_tfidf(x, lang),
            ngram_range=(1, 3),
            min_df=2,
            max_df=en_max_df,
            sublinear_tf=True,
            token_pattern=None,
        )

    if algo == "svm":
        clf = LinearSVC(C=2.5)
    elif algo == "lr":
        clf = LogisticRegression(max_iter=lr_max_iter, C=C, class_weight="balanced", solver="liblinear")
    else:  # rf
        clf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, n_jobs=-1, random_state=42)

    return Pipeline([( "tfidf", tfidf ), ( "clf", clf )])


def eval_and_print(model, texts, labels, lang_name: str, tag: str):
    """Evaluate on hold-out set and print metrics."""
    preds = model.predict(texts)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"[{lang_name}] {tag} Accuracy: {acc:.4f}  F1: {f1:.4f}")
    print(classification_report(labels, preds, digits=4))


# ---------------------------- Test Inference ----------------------------

def read_reviews_with_id(path: str) -> List[Tuple[int, str]]:
    """Parse reviews and keep their ID for submission output."""
    text = open(path, encoding="utf-8").read()
    blocks = re.findall(r'<review id="(\d+)">(.*?)</review>', text, flags=re.S)
    cleaned = []
    for rid, content in blocks:
        t = re.sub(r"\s+", " ", content).strip()
        if t:
            cleaned.append((int(rid), t))
    return cleaned


def read_labeled_reviews(path: str) -> List[Tuple[int, str, int]]:
    """Parse reviews with gold labels (label attr), robust to encoding."""
    # Try utf-8, fallback to gb18030, then latin-1
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            text = open(path, encoding=enc).read()
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover
        raise UnicodeDecodeError("Could not decode file", path, 0, 0, "unknown encoding")

    blocks = re.findall(r'<review id="(\d+)"\s+label="(\d+)">(.*?)</review>', text, flags=re.S)
    cleaned = []
    for rid, lab, content in blocks:
        t = re.sub(r"\s+", " ", content).strip()
        if t:
            cleaned.append((int(rid), t, int(lab)))
    return cleaned


def write_submission(out_path: str, team: str, run_tag: str, pairs: List[Tuple[int, str]], preds):
    with open(out_path, "w", encoding="utf-8") as f:
        for (rid, _), p in zip(pairs, preds):
            polarity = "positive" if p == 1 else "negative"
            f.write(f"{team} {run_tag} {rid} {polarity}\n")


def sweep_param_and_plot(param_name: str, lang: str, lang_name: str,
                         texts_train: List[str], labels_train: List[int],
                         texts_val: List[str], labels_val: List[int],
                         grid_values: List[float], cn_max_df: float, en_max_df: float,
                         out_dir: str, fixed_C: float, fixed_max_iter: int) -> Dict[str, List[float]]:
    """Generic LR sweep over a single parameter (C or max_df), plot Accuracy/F1.

    param_name: 'C' or 'max_df'
    grid_values: values to sweep over
    fixed_C: the non-swept C value to keep constant
    """
    accs: List[float] = []
    f1s: List[float] = []
    for val in grid_values:
        C_val = val if param_name == 'C' else fixed_C
        # Adjust max_df for the language when sweeping max_df
        cn_df = cn_max_df
        en_df = en_max_df
        if param_name == 'max_df':
            if lang == 'cn':
                cn_df = float(val)
            else:
                en_df = float(val)
        pipe = build_pipeline(lang, "lr", cn_df, en_df, C=C_val, lr_max_iter=fixed_max_iter)
        pipe.fit(texts_train, labels_train)
        preds = pipe.predict(texts_val)
        accs.append(accuracy_score(labels_val, preds))
        f1s.append(f1_score(labels_val, preds))

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(grid_values, accs, marker='o', label='Accuracy')
    plt.plot(grid_values, f1s, marker='s', label='F1')
    if param_name == 'C':
        plt.xscale('log')
        xlabel = 'C (log scale)'
    else:
        xlabel = 'max_df'
    plt.xlabel(xlabel)
    plt.ylabel('Score')
    plt.title(f'LogisticRegression {param_name} sweep - {lang_name}')
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    # Mark best F1 point
    try:
        best_idx = int(np.argmax(f1s))
        best_x = grid_values[best_idx]
        best_f1 = f1s[best_idx]
        plt.scatter([best_x], [best_f1], color='red', zorder=5, label='Best F1')
        plt.annotate(f"Best F1={best_f1:.4f}\n{param_name}={best_x}",
                     xy=(best_x, best_f1), xytext=(0, 10), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, color='red')
    except Exception:
        pass
    out_path = os.path.join(out_dir, f"results_{param_name}_sweep_{lang}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {out_path}")
    return {param_name: grid_values, "accuracy": accs, "f1": f1s}


# ---------------------------- Main ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join("训练集", "evaltask2_sample_data"), help="Root to evaltask2_sample_data")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples per language (0 = all)")
    parser.add_argument("--test_root", default=os.path.join("测试集", "Test"), help="Root to test data")
    parser.add_argument("--team", required=True, help="TeamName (no spaces)")
    parser.add_argument("--run", default="1", help="RunTag (1 or 2)")
    parser.add_argument("--label_root", default=os.path.join("测试集标注", "Sentiment Classification with Deep Learning"), help="Root to test labels (optional)")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Hold-out ratio for validation (0 = train on all)")
    parser.add_argument("--algo", choices=["svm","lr","rf"], default="lr")
    parser.add_argument("--cn_max_df", type=float, default=0.99)
    parser.add_argument("--en_max_df", type=float, default=0.995)
    parser.add_argument("--C", type=float, default=3.5, help="Regularization C for LR/SVM")
    parser.add_argument("--rf_n_estimators", type=int, default=600, help="Number of trees for RandomForest")
    parser.add_argument("--rf_max_depth", type=int, default=-1, help="Max depth for RandomForest (-1 for None)")
    parser.add_argument("--sweep", action="store_true", help="Sweep LogisticRegression over C values and plot")
    parser.add_argument("--c_grid", type=str, default="0.25,0.5,1,2,4,8,16", help="Comma-separated C values for LR sweep")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to save sweep plots")
    parser.add_argument("--sweep_param", type=str, choices=["C","max_df"], default="C", help="Choose parameter to sweep for LR (C or max_df)")
    parser.add_argument("--df_grid", type=str, default="0.95,0.97,0.98,0.99,0.995", help="Comma-separated max_df values for TF-IDF sweep")
    args = parser.parse_args()

    languages = ["cn", "en"]
    lang_names = {"cn": "Chinese", "en": "English"}
    suffixes = {"cn": "CN", "en": "EN"}

    for lang in languages:
        print("\n" + "=" * 60)
        print(f"Processing language: {lang_names[lang]} ({lang})")
        data = load_language_set(args.data_root, lang)
        if args.max_samples > 0:
            data = data[: args.max_samples]
        texts, labels = zip(*data)
        if args.val_ratio > 0:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=args.val_ratio, random_state=42, stratify=labels
            )
        else:
            train_texts, train_labels = list(texts), list(labels)
            val_texts, val_labels = [], []

        # If sweep requested, require validation split or use test labels
        if args.sweep and args.algo == "lr":
            if not val_texts:
                # Try using test labels for evaluation
                label_path = os.path.join(args.label_root, f"test.label.{lang}.txt")
                test_path = os.path.join(args.test_root, f"test.{lang}.txt")
                if os.path.exists(label_path) and os.path.exists(test_path):
                    gold = read_labeled_reviews(label_path)
                    gold_map = {rid: lab for rid, _, lab in gold}
                    test_pairs = read_reviews_with_id(test_path)
                    # Build evaluation set aligned by rid in gold
                    eval_texts: List[str] = []
                    eval_labels: List[int] = []
                    for rid, t in test_pairs:
                        if rid in gold_map:
                            eval_texts.append(t)
                            eval_labels.append(gold_map[rid])
                    if not eval_texts:
                        print("No labeled test examples found; please set --val_ratio>0 for sweep.")
                        continue
                    if args.sweep_param == 'C':
                        grid_vals = [float(x) for x in args.c_grid.split(',')]
                    else:  # max_df
                        grid_vals = [float(x) for x in args.df_grid.split(',')]
                    sweep_param_and_plot(
                        args.sweep_param, lang, lang_names[lang], list(train_texts), list(train_labels),
                        eval_texts, eval_labels, grid_vals, args.cn_max_df, args.en_max_df,
                        args.plot_dir, fixed_C=args.C, fixed_max_iter=5000
                    )
                else:
                    print("Sweep requires validation split (--val_ratio>0) or available test labels.")
                # After sweep, skip normal training for this lang and continue to next
                continue

        # Train final model on all training data with provided hyperparameters (no CV)
        rf_depth = None if args.rf_max_depth < 0 else args.rf_max_depth
        pipe = build_pipeline(
            lang, args.algo, args.cn_max_df, args.en_max_df,
            C=args.C,
            lr_max_iter=5000,
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=(rf_depth if rf_depth is not None else 30),
        )
        print(f"\nTraining {args.algo.upper()} ...")
        pipe.fit(train_texts, train_labels)
        if val_texts:
            eval_and_print(pipe, val_texts, val_labels, lang_names[lang], tag="Validation")

        # Predict test set and write submission
        test_path = os.path.join(args.test_root, f"test.{lang}.txt")
        test_pairs = read_reviews_with_id(test_path)
        preds = pipe.predict([t for _, t in test_pairs])
        out_path = f"{args.team}_{args.run}_{suffixes[lang]}.txt"
        write_submission(out_path, args.team, args.run, test_pairs, preds)
        print(f"Saved submission: {out_path}")

        # If label files exist, evaluate on test set labels
        label_path = os.path.join(args.label_root, f"test.label.{lang}.txt")
        if os.path.exists(label_path):
            gold = read_labeled_reviews(label_path)
            gold_map = {rid: lab for rid, _, lab in gold}
            y_true = []
            y_pred = []
            for idx, (rid, _) in enumerate(test_pairs):
                if rid in gold_map:
                    y_true.append(gold_map[rid])
                    y_pred.append(int(preds[idx]))
            if y_true:
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                print(f"[TEST {lang_names[lang]}] Accuracy: {acc:.4f}  F1: {f1:.4f}")
                print(classification_report(y_true, y_pred, digits=4))

    print("\nAll tasks finished.")


if __name__ == "__main__":
    main()

