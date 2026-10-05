import argparse
import sys, os
import re
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PARENT_DIR)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils
from gym_subgoal_automata.utils.subgoal_automaton import SubgoalAutomaton
from task_parser.ilasp_solution_parser import parse_ilasp_solutions
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_filename", help="json file containing test examples")
    parser.add_argument("solution_filename", help="filename of the ILASP task solution")
    return parser

def predict_label(trace):
    next_state = "u0"
    for observables in trace:
        next_state = learnt_automaton.get_next_state(next_state, observables)
    if next_state == "u_acc":
        label = "goal"
    elif next_state == "u_rej":
        label = "deadend"
    else:
        label = "inc"
    return label

def accuracy(true_labels, predicted_labels):
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    acc = correct / len(true_labels)
    return acc

if __name__ == "__main__":
    args = get_argparser().parse_args()
    test_data = utils.read_json_file(args.test_filename)
    learnt_automaton = SubgoalAutomaton()
    
    solution_arg = args.solution_filename
    # Convert to absolute path
    solution_path = os.path.abspath(solution_arg)

    learnt_automaton = parse_ilasp_solutions(solution_path)
    traces_per_class = {}
    true_labels = []
    pred_labels = []
    test_result = {}
    accuracy_per_class = {}
    for true_class, test_traces in test_data.items():
        clean_label = true_class.replace("_examples", "")
        test_result[clean_label] = []
        
        for trace in test_traces:
            trace_label = predict_label(trace)
            true_labels.append(clean_label)
            pred_labels.append(trace_label)
            test_result[clean_label].append([
                trace,
                {"pred_label": trace_label}
            ])
    
    total_traces = sum(len(traces) for traces in test_result.values())
    print(f"The total number of test examples is {total_traces}")
    class_stats = {}
    ratio_stats = {}
    total_cls = {}
    false_entries = {cls:{"false_negative":[]} for cls in test_result.keys()}
    for cls in test_result.keys():
        class_stats[cls] = {
            "true_positive": 0,
            "false_negative": 0,
            "false_positive": 0,
            # "true_negative": 0
        }
        ratio_stats[cls] = {
            "true_positive": 0,
            "false_negative": 0,
            "false_positive": 0,
            # "true_negative": 0
        }
        total_cls[cls] = 0
    
    for true_class, entries in test_result.items():
        total_cls[true_class] = len(test_result[true_class])
        for trace_entry in entries:
            trace_id, trace_info = trace_entry
            pred_label = trace_entry[1]["pred_label"]

            for cls in test_result.keys():
                if true_class == cls and pred_label == cls:
                    class_stats[cls]["true_positive"] += 1
                elif true_class == cls and pred_label != cls:
                    class_stats[cls]["false_negative"] += 1
                    false_entries[cls]["false_negative"].append(trace_entry)
                elif true_class != cls and pred_label == cls:
                    class_stats[cls]["false_positive"] += 1
                # else:
                #     class_stats[cls]["true_negative"] += 1
    
    # Compute ratios
    for cls, stats in class_stats.items():
        print(f"\nClass: {cls} has {total_cls[cls]} test examples where")
        for metric, count in stats.items():
            if metric != "false_positive":
                ratio = count / total_cls[cls] if total_cls[cls] > 0 else 0
            else:
                ratio = count / (total_traces - total_cls[cls]) if total_cls[cls] > 0 else 0
            ratio_stats[cls][metric] = ratio
            print(f"  {metric}: {count} ({ratio:.2f})")
            test_result[f"{metric}_{cls}"] = [[count, ratio]]

    # ── Per-class confusion matrices ────────────────────────────────────────────
    all_classes = sorted(test_result.keys()
                         - {k for k in test_result if any(
                             k.startswith(p) for p in
                             ["true_positive_", "false_negative_", "false_positive_"])})
    # # Derive the clean class list from true_labels
    # class_list = sorted(set(true_labels))

    # fix: explicit ordering to match metrics table — goal → deadend → inc
    class_list = ["goal", "deadend", "inc"]

    print("\n" + "="*60)
    print("PER-CLASS CONFUSION MATRICES (One-vs-Rest)")
    print("="*60)

    # ── Confusion matrix plots ────────────────────────────────────────────────────
    np.set_printoptions(precision=2)

    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]

    for title, normalize in titles_options:
        fig, ax = plt.subplots(figsize=(8, 6))

        cm = confusion_matrix(
            true_labels,
            pred_labels,
            labels=class_list
        )
        if normalize == "true":
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fmt = ".4f" if normalize == "true" else "d"
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_list
        )
        disp.plot(cmap=plt.cm.Blues, colorbar=False, ax=ax, values_format=fmt)
        ax.set_title(title)
        print(title)
        print(cm)

        filename_suffix = "normalized" if normalize else "raw"
        plot_filename = f"confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        print(f"Saved: {plot_filename}")

    # ── Per-class one-vs-rest confusion matrices ──────────────────────────────────
    per_class_cm = {}
    for cls in class_list:
        binary_true = [1 if l == cls else 0 for l in true_labels]
        binary_pred = [1 if l == cls else 0 for l in pred_labels]
        cm_binary = confusion_matrix(binary_true, binary_pred, labels=[1, 0])
        tp, fn, fp, tn = cm_binary[0][0], cm_binary[0][1], cm_binary[1][0], cm_binary[1][1]
        per_class_cm[cls] = {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}

        print(f"\nClass: '{cls}'")
        print(f"  {'':12s}  Pred {cls:<10}  Pred NOT {cls}")
        print(f"  {'True ' + cls:<14}  {tp:<16}  {fn}")
        print(f"  {'True NOT ' + cls:<14}  {fp:<16}  {tn}")

    # ── Macro Precision, Recall, F1 ───────────────────────────────────────────────
    precision_macro = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall_macro    = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1_macro        = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    precision_per = precision_score(true_labels, pred_labels, average=None, labels=class_list, zero_division=0)
    recall_per    = recall_score(true_labels, pred_labels, average=None, labels=class_list, zero_division=0)
    f1_per        = f1_score(true_labels, pred_labels, average=None, labels=class_list, zero_division=0)

    print("\n" + "="*60)
    print("PRECISION / RECALL / F1  (per class + macro)")
    print("="*60)
    print(f"\n{'Class':<12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 44)
    for cls, p, r, f in zip(class_list, precision_per, recall_per, f1_per):
        print(f"{cls:<12}  {p:>10.4f}  {r:>8.4f}  {f:>8.4f}")
    print("-" * 44)
    print(f"{'MACRO':<12}  {precision_macro:>10.4f}  {recall_macro:>8.4f}  {f1_macro:>8.4f}")

    # ── Overall accuracy ─────────────────────────────────────────────────────────
    test_accuracy = accuracy(true_labels, pred_labels)
    print(f"\nOverall Accuracy: {test_accuracy:.4f}")

    # ── Save results ─────────────────────────────────────────────────────────────
    # ── Save results ─────────────────────────────────────────────────────────────
    test_result["Accuracy"] = [[test_accuracy]]
    test_result["Total test examples"] = [[total_traces]]

    output_filename = f"test_result.json"
    utils.write_json_obj_pretty(test_result, output_filename)

    # ── Save confusion matrices + F1/precision/recall to a separate file ─────────
    metrics_output = {
        "overall_accuracy": test_accuracy,
        "total_test_examples": total_traces,
        "macro": {
            "precision": precision_macro,
            "recall": recall_macro,
            "f1": f1_macro
        },
        "per_class_metrics": {
            cls: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f)
            }
            for cls, p, r, f in zip(class_list, precision_per, recall_per, f1_per)
        },
        "per_class_confusion_matrices": {
            cls: {"TP": int(tp_val), "FN": int(fn_val), "FP": int(fp_val), "TN": int(tn_val)}
            for cls, (tp_val, fn_val, fp_val, tn_val) in {
                cls: (
                    per_class_cm[cls]["TP"],
                    per_class_cm[cls]["FN"],
                    per_class_cm[cls]["FP"],
                    per_class_cm[cls]["TN"]
                )
                for cls in class_list
            }.items()
        }
    }

    metrics_filename = f"metrics.json"
    utils.write_json_obj_pretty(metrics_output, metrics_filename)
    print(f"\nMetrics saved to {metrics_filename}")
    
    output_filename = f"test_result.json"
    utils.write_json_obj_pretty(test_result, output_filename)

    output_data = {
    cls: entries
    for cls, entries in false_entries.items()
    }
    false_neg_filename = f"false_negatives_test_result.json"
    utils.write_json_obj_pretty(output_data, false_neg_filename)
