import argparse
import json
import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, export_text
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_config", help="json file containing training examples")
    parser.add_argument("test_config", help="json file containing test examples")
    return parser

def load_dataset(filename):
    with open(filename, "r") as f:
        return json.load(f)

def encode_dataset(data_dict):
    X, y = [], []
    for label in target_labels:
        traces = data_dict.get(label, [])
        for trace in traces:
            trace = trace[:max_timesteps]
            while len(trace) < max_timesteps:
                trace.append([])
            encoded_trace = mlb.transform(trace)
            X.append(encoded_trace.flatten())
            y.append(label_map[label])
    return np.array(X), np.array(y)

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

# --------------------------------------------------
# Model
# --------------------------------------------------

def build_random_forest_model(seed):
    # Hyperparameters (tune as needed)
    n_estimators       = 100
    max_depth          = None   # None = grow until leaves are pure / min_samples_leaf reached
    min_samples_split  = 2
    min_samples_leaf   = 1
    criterion          = "gini"  # "gini" or "entropy"
    max_features       = "sqrt"

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=-1
    )
    return model

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    args = get_argparser().parse_args()

    target_labels = ["goal_examples", "deadend_examples", "inc_examples"]
    label_map     = {"goal_examples": 0, "deadend_examples": 1, "inc_examples": 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    max_timesteps = 10

    train_data = load_dataset(args.task_config)
    test_data  = load_dataset(args.test_config)

    all_requests = set()
    for label in target_labels:
        for trace in train_data.get(label, []):
            for slot in trace:
                all_requests.update(slot)

    mlb = MultiLabelBinarizer(classes=sorted(all_requests))
    mlb.fit([all_requests])

    X_train, y_train = encode_dataset(train_data)
    X_test,  y_test  = encode_dataset(test_data)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    os.makedirs("results", exist_ok=True)

    NUM_RUNS = 30

    test_accuracies_run  = np.zeros(NUM_RUNS)
    train_accuracies_run = np.zeros(NUM_RUNS)
    mean_depth_per_run   = np.zeros(NUM_RUNS)  # avg tree depth within each run's forest
    std_depth_per_run    = np.zeros(NUM_RUNS)  # depth variability across trees within that run

    all_run_predictions = []
    all_true_labels     = []
    all_pred_labels     = []

    run_times = []
    overall_start = time.time()

    for run in range(NUM_RUNS):
        print(f"\n===== Run {run+1}/{NUM_RUNS} =====")
        set_seed(run)
        run_start = time.time()

        model = build_random_forest_model(seed=run)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc  = model.score(X_test, y_test)
        train_accuracies_run[run] = train_acc
        test_accuracies_run[run]  = test_acc
        print(f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

        tree_depths = [est.get_depth() for est in model.estimators_]
        mean_depth_per_run[run] = np.mean(tree_depths)
        std_depth_per_run[run]  = np.std(tree_depths)
        print(f"avg tree depth={mean_depth_per_run[run]:.2f} (+/- {std_depth_per_run[run]:.2f} across {len(tree_depths)} trees)")

        y_pred = model.predict(X_test)
        all_run_predictions.append(y_pred)

        all_true_labels.extend([inv_label_map[int(t)] for t in y_test])
        all_pred_labels.extend([inv_label_map[int(p)] for p in y_pred])

        run_time = time.time() - run_start
        run_times.append(run_time)
        print(f"Run {run+1} completed in {run_time:.2f} seconds")

    # -------------------------------
    # Final accuracy summary
    # -------------------------------
    mean_accuracies_run = np.mean(test_accuracies_run)
    std_accuracies_run  = np.std(test_accuracies_run)
    mean_train_run      = np.mean(train_accuracies_run)
    std_train_run       = np.std(train_accuracies_run)

    print("\n===== Final Test Performance over runs =====")
    print("Testing accuracies over runs:", [f"{a:.4f}" for a in test_accuracies_run])
    print(f"Average Test Accuracy: {mean_accuracies_run:.4f}")
    print(f"Std Dev:               {std_accuracies_run:.4f}")

    # -------------------------------
    # Tree depth summary across runs
    # -------------------------------
    # mean_depth_per_run.mean() = average tree depth, averaged over trees AND runs
    # mean_depth_per_run.std()  = how much that run-level average depth swings run-to-run
    overall_mean_depth = np.mean(mean_depth_per_run)
    overall_std_depth  = np.std(mean_depth_per_run)

    print("\n===== Tree Depth Summary (Random Forest) =====")
    print("Avg tree depth per run:", [f"{d:.2f}" for d in mean_depth_per_run])
    print(f"Overall average tree depth: {overall_mean_depth:.2f}")
    print(f"Std Dev across runs:        {overall_std_depth:.2f}")

    # use last run predictions only for the JSON trace output
    y_pred = all_run_predictions[-1]

    total_time = time.time() - overall_start
    mean_run_time = np.mean(run_times)
    std_run_time  = np.std(run_times)

    print("\n===== Timing Summary =====")
    print(f"Total compilation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Mean time per run: {mean_run_time:.2f} seconds")
    print(f"Std Dev per run:   {std_run_time:.2f} seconds")

    # -------------------------------
    # Save run-level accuracy results
    # -------------------------------
    with open("results/randomforest_acc_task1.json", "w") as f:
        json.dump({
            "test_accuracies_per_run":  test_accuracies_run.tolist(),
            "train_accuracies_per_run": train_accuracies_run.tolist(),
            "mean_test_accuracy":  float(mean_accuracies_run),
            "std_test_accuracy":   float(std_accuracies_run),
            "mean_train_accuracy": float(mean_train_run),
            "std_train_accuracy":  float(std_train_run),
            "mean_tree_depth_per_run": mean_depth_per_run.tolist(),
            "std_tree_depth_per_run":  std_depth_per_run.tolist(),
            "overall_mean_tree_depth": float(overall_mean_depth),
            "overall_std_tree_depth":  float(overall_std_depth)
        }, f, indent=2)

    # -------------------------------
    # Metrics averaged over all runs
    # -------------------------------
    class_list     = ["goal_examples", "deadend_examples", "inc_examples"]
    display_labels = [cls.replace("_examples", "") for cls in class_list]

    precision_macro = precision_score(all_true_labels, all_pred_labels, average="macro", zero_division=0)
    recall_macro    = recall_score(all_true_labels, all_pred_labels, average="macro", zero_division=0)
    f1_macro        = f1_score(all_true_labels, all_pred_labels, average="macro", zero_division=0)

    precision_per = precision_score(all_true_labels, all_pred_labels, average=None, labels=class_list, zero_division=0)
    recall_per    = recall_score(all_true_labels, all_pred_labels, average=None, labels=class_list, zero_division=0)
    f1_per        = f1_score(all_true_labels, all_pred_labels, average=None, labels=class_list, zero_division=0)

    print("\n" + "="*60)
    print("PRECISION / RECALL / F1  (averaged over all runs)")
    print("="*60)
    print(f"\n{'Class':<20}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 52)
    for cls, p, r, f in zip(display_labels, precision_per, recall_per, f1_per):
        print(f"{cls:<20}  {p:>10.4f}  {r:>8.4f}  {f:>8.4f}")
    print("-" * 52)
    print(f"{'MACRO':<20}  {precision_macro:>10.4f}  {recall_macro:>8.4f}  {f1_macro:>8.4f}")

    metrics_output = {
        "overall_accuracy": float(mean_accuracies_run),
        "total_test_examples": len(y_test),
        "total_predictions_across_runs": len(all_pred_labels),
        "macro": {
            "precision": float(precision_macro),
            "recall":    float(recall_macro),
            "f1":        float(f1_macro)
        },
        "per_class_metrics": {
            cls: {"precision": float(p), "recall": float(r), "f1": float(f)}
            for cls, p, r, f in zip(class_list, precision_per, recall_per, f1_per)
        }
    }
    with open("randomforest_metrics_task1.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    print("\nMetrics saved to randomforest_metrics.json")

    # -------------------------------
    # Confusion matrices over all runs
    # -------------------------------
    np.set_printoptions(precision=2)
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    model_name = "RandomForest"
    for title, normalize in titles_options:
        fig, ax = plt.subplots(figsize=(8, 6))

        cm = confusion_matrix(
            all_true_labels,
            all_pred_labels,
            labels=class_list
        )
        if normalize == "true":
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels
        )
        disp.plot(cmap=plt.cm.Blues, colorbar=False, ax=ax, values_format=".4f")
        ax.set_title(f"{title} ({model_name}, averaged over {NUM_RUNS} runs)")
        print(f"\n{title}")
        print(cm)
        filename_suffix = "normalized" if normalize else "raw"
        plot_filename = f"confusion_matrix_{filename_suffix}_{model_name}.png"
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        print(f"Saved: {plot_filename}")

    # -------------------------------
    # Accuracy-across-runs plot
    # -------------------------------
    runs_axis = np.arange(1, NUM_RUNS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(runs_axis, train_accuracies_run, label="Train Accuracy", color="green", marker="o")
    plt.plot(runs_axis, test_accuracies_run, label="Test Accuracy", color="blue", marker="o")
    plt.axhline(mean_accuracies_run, color="blue", linestyle="--", alpha=0.5,
                label=f"Mean Test Acc = {mean_accuracies_run:.3f}")
    plt.title("Random Forest — Train vs Test Accuracy across Runs")
    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_test_accuracy_vs_runs_randomforest.png", dpi=300)
    plt.close()

    # -------------------------------
    # Forest visualizations (last run's forest)
    # -------------------------------
    # Reconstruct interpretable feature names: one per (timestep, request-type) cell,
    # matching the flatten order used in encode_dataset (timestep-major).
    feature_names = [f"t{t}_{req}" for t in range(max_timesteps) for req in mlb.classes_]

    # Aggregated Gini feature importance across all trees in the forest
    importances = model.feature_importances_
    top_n = min(20, len(feature_names))
    top_idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[top_idx][::-1], color="steelblue")
    plt.yticks(range(top_n), [feature_names[i] for i in top_idx][::-1])
    plt.xlabel("Importance (mean decrease in Gini impurity)")
    plt.title(f"Random Forest — Top {top_n} Feature Importances (last run)")
    plt.tight_layout()
    plt.savefig("randomforest_feature_importance_task1.png", dpi=150)
    plt.close()
    print("Saved: randomforest_feature_importance.png")

    # Structure of a couple of individual trees within the forest (illustrative only —
    # no single tree "is" the forest's decision rule, the forest votes across all of them).
    num_trees_to_plot = 2
    plot_depth = 3  # limits how much of each tree is DRAWN, not how deep it actually grew
    for i in range(min(num_trees_to_plot, len(model.estimators_))):
        plt.figure(figsize=(20, 10))
        plot_tree(
            model.estimators_[i],
            feature_names=feature_names,
            class_names=display_labels,
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=plot_depth
        )
        plt.title(f"Random Forest — Tree #{i} structure (showing top {plot_depth} levels)")
        plt.tight_layout()
        plt.savefig(f"randomforest_tree{i}_structure_task1.png", dpi=150)
        plt.close()
        print(f"Saved: randomforest_tree{i}_structure_task1.png")

    tree0_rules = export_text(model.estimators_[0], feature_names=feature_names)
    with open("randomforest_tree0_rules_task1.txt", "w") as f:
        f.write(tree0_rules)
    print("Saved: randomforest_tree0_rules.txt (rules for tree #0, illustrative)")
