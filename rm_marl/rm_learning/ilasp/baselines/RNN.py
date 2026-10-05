import argparse
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PARENT_DIR)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout # type: ignore
from utils import utils
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

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
            while len(trace) < max_timesteps:
                trace.append([])
            encoded_trace = mlb.transform(trace)
            X.append(encoded_trace)
            y.append(label_map[label])
    return np.array(X), np.array(y)

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_RNN_model(max_timesteps, num_features, num_classes):
    model = Sequential([
        Input(shape=(max_timesteps, num_features)),
        SimpleRNN(32),
        # Dense(16, activation='relu'),
        # Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":

    script_start = time.time()

    args = get_argparser().parse_args()

    target_labels = ["goal_examples", "deadend_examples", "inc_examples"]
    label_map     = {"goal_examples": 0, "deadend_examples": 1, "inc_examples": 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    max_timesteps = 10
    train_file = args.task_config
    test_file  = args.test_config

    print("Running on:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

    # -------------------------------
    # 2. Load data
    # -------------------------------
    train_data = load_dataset(train_file)
    test_data  = load_dataset(test_file)

    # -------------------------------
    # 3. Gather all request types (for encoding)
    # -------------------------------
    all_requests = set()
    for label in target_labels:
        for trace in train_data.get(label, []):
            for slot in trace:
                all_requests.update(slot)

    all_requests = sorted(all_requests)
    print(f"Total unique request types: {len(all_requests)}")

    mlb = MultiLabelBinarizer(classes=all_requests)
    mlb.fit([all_requests])

    X_train, y_train = encode_dataset(train_data)
    X_test,  y_test  = encode_dataset(test_data)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")   # fix: was printing X_train by mistake
    print(f"y_test shape:  {y_test.shape}")   # fix: was printing y_train by mistake

    NUM_RUNS = 30
    EPOCHS   = 50

    test_acc_matrix = np.zeros((NUM_RUNS, EPOCHS))

    all_run_predictions = []
    all_true_labels     = []   # fix: collect across all runs
    all_pred_labels     = []   # fix: collect across all runs

    run_times = []
    overall_start = time.time()

    for run in range(NUM_RUNS):
        print(f"\n===== Run {run+1}/{NUM_RUNS} =====")
        set_seed(run)
        run_start = time.time()

        # -------------------------------
        # 4. Build RNN model
        # -------------------------------
        model = build_RNN_model(
            max_timesteps=max_timesteps,
            num_features=X_train.shape[2],
            num_classes=len(target_labels)
        )

        for epoch in range(EPOCHS):

            # -------------------------------
            # 5. Train
            # -------------------------------
            history = model.fit(
                X_train, y_train,
                epochs=1,
                batch_size=8,
                validation_split=0,
                verbose=0
            )

            train_acc  = history.history['accuracy'][0]
            train_loss = history.history['loss'][0]

            # -------------------------------
            # 6. Evaluate
            # -------------------------------
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}  test_loss={test_loss:.4f}")

            # -------------------------------
            # 7. Predict
            # -------------------------------
            y_pred_probs = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            acc = np.mean(y_pred == y_test)
            test_acc_matrix[run, epoch] = acc

        # store final predictions for this run
        all_run_predictions.append((y_pred, y_pred_probs))

        # fix: collect string labels from every run for averaged metrics
        all_true_labels.extend([inv_label_map[int(t)] for t in y_test])
        all_pred_labels.extend([inv_label_map[int(p)] for p in y_pred])

        run_time = time.time() - run_start
        run_times.append(run_time)
        print(f"Run {run+1} completed in {run_time:.2f} seconds")

    # -------------------------------
    # Final accuracy summary
    # -------------------------------
    test_accuracies_run = test_acc_matrix[:, -1]
    mean_accuracies_run = np.mean(test_accuracies_run)
    std_accuracies_run  = np.std(test_accuracies_run)

    print("\n===== Final Test Performance over runs =====")
    print("Testing accuracies over runs:", [f"{a:.4f}" for a in test_accuracies_run])
    print(f"Average Test Accuracy: {mean_accuracies_run:.4f}")
    print(f"Std Dev:               {std_accuracies_run:.4f}")

    mean_acc_epoch = test_acc_matrix.mean(axis=0)
    std_acc_epoch  = test_acc_matrix.std(axis=0)

    print("Testing accuracies over epochs:", [f"{a:.4f}" for a in mean_acc_epoch])
    print("Std of testing accuracies over epochs:", [f"{a:.4f}" for a in std_acc_epoch])

    epochs = np.arange(1, EPOCHS + 1)

    # use last run predictions only for the JSON trace output
    y_pred, y_pred_probs = all_run_predictions[-1]

    total_time = time.time() - overall_start
    mean_run_time = np.mean(run_times)
    std_run_time  = np.std(run_times)

    print("\n===== Timing Summary =====")
    print(f"Total compilation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Mean time per run: {mean_run_time:.2f} seconds")
    print(f"Std Dev per run:   {std_run_time:.2f} seconds")

    # -------------------------------
    # 8. Save trace results to JSON
    # -------------------------------
    results = {label: [] for label in target_labels}
    idx = 0
    for label in target_labels:
        for trace in test_data.get(label, []):
            pred_label = inv_label_map[int(y_pred[idx])]
            confidence = float(np.max(y_pred_probs[idx]))
            results[label].append({
                "true_trace": trace,
                "predicted_label": pred_label,
                "confidence": confidence
            })
            idx += 1

    output_file = "RNN_test_results_dst130_st6.json"
    utils.write_json_obj_pretty(results, output_file)

    acc_results = {
        "mean_test_accuracy": mean_acc_epoch.tolist(),
        "std_test_accuracy":  std_acc_epoch.tolist(),
        "epochs": list(range(1, EPOCHS + 1))
    }
    with open("results/rnn_acc.json", "w") as f:
        json.dump(acc_results, f, indent=2)

    # -------------------------------
    # 9. Metrics averaged over all runs
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
    with open("rnn_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    print("\nMetrics saved to rnn_metrics.json")

    # -------------------------------
    # 10. Confusion matrices over all runs
    # -------------------------------
    np.set_printoptions(precision=2)
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    model_name = "RNN"
    for title, normalize in titles_options:
        fig, ax = plt.subplots(figsize=(8, 6))

        # fix: compute confusion matrix with explicit label ordering
        cm = confusion_matrix(
            all_true_labels,
            all_pred_labels,
            labels=class_list        # forces goal→deadend→inc order
        )
        if normalize == "true":
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels
        )
        # disp.plot(cmap=plt.cm.Blues, colorbar=False, ax=ax)
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
    # 11. Plot testing accuracy vs epochs
    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_acc_epoch, label="Mean Test Accuracy")
    plt.fill_between(
        epochs,
        mean_acc_epoch - std_acc_epoch,
        mean_acc_epoch + std_acc_epoch,
        alpha=0.3,
        label="±1 Std Dev"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Average Test Accuracy vs Epochs (RNN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_test_accuracy_vs_epochs_RNN.png", dpi=300)
    plt.close()
