import argparse
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PARENT_DIR)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import tensorflow as tf # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer # type: ignore
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from utils import utils
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# from tensorflow.keras.utils import to_categorical

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
            encoded_trace = mlb.transform(trace)  # transform only, never fit_transform
            X.append(encoded_trace)
            y.append(label_map[label])
    return np.array(X), np.array(y)

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_lstm_model(max_timesteps, num_features, num_classes):
    model = Sequential(
        [
            Input(shape=(max_timesteps, num_features)),
            LSTM(32),
            # Dense(16, activation='relu'),
            # Dropout(0.1),
            # Dense(32),
            Dense(num_classes, activation='softmax')
        ]
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":

    args = get_argparser().parse_args()

    target_labels = ["goal_examples", "deadend_examples", "inc_examples"]
    label_map = {"goal_examples": 0, "deadend_examples": 1, "inc_examples": 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    max_timesteps = 10
    train_file = args.task_config
    test_file = args.test_config

    print("Running on:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")
    
    # -------------------------------
        # 2. Load data
    # -------------------------------
    train_data = load_dataset(train_file)
    test_data = load_dataset(test_file)

    # -------------------------------
        # 3. Gather all request types (for encoding)
    # -------------------------------
    all_requests = set()
    for label in target_labels:
        traces = train_data.get(label, [])
        for trace in traces:
            for slot in trace:
                all_requests.update(slot)

    all_requests = sorted(all_requests)
    print(f"Total unique request types: {len(all_requests)}")

    mlb = MultiLabelBinarizer(classes=all_requests)
    mlb.fit([all_requests])

    X_train, y_train = encode_dataset(train_data)
    X_test, y_test = encode_dataset(test_data)

    print(f"X_train_shape:", X_train.shape)  # (N, 10, number_of_unique_requests)
    print(f"y_train_shape:", y_train.shape)
    print(f"X_test_shape:", X_test.shape)
    print(f"y_test_shape:", y_test.shape)

    NUM_RUNS = 30
    EPOCHS = 50

    test_acc_matrix = np.zeros((NUM_RUNS, EPOCHS))
    test_accuracies = []

    all_run_predictions = []
    all_true_labels     = []
    all_pred_labels     = []

    run_times = []
    overall_start = time.time()

    for run in range(NUM_RUNS):
        print(f"\n===== Run {run+1}/{NUM_RUNS} =====")

        set_seed(run)
        run_start = time.time()
        
        # -------------------------------
            # 4. Build LSTM model
        # -------------------------------
        model = build_lstm_model(
            max_timesteps=max_timesteps,
            num_features=X_train.shape[2],
            num_classes=len(target_labels)
        )
        # model.summary()
    
        # -------------------------------
            # 5. Train
        # -------------------------------

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,         # stop if no improvement for 10 epochs
            restore_best_weights=True
        )

        for epoch in range(EPOCHS):
            history = model.fit(
                X_train, y_train, 
                epochs=1, 
                batch_size=8, 
                validation_split=0,
                callbacks=[early_stop],
                verbose=0
            )

            train_acc  = history.history['accuracy'][0]
            train_loss = history.history['loss'][0]
            # print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}  train_loss={train_loss:.4f}")

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

        

        all_run_predictions.append((y_pred, y_pred_probs))

        # collect string labels from every run
        all_true_labels.extend([inv_label_map[int(t)] for t in y_test])
        all_pred_labels.extend([inv_label_map[int(p)] for p in y_pred])

        run_time = time.time() - run_start
        run_times.append(run_time)
        print(f"Run {run+1} completed in {run_time:.2f} seconds")
    
    test_accuracies_run = test_acc_matrix[:,-1]
    mean_accuracies_run = np.mean(test_accuracies_run)
    std_accuracies_run = np.std(test_accuracies_run)

    print("\n===== Final Test Performance over runs =====")
    formatted_mean_run = [f"{accuracy_run:.4f}" for accuracy_run in test_accuracies_run]
    print("Testing accuracies over runs:", formatted_mean_run)
    print(f"Average Test Accuracy: {mean_accuracies_run:.4f}")
    print(f"Std Dev: {std_accuracies_run:.4f}")

    
    mean_acc_epoch = test_acc_matrix.mean(axis=0)
    std_acc_epoch  = test_acc_matrix.std(axis=0)
    
    formatted_mean_epoch = [f"{accuracy_epoch:.4f}" for accuracy_epoch in mean_acc_epoch]
    formatted_std_epoch = [f"{std_accuracy_epoch:.4f}" for std_accuracy_epoch in std_acc_epoch]
    print("Testing accuracies over epochs:", formatted_mean_epoch)
    print("Std of testing accuracies over epochs:", formatted_std_epoch)
    # print(f"Average Test Accuracy: {np.mean(mean_acc_epoch):.4f}")
    # print(f"Std Dev: {np.std(mean_acc_epoch):.4f}")

    epochs = np.arange(1, EPOCHS+1)

    y_pred, y_pred_probs = all_run_predictions[-1]

    total_time = time.time() - overall_start
    mean_run_time = np.mean(run_times)
    std_run_time  = np.std(run_times)

    print("\n===== Timing Summary =====")
    print(f"Total compilation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Mean time per run: {mean_run_time:.2f} seconds")
    print(f"Std Dev per run:   {std_run_time:.2f} seconds")

    # -------------------------------
        # 7. Save results to JSON
    # -------------------------------
    results = {label: [] for label in target_labels}
    idx = 0
    for label in target_labels:
        traces = test_data.get(label, [])
        for trace in traces:
            pred_label = inv_label_map[int(y_pred[idx])]
            confidence = float(np.max(y_pred_probs[idx]))
            results[label].append({
                "true_trace": trace,
                "predicted_label": pred_label,
                "confidence": confidence
            })
            idx += 1

    output_file = "LSTM_test_results_dst130_st6.json"
    utils.write_json_obj_pretty(results, output_file)

    acc_results = {
        "mean_test_accuracy": mean_acc_epoch.tolist(),
        "std_test_accuracy": std_acc_epoch.tolist(),
        "epochs": list(range(1, EPOCHS + 1))
    }
    with open("results/lstm_acc.json", "w") as f:
        json.dump(acc_results, f, indent=2)

    # -------------------------------------------------------------------------------
        # 8. Metrics (precision, recall, F1, confusion matrix) averaged over all runs
    # -------------------------------------------------------------------------------

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
        "timing": {
            "total_seconds": float(total_time),
            "mean_seconds_per_run": float(mean_run_time),
            "std_seconds_per_run": float(std_run_time)
        },
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
    metrics_filename = "lstm_metrics.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"\nMetrics saved to {metrics_filename}")

    # -------------------------------
    #  Confusion matrices over all runs
    # -------------------------------
    np.set_printoptions(precision=2)
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    model_name = "LSTM"
    for title, normalize in titles_options:
        fig, ax = plt.subplots(figsize=(8, 6))

        # compute confusion matrix with explicit label ordering
        cm = confusion_matrix(
            all_true_labels,
            all_pred_labels,
            labels=class_list        # force same order as metrics
        )
        if normalize == "true":
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels   # clean names, same order
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
        # 9. Plot testing accuracy vs epochs
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
    plt.title("Average Test Accuracy vs Epochs (LSTM)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_test_accuracy_vs_epochs.png", dpi=300)
    plt.close()
