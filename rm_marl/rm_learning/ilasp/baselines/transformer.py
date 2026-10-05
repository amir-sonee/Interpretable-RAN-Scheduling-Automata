import argparse
import json
import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, Model, Input  # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
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

# --------------------------------------------------
# Transformer components
# --------------------------------------------------

class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return x + self.pos_emb(positions)

def transformer_encoder(x, num_heads, d_model, d_ff, dropout):
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff_output = layers.Dense(d_ff, activation="relu")(x)
    ff_output = layers.Dense(d_model)(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    return x

def build_transformer_model(max_timesteps, num_features, num_classes):
    d_model    = 32 #32 # 512
    num_heads  = 4 #4 # 8
    d_ff       = 128 #64 # 1024 or 2048
    num_layers = 2 #2 # 6
    dropout  = 0

    inputs = Input(shape=(max_timesteps, num_features))
    
    x = layers.Dense(d_model)(inputs)
    x = PositionalEmbedding(max_timesteps, d_model)(x)
    x = layers.Dropout(dropout)(x)
    # ----------------------------
    # Stack of N layers of Encoder
    for _ in range(num_layers):
        # x = transformer_encoder(x, num_heads, d_model, d_ff)
        x = transformer_encoder(x, num_heads, d_model, d_ff, dropout)
    # ----------------------------

    x = layers.GlobalAveragePooling1D()(x) # removing the timesteps (time sequential) for classfication
    # x = layers.Dense(16, activation="relu")(x)
    # x = layers.Dropout(dropout)(x)
    # x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
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

    NUM_RUNS = 30
    EPOCHS   = 50

    test_acc_matrix  = np.zeros((NUM_RUNS, EPOCHS))
    train_acc_matrix = np.zeros((NUM_RUNS, EPOCHS))

    all_run_predictions = []
    all_true_labels     = []   # fix: collect across all runs
    all_pred_labels     = []   # fix: collect across all runs

    run_times = []
    overall_start = time.time()

    for run in range(NUM_RUNS):
        print(f"\n===== Run {run+1}/{NUM_RUNS} =====")
        set_seed(run)
        run_start = time.time()

        model = build_transformer_model(
            max_timesteps,
            X_train.shape[2],
            len(target_labels)
        )

        for epoch in range(EPOCHS):
            history = model.fit(
                X_train, y_train,
                batch_size=8,
                epochs=1,
                verbose=0
            )

            train_acc = history.history['accuracy'][0]
            train_acc_matrix[run, epoch] = train_acc

            _, test_acc = model.evaluate(X_test, y_test, verbose=0)
            test_acc_matrix[run, epoch] = test_acc
            print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

        # fix: store predictions after each run's epoch loop completes
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
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
    mean_acc_epoch   = test_acc_matrix.mean(axis=0)
    std_acc_epoch    = test_acc_matrix.std(axis=0)
    mean_train_epoch = train_acc_matrix.mean(axis=0)
    std_train_epoch  = train_acc_matrix.std(axis=0)

    test_accuracies_run = test_acc_matrix[:, -1]
    mean_accuracies_run = np.mean(test_accuracies_run)
    std_accuracies_run  = np.std(test_accuracies_run)

    print("\n===== Final Test Performance over runs =====")
    print("Testing accuracies over runs:", [f"{a:.4f}" for a in test_accuracies_run])
    print(f"Average Test Accuracy: {mean_accuracies_run:.4f}")
    print(f"Std Dev:               {std_accuracies_run:.4f}")

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
    # Save trace results to JSON
    # -------------------------------
    with open("results/transformer_acc.json", "w") as f:
        json.dump({
            "mean_test_accuracy":  mean_acc_epoch.tolist(),
            "std_test_accuracy":   std_acc_epoch.tolist(),
            "mean_train_accuracy": mean_train_epoch.tolist(),
            "std_train_accuracy":  std_train_epoch.tolist(),
            "epochs": list(range(1, EPOCHS + 1))
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
    with open("transformer_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    print("\nMetrics saved to transformer_metrics.json")

    # -------------------------------
    # Confusion matrices over all runs
    # -------------------------------
    np.set_printoptions(precision=2)
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    model_name = "Transformer"
    for title, normalize in titles_options:
        fig, ax = plt.subplots(figsize=(8, 6))

        # fix: compute with explicit label ordering matching metrics
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
    # Accuracy plot
    # -------------------------------
    epochs = np.arange(1, EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_train_epoch, label="Mean Train Accuracy", color="green")
    plt.fill_between(
        epochs,
        mean_train_epoch - std_train_epoch,
        mean_train_epoch + std_train_epoch,
        alpha=0.2, color="green"
    )
    plt.plot(epochs, mean_acc_epoch, label="Mean Test Accuracy", color="blue")
    plt.fill_between(
        epochs,
        mean_acc_epoch - std_acc_epoch,
        mean_acc_epoch + std_acc_epoch,
        alpha=0.2, color="blue"
    )
    plt.title("Transformer — Train vs Test Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("avg_test_accuracy_vs_epochs_transformer.png", dpi=300)
    plt.close()
