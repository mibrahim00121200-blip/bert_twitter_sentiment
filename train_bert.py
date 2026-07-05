"""
train_bert.py
Trains a BERT-based sentiment classifier on a Twitter sentiment dataset (CSV).
Saves the trained model, tokenizer, label mapping, and evaluation graphs.

Usage:
    python train_bert.py --data dataset/twitter_sentiment.csv --text_col text --label_col sentiment --epochs 3

Requirements column names can be anything -- just point --text_col / --label_col
to whatever columns hold the tweet text and the sentiment label in your CSV.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)


class TweetDataset(Dataset):
    """Wraps tweet texts + integer labels into tokenized tensors for BERT."""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_data(csv_path, text_col, label_col):
    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    df["text"] = df["text"].astype(str)

    classes = sorted(df["label"].astype(str).unique().tolist())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    df["label_id"] = df["label"].astype(str).map(label2id)

    return df, label2id, id2label


def plot_class_distribution(df, out_dir):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="label", data=df, order=df["label"].value_counts().index)
    plt.title("Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"))
    plt.close()


def plot_loss_curve(train_losses, val_losses, out_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()


def plot_accuracy_curve(train_accs, val_accs, out_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, id2label, out_dir):
    labels_sorted = [id2label[i] for i in sorted(id2label.keys())]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels_sorted, yticklabels=labels_sorted,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to twitter_sentiment.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="sentiment")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="saved_bert_model")
    parser.add_argument("--report_dir", type=str, default="report_graphs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, label2id, id2label = load_data(args.data, args.text_col, args.label_col)
    plot_class_distribution(df, args.report_dir)

    train_df, val_df = train_test_split(
        df, test_size=args.test_size, random_state=42, stratify=df["label_id"]
    )

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id)
    ).to(device)

    train_ds = TweetDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer, args.max_len)
    val_ds = TweetDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Training]")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
        )

    plot_loss_curve(train_losses, val_losses, args.report_dir)
    plot_accuracy_curve(train_accs, val_accs, args.report_dir)

    _, final_val_acc, y_true, y_pred = evaluate(model, val_loader, device)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    print("\n=== Final Evaluation ===")
    print(f"Accuracy:  {final_val_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    target_names = [id2label[i] for i in sorted(id2label)]
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    plot_confusion_matrix(y_true, y_pred, id2label, args.report_dir)

    with open(os.path.join(args.report_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {final_val_acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=target_names))

    # Save model, tokenizer, and label mapping so the GUI can load them later
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print(f"\nModel, tokenizer, and label map saved to: {args.output_dir}")
    print(f"Graphs and metrics saved to: {args.report_dir}")


if __name__ == "__main__":
    main()
