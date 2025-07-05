# scripts/evaluate.py
# -*- coding: utf-8 -*-
"""
미세 조정된 CAN‑BERT 분류 모델 평가 스크립트
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import BertConfig

from core.classification_dataset import ClassificationDataset
from core.tokenizer import CANTokenizer
from utils.data_loader import load_can_data


# -------------- 시퀀스 & 라벨 ---------------- #
def _make_sequences(
    df,
    tok: CANTokenizer,
    seq_len: int,
) -> Tuple[List[List[int]], List[int]]:
    frames = []
    for cid, payload, lbl in zip(df["CAN_ID"], df["Data"], df["Label"]):
        id_tok = str(int(cid, 16) + tok.id_offset)
        frame_tok = [id_tok] + payload + ["<VOID>"] * (8 - len(payload))
        frames.append((frame_tok, lbl))

    frames_per_seq = seq_len // 9
    seqs, labels = [], []
    for i in range(len(frames) - frames_per_seq + 1):
        win = frames[i : i + frames_per_seq]
        seqs.append(tok.encode([tk for fr, _ in win for tk in fr]))
        labels.append(1 if any(l == 1 for _, l in win) else 0)
    return seqs, labels


# -------------- main ---------------- #
def main() -> None:
    ap = argparse.ArgumentParser("Evaluate CAN‑BERT classifier")
    ap.add_argument("--test_data_path", required=True)
    ap.add_argument("--vocab_path", required=True)
    ap.add_argument("--finetuned_model_path", required=True)
    ap.add_argument("--seq_len", type=int, default=126)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CANTokenizer()
    tok.load_vocab(args.vocab_path)

    df = load_can_data(args.test_data_path)
    seqs, lbls = _make_sequences(df, tok, args.seq_len)
    loader = DataLoader(
        ClassificationDataset(seqs, lbls), batch_size=args.batch_size
    )

    from models.teacher_classifier import CANBertForClassification

    cfg = BertConfig(
        vocab_size=tok.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=1,
        intermediate_size=512,
        num_labels=2,
        max_position_embeddings=args.seq_len,
    )
    model = CANBertForClassification(cfg).to(device)
    model.load_state_dict(torch.load(args.finetuned_model_path, map_location=device))
    model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            logit = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            preds.extend(torch.argmax(logit, 1).cpu().tolist())
            gts.extend(batch["labels"].tolist())

    print(classification_report(gts, preds, target_names=["Normal", "Attack"]))

    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    out_png = Path("confusion_matrix.png")
    plt.savefig(out_png)
    print(f"[INFO] confusion matrix saved → {out_png}")


if __name__ == "__main__":
    main()
