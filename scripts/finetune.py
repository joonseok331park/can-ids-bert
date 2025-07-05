# scripts/finetune.py
# -*- coding: utf-8 -*-
"""
CAN‑BERT 분류용 미세 조정 스크립트
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertConfig, get_linear_schedule_with_warmup

from core.classification_dataset import ClassificationDataset
from core.tokenizer import CANTokenizer
from utils.data_loader import load_can_data


# ---------------- 시퀀스 & 라벨 ---------------- #
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
        tokens = [tk for fr, _ in win for tk in fr]
        seqs.append(tok.encode(tokens))
        labels.append(1 if any(l == 1 for _, l in win) else 0)
    return seqs, labels


# ---------------- main ---------------- #
def main() -> None:
    ap = argparse.ArgumentParser("Fine‑tune CAN‑BERT classifier")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--vocab_path", required=True)
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--output_dir", default="checkpoints")
    ap.add_argument("--seq_len", type=int, default=126)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--mask_prob", type=float, default=0.45, help="Mask probability")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    tok = CANTokenizer()
    tok.load_vocab(args.vocab_path)

    df = load_can_data(args.data_path)
    seqs, lbls = _make_sequences(df, tok, args.seq_len)
    X_tr, X_val, y_tr, y_val = train_test_split(
        seqs, lbls, test_size=0.2, stratify=lbls, random_state=42
    )

    tr_loader = DataLoader(
        ClassificationDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ClassificationDataset(X_val, y_val), batch_size=args.batch_size
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

    # Teacher pretrain 가중치 로드
    ckpt = torch.load(args.pretrained_ckpt, map_location=device)
    bert_sd = {
        k.replace("bert.", ""): v
        for k, v in ckpt["model_state"].items()
        if k.startswith("bert.")
    }
    model.bert.load_state_dict(bert_sd, strict=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(
        optim, 0, len(tr_loader) * args.epochs
    )

    for ep in range(args.epochs):
        model.train()
        for batch in tqdm(tr_loader, desc=f"Epoch {ep+1}/{args.epochs}"):
            optim.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

        # validation
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                logit = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                preds.extend(torch.argmax(logit, 1).cpu().tolist())
                gts.extend(batch["labels"].tolist())
        f1 = f1_score(gts, preds)
        print(f"[VAL] epoch {ep+1}: F1 = {f1:.4f}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dst = Path(args.output_dir) / "can-bert-finetuned.pt"
    torch.save(model.state_dict(), dst)
    print(f"[INFO] saved → {dst}")


if __name__ == "__main__":
    main()
