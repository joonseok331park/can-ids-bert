# models/teacher.py
# -*- coding: utf-8 -*-
"""
CAN‑BERT 'Teacher' 모델 (MLM) – Transformers 기반 구현
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead

__all__ = ["CANBertForMaskedLM"]


class CANBertForMaskedLM(nn.Module):
    """CAN‑BERT 교사 모델 (Masked‑LM 용)"""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.cls = BertLMPredictionHead(config)

    def forward(  # noqa: D401 – 스타일 유지
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.cls(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
        return (loss, logits) if loss is not None else (logits,)


# ---------------------------------------------------------------------- #
# 단독 실행 테스트 (선택)                                                 #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import pandas as pd
    from core.tokenizer import CANTokenizer

    df = pd.DataFrame(
        {
            "CAN_ID": ["01A", "02B"],
            "Data": [["11", "22", "33"], ["AA", "BB", "CC"]],
        }
    )
    tok = CANTokenizer()
    tok.build_vocab(df)

    # ✅ id_offset 프로퍼티 사용
    print("id_offset =", tok.id_offset)

    cfg = BertConfig(
        vocab_size=tok.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=1,
        intermediate_size=512,
        max_position_embeddings=126,
    )
    model = CANBertForMaskedLM(cfg)
    print(model)
