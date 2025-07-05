# models/teacher_classifier.py
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

class CANBertForClassification(nn.Module):
    """
    사전 훈련된 CAN-BERT 모델 위에 분류 헤드를 추가한 미세 조정용 모델.
    """
    def __init__(self, config: BertConfig, num_labels: int = 2):
        """
        모델을 초기화합니다.

        Args:
            config (BertConfig): 사전 훈련된 BERT 모델의 설정.
            num_labels (int): 분류할 라벨의 수 (기본값: 2, 정상/공격).
        """
        super().__init__()
        self.num_labels = num_labels
        self.config = config

        # CAN-BERT 논문에 따라 사전 훈련된 BERT 모델을 불러옵니다.
        self.bert = BertModel(config)
        # 드롭아웃 레이어
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # BERT의 [CLS] 토큰 출력을 받아 라벨 수만큼의 출력으로 변환하는 분류 레이어
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        모델의 순전파를 수행합니다.

        Args:
            input_ids (torch.Tensor): 입력 토큰 ID 텐서.
            attention_mask (torch.Tensor, optional): 어텐션 마스크 텐서.
            labels (torch.Tensor, optional): 분류를 위한 실제 라벨 텐서.

        Returns:
            SequenceClassifierOutput: Hugging Face 표준 출력 형식에 따른 결과 (loss, logits 포함).
        """
        # 기본 BERT 모델로부터 출력을 얻습니다.
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # BERT의 출력 중 [CLS] 토큰에 해당하는 벡터를 사용합니다.
        # 이 벡터는 전체 시퀀스의 의미를 요약하는 역할을 합니다.
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        # 분류 레이어를 통과시켜 최종 로짓(logits)을 얻습니다.
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # 손실 함수 계산
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )