import unittest

import torch
from transformers import BertConfig

from models.teacher import CANBertForMaskedLM
from models.teacher_classifier import CANBertForClassification


class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
        )
        self.input_ids = torch.randint(0, 32, (2, 8))
        self.attention_mask = torch.ones_like(self.input_ids)

    def test_mlm_forward_shape(self):
        model = CANBertForMaskedLM(self.config).eval()
        with torch.no_grad():
            (logits,) = model(self.input_ids, self.attention_mask)
        self.assertEqual(tuple(logits.shape), (2, 8, 32))

    def test_classifier_forward_shape_and_loss(self):
        model = CANBertForClassification(self.config, num_labels=4).eval()
        labels = torch.tensor([0, 3])
        with torch.no_grad():
            output = model(self.input_ids, self.attention_mask, labels)
        self.assertEqual(tuple(output.logits.shape), (2, 4))
        self.assertEqual(output.loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()
