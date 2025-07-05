# -*- coding: utf-8 -*-
"""
CAN-BERT 데이터 로딩 테스트
"""
import io
import pytest
from pathlib import Path

from utils.data_loader import (
    load_can_data_from_file_stream, 
    _parse_line, 
    estimate_valid_line_ratio,
    _safe_match
)
from core.tokenizer import CANTokenizer
from core.dataset import MLMDataset


# 테스트용 샘플 데이터
SAMPLE_CANDUMP = """\
(1613599955.394625) can0 0C8#0000000000000000
1613599956.10 can1 123#ABCDEF
(1613599957.123456) vcan0 7FF#123456789ABCDEF0
1613599958.999 789#
invalid line without proper format
123.456 can2 ABC#112233
"""

MINIMAL_SAMPLE = """\
(1613599955.394625) can0 0C8#0000000000000000
1613599956.10 can1 123#ABCDEF
"""


class TestRegexParsing:
    """정규식 파싱 테스트"""
    
    def test_safe_match_valid_lines(self):
        """유효한 라인들에 대한 정규식 매칭 테스트"""
        valid_lines = [
            "(1613599955.394625) can0 0C8#0000000000000000",
            "1613599956.10 can1 123#ABCDEF", 
            "(1613599957.123456) vcan0 7FF#123456789ABCDEF0",
            "1613599958.999 789#",
            "123.456 can2 ABC#112233",
        ]
        
        for line in valid_lines:
            match = _safe_match(line)
            assert match is not None, f"라인이 매칭되어야 함: {line}"
            assert "ts" in match.groupdict()
            assert "id" in match.groupdict()
            assert "payload" in match.groupdict()

    def test_safe_match_invalid_lines(self):
        """유효하지 않은 라인들에 대한 정규식 매칭 테스트"""
        invalid_lines = [
            "invalid line without proper format",
            "no timestamp here can0 123#ABCD",
            "123.456 # missing id",
            "",
            "   ",
        ]
        
        for line in invalid_lines:
            match = _safe_match(line)
            assert match is None, f"라인이 매칭되지 않아야 함: {line}"

    def test_parse_line_functionality(self):
        """_parse_line 함수 기능 테스트"""
        line = "(1613599955.394625) can0 0C8#0000000000000000"
        result = _parse_line(line)
        
        assert result is not None
        assert result["ts"] == 1613599955.394625
        assert result["id"] == "0C8"
        assert result["payload"] == "0000000000000000"
        assert result["len"] == 8

    def test_parse_line_empty_payload(self):
        """빈 payload 테스트"""
        line = "1613599958.999 789#"
        result = _parse_line(line)
        
        assert result is not None
        assert result["id"] == "789"
        assert result["payload"] == ""
        assert result["len"] == 0


class TestDataLoader:
    """데이터 로더 테스트"""
    
    def test_load_can_data_from_file_stream(self):
        """파일 스트림에서 CAN 데이터 로딩 테스트"""
        stream = io.StringIO(SAMPLE_CANDUMP)
        rows = list(load_can_data_from_file_stream(stream))
        
        # 유효한 라인은 5개여야 함 (invalid line 제외)
        assert len(rows) == 5
        
        # 첫 번째 행 검증
        assert rows[0]["id"] == "0C8"
        assert rows[0]["payload"] == "0000000000000000"
        assert rows[0]["len"] == 8
        
        # 두 번째 행 검증  
        assert rows[1]["id"] == "123"
        assert rows[1]["payload"] == "ABCDEF"
        assert rows[1]["len"] == 3

    def test_estimate_valid_line_ratio(self, tmp_path):
        """유효 라인 비율 추정 테스트"""
        test_file = tmp_path / "test.candump"
        test_file.write_text(SAMPLE_CANDUMP)
        
        ratio = estimate_valid_line_ratio(test_file, sample_size=10)
        
        # 6개 라인 중 5개가 유효하므로 약 0.83
        assert 0.8 <= ratio <= 0.9

    def test_estimate_valid_line_ratio_empty_file(self, tmp_path):
        """빈 파일에 대한 유효 라인 비율 테스트"""
        test_file = tmp_path / "empty.candump"
        test_file.write_text("")
        
        ratio = estimate_valid_line_ratio(test_file)
        assert ratio == 0.0


class TestMLMDataset:
    """MLMDataset 테스트"""
    
    def test_mlm_dataset_with_stream_override(self):
        """스트림 오버라이드를 사용한 MLMDataset 테스트"""
        tokenizer = CANTokenizer()
        
        # 간단한 테스트를 위해 작은 시퀀스 길이 사용
        dataset = MLMDataset(
            data_path=Path("dummy"),  # 실제로는 사용되지 않음
            tokenizer=tokenizer,
            seq_len=9,  # 1 프레임 = 9 토큰
            stride=9,
            stream_override=io.StringIO(MINIMAL_SAMPLE)
        )
        
        # 길이 확인
        assert len(dataset) > 0
        
        # iterator 테스트
        iterator = iter(dataset)
        sample = next(iterator)
        
        assert "input_ids" in sample
        assert "attention_mask" in sample  
        assert "labels" in sample
        assert sample["input_ids"].shape[0] == 9
        assert sample["attention_mask"].shape[0] == 9
        assert sample["labels"].shape[0] == 9

    def test_mlm_dataset_len_consistency(self):
        """__len__()과 __iter__() 일관성 테스트"""
        tokenizer = CANTokenizer()
        
        dataset = MLMDataset(
            data_path=Path("dummy"),
            tokenizer=tokenizer, 
            seq_len=18,  # 2 프레임 = 18 토큰
            stride=9,    # 1 프레임씩 이동
            stream_override=io.StringIO(MINIMAL_SAMPLE)
        )
        
        expected_len = len(dataset)
        actual_samples = list(iter(dataset))
        
        # 길이와 실제 샘플 수가 일치해야 함
        assert len(actual_samples) == expected_len

    def test_mlm_dataset_empty_stream(self):
        """빈 스트림에 대한 MLMDataset 테스트"""
        tokenizer = CANTokenizer()
        
        dataset = MLMDataset(
            data_path=Path("dummy"),
            tokenizer=tokenizer,
            seq_len=9,
            stride=9,
            stream_override=io.StringIO("")
        )
        
        # 빈 스트림은 길이가 0이어야 함
        assert len(dataset) == 0
        
        # iterator도 아무것도 반환하지 않아야 함
        samples = list(iter(dataset))
        assert len(samples) == 0

    def test_mlm_dataset_masking(self):
        """MLM 마스킹 기능 테스트"""
        tokenizer = CANTokenizer()
        
        dataset = MLMDataset(
            data_path=Path("dummy"),
            tokenizer=tokenizer,
            seq_len=9,
            stride=9,
            mask_prob=0.5,  # 높은 마스킹 확률로 테스트
            stream_override=io.StringIO(MINIMAL_SAMPLE)
        )
        
        sample = next(iter(dataset))
        labels = sample["labels"]
        
        # 일부 토큰이 마스킹되어야 함 (labels에 -100이 아닌 값이 있어야 함)
        masked_tokens = (labels != -100).sum().item()
        assert masked_tokens > 0, "일부 토큰이 마스킹되어야 함"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 