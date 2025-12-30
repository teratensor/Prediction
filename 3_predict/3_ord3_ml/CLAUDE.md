# ord3 예측 ML 모델

## 개요
ord1, ord2가 주어졌을 때 ord3를 예측하는 ML 모델.

## 제약 조건
- ord2 < ord3 < ord6
- ord3 최빈 범위: 15-24 (평균 19.4)

## 주요 피처

- `candidate`: 후보 번호
- `range_idx`: 범위 인덱스 (0-4)
- `pos3_freq`: ord3 위치 출현 빈도
- `is_hot`, `is_cold`: HOT/COLD 비트 여부
- `is_prime`: 소수 여부
- `overall_freq`: 전체 출현 빈도
- `in_recent_3`: 최근 3회 출현 여부

## 실행 방법

```bash
python backtest.py
```

## 결과 파일

- `results/backtest_results.csv`: 백테스트 결과
- `results/feature_importance.csv`: 피처 중요도

## 성능 (378회차)

| 지표 | 값 |
|------|-----|
| Top-1 | 11.0% |
| Top-5 | 42.8% |
| Top-10 | 70.2% |
