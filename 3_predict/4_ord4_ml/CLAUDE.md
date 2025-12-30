# ord4 예측 ML 모델

## 개요
ord1, ord6가 주어졌을 때 ord4를 예측하는 ML 모델.

## 제약 조건
- ord3 < ord4 < ord5
- ord4 공식: `round(ord1 + (ord6 - ord1) * 0.60) ± 10`

## 주요 피처

- `candidate`: 후보 번호
- `range_idx`: 범위 인덱스 (0-4)
- `pos4_freq`: ord4 위치 출현 빈도
- `is_hot`, `is_cold`: HOT/COLD 비트 여부
- `is_prime`: 소수 여부
- `overall_freq`: 전체 출현 빈도
- `in_recent_3`: 최근 3회 출현 여부
- `formula_distance`: 공식 예측값과의 거리

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
| Top-1 | 8.8% |
| Top-5 | 30.5% |
| Top-10 | 49.8% |
