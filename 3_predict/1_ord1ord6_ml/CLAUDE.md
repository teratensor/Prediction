# ord1, ord6 쌍 예측 ML 모델

## 개요
(ord1, ord6) 쌍을 예측하는 ML 모델. 첫수(ord1)와 끝수(ord6)를 동시에 예측.

## 모델 버전

| 버전 | 설명 | Top-100 적중률 |
|------|------|----------------|
| v1 | XGBoost 단독 | 54.2% |
| **v2** | **앙상블 (XGBoost 70% + 빈도 30%)** | **59.9%** |
| v3 | 개별 예측 후 조합 | 50.5% |

## 주요 피처

- `pair_freq`: (ord1, ord6) 쌍 출현 빈도
- `ord1_freq`, `ord6_freq`: 개별 위치 빈도
- `span`: ord6 - ord1 (범위)
- `span_freq`: 범위 빈도
- `ord1_trend`, `ord6_trend`: 최근 트렌드
- `ord1_gap`, `ord6_gap`: 마지막 출현 이후 간격

## 실행 방법

```bash
# v2 앙상블 백테스트 (권장)
python backtest_v2.py
```

## 결과 파일

- `results/backtest_results_v2.csv`: 백테스트 결과
- Top-100, Top-150, Top-200 적중률 포함

## 성능 (v2 앙상블, 378회차)

| 지표 | 값 |
|------|-----|
| Top-30 | 27.4% |
| Top-50 | 42.1% |
| Top-100 | 59.9% |
| Top-150 | 72.0% |
| Top-200 | 78.7% |
| 평균 순위 | 121.0 |
