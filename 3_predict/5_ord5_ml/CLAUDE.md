# ord5 예측 ML 모델

## 개요
ord1, ord4, ord6가 주어졌을 때 ord5를 예측하는 ML 모델.

## 제약 조건
- ord4 < ord5 < ord6
- ord5 최빈 범위: 30-39 (54.6%)

## 주요 피처

- `candidate`: 후보 번호
- `range_idx`: 범위 인덱스 (0-4)
- `pos5_freq`: ord5 위치 출현 빈도
- `is_hot`, `is_cold`: HOT/COLD 비트 여부 (강화 적용)
- `is_prime`: 소수 여부
- `overall_freq`: 전체 출현 빈도
- `in_recent_3`: 최근 3회 출현 여부
- `is_in_optimal_range`: 최빈 범위(30-39) 포함 여부

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
| Top-1 | 17.5% |
| Top-5 | 56.4% |
| Top-10 | 79.0% |

## 특이사항

- ord5는 가장 예측하기 쉬운 위치 (Top-10 79%)
- 30-39 범위에 54.6% 집중
