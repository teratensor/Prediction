# 9_ord2_ml - ord2 예측 ML 모델

8개 인사이트를 피처로 사용한 XGBoost 기반 ord2 예측 모델

## 실행

```bash
# 백테스트 실행
python 2_insights/9_ord2_ml/backtest.py

# 피처 테스트
python 2_insights/9_ord2_ml/features.py
```

## 백테스트 결과 (329회차)

| 지표 | 값 |
|------|-----|
| 정확도 (Top-1) | 11.2% (37회) |
| Top-3 적중률 | 26.4% (87회) |
| Top-5 적중률 | 41.3% (136회) |
| Top-10 적중률 | 69.6% (229회) |
| 평균 오차 | 5.32 |
| 평균 순위 | 7.97 / 32개 |

## 피처 중요도

| 순위 | 피처 | 중요도 | 설명 |
|------|------|--------|------|
| 1 | relative_position | 22.4% | ord2의 상대적 위치 (ord1~ord6 사이) |
| 2 | is_prime | 8.5% | 소수 여부 |
| 3 | pos2_freq | 7.4% | 포지션2 출현 빈도 |
| 4 | sum_contribution | 7.2% | 합계 기여도 |
| 5 | range_prob | 7.0% | 범위 확률 |
| 6 | consecutive_prob | 6.8% | 연속수 확률 |
| 7 | overall_freq | 6.7% | 전체 출현 빈도 |
| 8 | expected_remaining_avg | 6.3% | 남은 합계 평균 |
| 9 | segment | 5.9% | Top24/Mid14/Rest7 세그먼트 |
| 10 | in_recent_3 | 5.7% | 최근 3회 출현 여부 |

## 범위별 성능

| 범위 | 회수 | 정확도 | Top-5 |
|------|------|--------|-------|
| 00-09 | 119회 | 16.8% | 59.7% |
| 10-19 | 157회 | 7.6% | 31.8% |
| 20-29 | 50회 | 10.0% | 30.0% |
| 30-39 | 3회 | 0.0% | 0.0% |

## 8개 인사이트 활용

| 인사이트 | 피처 | 기여도 |
|----------|------|--------|
| 4_range | range_prob, range_idx | 중상 |
| 7_onehot | pos2_freq, is_hot, is_cold | 중상 |
| 3_sum | sum_contribution, expected_remaining_avg | 중상 |
| 6_shortcode | segment | 중간 |
| 5_prime | is_prime | 높음 |
| 1_consecutive | is_consecutive, consecutive_prob | 중간 |
| 2_lastnum | relative_position | 매우 높음 |
| 8_ac | (간접 사용) | 낮음 |

## 파일 구조

| 파일 | 설명 |
|------|------|
| features.py | 8개 인사이트 기반 피처 추출 |
| backtest.py | Rolling Window 백테스트 |
| results/backtest_results.csv | 백테스트 결과 |
| results/feature_importance.csv | 피처 중요도 |

## 개선 방향

1. **relative_position 강화**: 가장 중요한 피처, 더 정밀한 위치 예측 필요
2. **범위 10-19 개선**: 가장 많이 출현하지만 정확도 낮음
3. **is_prime 활용**: 소수가 의외로 높은 중요도
4. **앙상블 모델**: XGBoost + 규칙 기반 모델 조합
