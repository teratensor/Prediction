# Fifth(5번째 번호) 인사이트

★ 가장 중요한 분석 - ball5는 ball6(끝수) 직전 번호로 조합 결정에 핵심

## 실행

```bash
python generate.py
```

## 분석 항목

1. ball5 기본 빈도 분석
2. ball5 구간별 분포 (0-4)
3. ★ ball1~4 합계별 ball5 범위 예측 (핵심!)
4. ball1~4 소수 개수별 ball5 분석
5. ball4-ball5 연속수 분석
6. ball5-ball6 연속수 분석
7. ball5 소수 여부 분석
8. ball5 홀짝 분석
9. ball5 끝자리 분석
10. Shortcode ord_code별 ball5 분석
11. 복합 패턴 분석 (합계구간 + 소수개수 + shortcode)
12. 최근 트렌드 분석

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| ball5_distribution.csv | ball5 빈도 분포 |
| ball5_range_distribution.csv | ball5 구간별 분포 |
| sum14_ball5_analysis.csv | ★ ball1~4 합계별 ball5 |
| prime_count_ball5.csv | 소수 개수별 ball5 |
| ball4_ball5_gap_distribution.csv | ball4-ball5 간격 분포 |
| ball5_ball6_gap_distribution.csv | ball5-ball6 간격 분포 |
| ball5_prime_distribution.csv | ball5 소수 여부 |
| ball5_oddeven_distribution.csv | ball5 홀짝 분포 |
| ball5_lastdigit_distribution.csv | ball5 끝자리 분포 |
| shortcode_ord_ball5.csv | ord_code별 ball5 |
| shortcode_ball_ball5.csv | ball_code별 ball5 |
| complex_pattern_ball5.csv | 복합 패턴별 ball5 |
| ball5_trend_analysis.csv | 최근 트렌드 |
| summary.csv | 요약 통계 |

## 주요 통계

- **평균**: 33.0, **중앙값**: 34, **최빈값**: 33, 34
- **구간3(30-39)**: 54.6% (가장 빈번)
- **구간2(20-29)**: 25.9%
- **소수 비율**: 22.4%
- **ball4-ball5 연속수**: 14.8%
- **ball5-ball6 연속수**: 15.0%

## 핵심 발견 ★

### 1. ball1~4 합계로 ball5 범위 예측
| 합계 범위 | ball5 평균 | 권장 범위 |
|-----------|-----------|----------|
| ~40 | 26.0 | 18-33 |
| 41-50 | 28.9 | 22-35 |
| 51-60 | 31.7 | 25-37 |
| 61-70 | 33.3 | 28-38 |
| 71-80 | 35.3 | 30-39 |
| 81-90 | 36.3 | 31-40 |
| 91~ | 38.7 | 35-41 |

### 2. 소수 개수와 ball5 관계
- ball1~4에 소수가 많을수록 ball5 평균이 낮아짐 (34.8 → 25.0)
- ball1~4에 소수가 3개 이상이면 ball5 소수 확률 급감 (13.2%)

### 3. Shortcode 패턴
- 321: ball5 평균 31.0 (낮음)
- 420, 411: ball5 평균 33.4~33.7 (높음)

### 4. 최근 상승 트렌드
- 37★, 39, 38, 31★, 28
