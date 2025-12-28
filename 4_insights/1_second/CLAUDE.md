# Second(2번째 번호) 인사이트

정렬된 당첨번호 중 2번째 번호(ball2) 분석

## 실행

```bash
python generate.py
```

## 분석 항목

1. ball2 기본 빈도 분석
2. ball2 구간별 분포 (0-4)
3. ball1 소수 여부별 ball2 분석
4. ball1-ball2 연속수 분석
5. ball2 자신의 소수 여부 분석
6. ball2 홀짝 분석
7. ball2 끝자리(0-9) 분석
8. ball1별 ball2 분포 분석
9. ball1-ball2 간격별 ball3 관계
10. ball1-ball2 동시 소수 패턴
11. 복합 패턴 분석
12. 최근 트렌드 분석

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| ball2_distribution.csv | ball2 빈도 분포 |
| ball2_range_distribution.csv | ball2 구간별 분포 |
| ball1_prime_ball2.csv | ball1 소수여부별 ball2 |
| ball1_ball2_gap_distribution.csv | ball1-ball2 간격 분포 |
| ball2_prime_distribution.csv | ball2 소수 여부 |
| ball2_oddeven_distribution.csv | ball2 홀짝 분포 |
| ball2_lastdigit_distribution.csv | ball2 끝자리 분포 |
| ball1_ball2_analysis.csv | ball1별 ball2 분석 |
| gap12_ball3_relation.csv | 간격별 ball3 관계 |
| ball1_ball2_prime_pattern.csv | 동시 소수 패턴 |
| complex_pattern_distribution.csv | 복합 패턴 |
| ball2_trend_analysis.csv | 최근 트렌드 |
| summary.csv | 요약 통계 |

## 주요 통계

- **평균**: 13.2, **중앙값**: 12, **최빈값**: 7★
- **구간1(10-19)**: 48.3% (가장 빈번)
- **구간0(01-09)**: 34.6%
- **구간2(20-29)**: 14.0%
- **구간3,4**: 거의 없음 (3.2%, 0%)
- **소수 비율**: 34.3%
- **ball1-ball2 연속수**: 12.4%

## 핵심 발견

### 1. ball1 소수 여부가 ball2에 미치는 영향
| ball1 상태 | ball2 평균 | ball2 소수 비율 |
|-----------|-----------|----------------|
| 소수 | 11.8 | 36.6% |
| 비소수 | 14.4 | 32.4% |

→ ball1이 소수이면 ball2가 더 낮고, ball2도 소수일 확률이 높음

### 2. ball1-ball2 동시 소수 패턴
- NN (둘다 비소수): 36.9%
- PN (ball1만 소수): 28.8%
- NP (ball2만 소수): 17.7%
- PP (둘다 소수): 16.6%

### 3. ball2 끝자리 분포
- 가장 빈번: 6 (14.5%)
- 그 다음: 4 (11.9%), 7 (10.6%), 1 (10.3%), 3 (10.3%)

### 4. 최근 상승 트렌드
- 9, 13★, 16, 12 상승중
