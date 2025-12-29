# Lastnum(끝수/6번째 번호) 인사이트

당첨번호의 끝수(최대값, ball6) 분석

## 실행

```bash
python generate.py
```

## 분석 항목

1. ball6 기본 빈도 분석
2. ball6 구간별 분포 (0-4)
3. ball5-ball6 연속수 분석
4. ball6 소수 여부 분석
5. ball6 홀짝 분석
6. ball6 끝자리(0-9) 분석
7. ball1~5 합계와 ball6 관계
8. 범위(span = ball6 - ball1) 분석
9. 복합 패턴 분석
10. 최근 트렌드 분석

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| ball6_distribution.csv | ball6 빈도 분포 |
| ball6_range_distribution.csv | ball6 구간별 분포 |
| ball5_ball6_gap_distribution.csv | ball5-ball6 간격 분포 |
| ball6_prime_distribution.csv | ball6 소수 여부 |
| ball6_oddeven_distribution.csv | ball6 홀짝 분포 |
| ball6_lastdigit_distribution.csv | ball6 끝자리 분포 |
| sum15_ball6_relation.csv | ball1~5 합계와 ball6 관계 |
| span_distribution.csv | 범위 분포 |
| complex_pattern_distribution.csv | 복합 패턴 |
| ball6_trend_analysis.csv | 최근 트렌드 |
| summary.csv | 요약 통계 |

## 주요 통계

- **평균**: 39.4, **중앙값**: 41, **최빈값**: 45
- **구간4(40-45)**: 57.8% (가장 빈번)
- **구간3(30-39)**: 35.9%
- **구간0,1,2**: 합계 6.3%
- **소수 비율**: 24.8%
- **ball5-ball6 연속수**: 15.0%
- **평균 범위(span)**: 32.3

## ball6 구간별 분포

| 구간 | 비율 |
|------|------|
| 0(01-09) | 0.0% |
| 1(10-19) | 0.8% |
| 2(20-29) | 5.5% |
| 3(30-39) | 35.9% |
| **4(40-45)** | **57.8%** |

## 핵심 발견

### 1. ball6 빈도 상위
- 45: 14.5% (최다)
- 44: 12.4%
- 43★: 9.0%
- 42: 8.4%

### 2. 범위(span) 분포
- 평균 범위: 32.3
- 가장 빈번: 37 (6.6%)
- 31~38 범위가 대부분

### 3. 홀짝 분포
- 홀수: 53.6%
- 짝수: 46.4%

### 4. 최근 상승 트렌드
- 39, 40, 32, 28 상승중
