# Third(3번째 번호) 인사이트

정렬된 당첨번호 중 3번째 번호(ball3) 분석

## 실행

```bash
python generate.py
```

## 분석 항목

1. ball3 기본 빈도 분석
2. ball3 구간별 분포 (01-09, 10-19, 20-29, 30-39, 40-45)
3. ball2-ball3 연속수 분석
4. ball1, ball2 소수 패턴 분석
5. ball1-ball2 간격과 ball3 관계 분석
6. ball3 홀짝 분석
7. ball3 끝자리(0-9) 분석
8. (ball1, ball2) 조합별 ball3 분석
9. 복합 조건 분석
10. 최근 N회 트렌드 분석

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| ball3_distribution.csv | ball3 빈도 분포 |
| ball3_range_distribution.csv | ball3 구간별 분포 |
| ball2_ball3_gap_distribution.csv | ball2-ball3 간격 분포 |
| prime_pattern_distribution.csv | ball1,ball2 소수 패턴 |
| gap12_ball3_relation.csv | ball1-ball2 간격과 ball3 관계 |
| ball3_oddeven_distribution.csv | ball3 홀짝 분포 |
| ball3_lastdigit_distribution.csv | ball3 끝자리(0-9) 분포 |
| pair12_ball3_analysis.csv | (ball1,ball2) 조합별 ball3 분석 |
| complex_pattern_distribution.csv | 복합 조건 분석 |
| ball3_trend_analysis.csv | 최근 트렌드 분석 |
| summary.csv | 요약 통계 |

## 주요 통계

- **평균**: 20.2, **중앙값**: 19, **최빈값**: 12
- **구간1(10-19)**: 45.4% (가장 빈번)
- **구간2(20-29)**: 36.7%
- **구간0,3,4**: 합계 18%
- **ball2-ball3 연속수**: 10.8%

## 핵심 발견

### 1. ball3 구간 분포
- 82.1%가 10-29 범위 집중
- 구간0(01-09): 5.0%
- 구간4(40-45): 0.3% (거의 없음)

### 2. ball1-ball2 소수 패턴
- NN (둘다 비소수): 36.9%
- PN (ball1만 소수): 28.8%
- NP (ball2만 소수): 17.7%
- PP (둘다 소수): 16.6%

### 3. ball3 홀짝
- 홀수: 48.3%
- 짝수: 51.7%

### 4. ball3 끝자리 상위
- 0: 10.3%
- 2, 7, 8, 9: 각 9~10%
