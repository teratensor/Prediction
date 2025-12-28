# Fourth(4번째 번호) 인사이트

정렬된 당첨번호 중 4번째 번호(ball4) 분석

## 실행

```bash
python generate.py
```

## 분석 항목

1. ball4 기본 빈도 분석
2. ball4 구간별 분포 (0-4)
3. ball1,2,3 소수 개수별 ball4 분석
4. ball6(끝수) 소수 여부별 ball4 분석
5. ball3-ball4 연속수 분석
6. ball4 자신의 소수 여부 분석
7. ball4 홀짝 분석
8. ball4 끝자리(0-9) 분석
9. ball4 "적당한 수" 분석 (중앙값 ± 10)
10. ball1,2,3,6 복합 패턴과 ball4 관계
11. 최근 트렌드 분석

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| ball4_distribution.csv | ball4 빈도 분포 |
| ball4_range_distribution.csv | ball4 구간별 분포 |
| ball123_prime_count_ball4.csv | ball1,2,3 소수개수별 ball4 |
| ball6_prime_ball4.csv | ball6 소수여부별 ball4 |
| ball3_ball4_gap_distribution.csv | ball3-ball4 간격 분포 |
| ball4_prime_distribution.csv | ball4 소수 여부 |
| ball4_oddeven_distribution.csv | ball4 홀짝 분포 |
| ball4_lastdigit_distribution.csv | ball4 끝자리 분포 |
| ball4_appropriate_distribution.csv | ball4 적당한수 분포 |
| complex_pattern_ball4.csv | 복합 패턴별 ball4 |
| ball4_trend_analysis.csv | 최근 트렌드 |
| summary.csv | 요약 통계 |

## 주요 통계

- **평균**: 26.5, **중앙값**: 27
- **구간2(20-29)**: 42.7% (가장 빈번)
- **구간3(30-39)**: 34.8%
- **소수 비율**: 25.6%
- **적당한 수(17-37)**: 81.8%
- **ball3-ball4 연속수**: 12.1%

## 핵심 발견

- ball1,2,3에 소수가 많을수록 ball4 평균이 낮아짐
- ball1,2,3에 소수가 많을수록 ball4도 소수일 확률 증가 (22% → 36%)
- ball6 소수 여부는 ball4에 영향 없음
