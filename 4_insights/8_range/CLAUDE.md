# Range(구간) 인사이트

당첨번호의 구간별 분포 분석

## 구간 정의

| 구간 | 범위 | 번호 수 |
|------|------|--------|
| 0 | 1-9 | 9개 |
| 1 | 10-19 | 10개 |
| 2 | 20-29 | 10개 |
| 3 | 30-39 | 10개 |
| 4 | 40-45 | 6개 |

## 실행

```bash
python generate.py
```

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| range_frequency.csv | 구간별 출현 빈도 |
| range_code_distribution.csv | 6자리 구간 코드 분포 |
| range_pattern_distribution.csv | 구간 분포 패턴 (각 구간별 개수) |
| unique_ranges_distribution.csv | 사용 구간 수 분포 |
| consecutive_ranges_analysis.csv | 연속 구간 패턴 |
| position_range_distribution.csv | 포지션별 구간 분포 |
| range_skip_pattern.csv | 건너뛴 구간 패턴 |
| range_code_trend.csv | 최근 트렌드 분석 |
| range_oddeven_distribution.csv | 구간별 홀짝 분포 |
| range_code_theory_vs_actual.csv | 이론 vs 실제 빈도 |
| summary.csv | 요약 통계 |

## 코드 형식

**6자리 구간 코드** (예: `011234`)
- 각 자리: ball1~ball6이 속한 구간
- 예: `011234` = ball1은 구간0, ball2는 구간1, ball3은 구간1, ...

**구간 분포 패턴** (예: `1-2-1-1-1`)
- 각 구간별 번호 개수
- 순서: 구간0-구간1-구간2-구간3-구간4

## 주요 통계

- 4개 구간 사용: 51.2% (가장 빈번)
- 57.5% 연속 구간 사용 (건너뛰지 않음)
- 구간3(30-39) 이론보다 +6.3% 과다 출현
