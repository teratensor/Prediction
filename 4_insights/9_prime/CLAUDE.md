# 소수(Prime) 인사이트

당첨번호 중 소수 개수 분석

## 소수 목록 (1-45 범위)

2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43 (총 14개)

## 실행

```bash
python generate.py
```

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| count_distribution.csv | 소수 개수별 분포 (0~5개) |
| prime_frequency.csv | 각 소수별 출현 빈도 |
| summary.csv | 요약 통계 (평균, 최소, 최대) |

## CSV 컬럼

- **count/prime**: 소수 개수 또는 소수 값
- **frequency**: 출현 횟수
- **ratio**: 비율 (%)
- **probability**: 확률 (0~1)
