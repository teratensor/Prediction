# 첫수끝수(FirstEnd) 인사이트

당첨번호의 최소값(첫수)과 최대값(끝수) 분석

## 실행

```bash
python generate.py
```

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| first_distribution.csv | 첫수(최소값) 분포 |
| end_distribution.csv | 끝수(최대값) 분포 |
| span_distribution.csv | 범위(끝수-첫수) 분포 |
| pair_distribution.csv | 첫수-끝수 조합 분포 |
| summary.csv | 요약 통계 (평균, 최소, 최대) |

## CSV 컬럼

- **value/span**: 값
- **frequency**: 출현 횟수
- **ratio**: 비율 (%)
- **probability**: 확률 (0~1)
