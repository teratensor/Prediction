# 첫수끝수(FirstEnd) 인사이트

당첨번호의 최소값(첫수)과 최대값(끝수) 분석

## 실행

```bash
# 통계 생성
python generate.py

# 점수 기반 쌍 선택
python select_pairs.py

# 백테스트
python backtest_pairs.py

# 비율별 적중률 테스트
python backtest_threshold.py
```

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| first_distribution.csv | 첫수(최소값) 분포 |
| end_distribution.csv | 끝수(최대값) 분포 |
| span_distribution.csv | 범위(끝수-첫수) 분포 |
| pair_distribution.csv | 첫수-끝수 조합 분포 |
| summary.csv | 요약 통계 (평균, 최소, 최대) |
| selected_pairs.csv | 점수 기반 선택된 (첫수,끝수) 쌍 |
| backtest_result.csv | 백테스트 결과 |

## CSV 컬럼

- **value/span**: 값
- **frequency**: 출현 횟수
- **ratio**: 비율 (%)
- **probability**: 확률 (0~1)

## 점수 기반 쌍 선택 (0단계)

1. 첫수 빈도 점수 (first_distribution.csv)
2. 끝수 빈도 점수 (end_distribution.csv)
3. (첫수,끝수) 쌍 빈도 점수 (pair_distribution.csv)
4. 최근 3회 출현 번호는 점수 차감 (-5점)
5. **상위 75% 조합 선택**

## 백테스트 결과

| 비율 | 적중률 | 평균 선택 수 |
|------|--------|-------------|
| 50% | 68.1% | 172개 |
| 60% | 76.6% | 223개 |
| 70% | 83.6% | 283개 |
| **75%** | **88.1%** | **318개** |
| 80% | 90.6% | 357개 |
| 90% | 95.4% | 455개 |

**현재 설정: 75% (적중률 88.1%, 평균 318개 쌍)**
