# 235 인사이트 (ord2, ord3, ord5)

둘째수(ord2), 셋째수(ord3), 다섯째수(ord5) 채우기

## 파이프라인 위치

```
2_predict/
├── 1_firstend/   # ord1, ord6 선택 (477개 쌍)
├── 2_146/        # ord4 계산 (1,431개 행)
└── 3_235/        # ord2, ord3, ord5 채우기 ← 현재 위치
```

## 실행

```bash
# 1. firstend에서 result.csv 생성
python ../1_firstend/export_result.py

# 2. 146에서 ord4 채우기
python ../2_146/fill_ord4.py

# 3. 조합 생성 (ord2, ord3, ord5 채우기)
python generate_combinations.py --round 1205 --count 500

# 4. 백테스트
python backtest_v3.py --start 900 --count 500
```

## 입출력

**입력**: `result/result.csv` (ord1, ord4, ord6 결정됨)

**출력**: 완성된 6개 번호 조합

## 적용 인사이트 (12개)

| 인사이트 | 가중치 | 조건 |
|----------|--------|------|
| firstend | 10점 | ball1: 1-10, ball6: 38-45 |
| second | 8점 | ball2: 10-19 (48.3%) |
| third | 8점 | ball3: 10-19 (45.4%) |
| fourth | 8점 | ball4: 20-29 (42.7%) |
| fifth | 10점 | ball5: 30-39 (54.6%) |
| consecutive | 8점 | 연속수 1쌍 (40.9%) |
| lastnum | 10점 | ball6: 40-45 (57.8%) |
| sum | 10점 | 합계 121-160 (48.5%) |
| range | 8점 | 4개 구간 사용 (51.2%) |
| prime | 10점 | 소수 1-2개 (66.7%) |
| shortcode | 10점 | Top24에서 3-4개 |
| onehot | 2점/개 | 핫 비트 보너스 |

## 파일 구조

| 파일 | 설명 |
|------|------|
| generate_combinations.py | 기존 방식 (빈도 상위만) |
| generate_mixed.py | 혼합 전략 (Top24+Rest21) |
| generate_v2.py | v2 방식 (firstend+점수화) |
| backtest_combinations.py | 기존 방식 백테스트 |
| backtest_mixed.py | 혼합 전략 백테스트 |
| backtest_v2.py | v2 백테스트 (50/50) |
| backtest_v3.py | v3 백테스트 (최고 성능) |
| analyze_exclusions.py | 인사이트별 탈락 분석 |
| analyze_winning_ranks.py | 당첨번호 빈도순위 분석 |

## 백테스트 결과 비교 (305회차)

| 지표 | 혼합 전략 | v2 (50/50) | **v3 (최고)** |
|------|----------|------------|---------------|
| 5개+ 적중 | 2회 (0.7%) | 1회 (0.3%) | **4회 (1.3%)** |
| 4개+ 적중 | 31회 (10.2%) | 26회 (8.5%) | **34회 (11.1%)** |
| 3개+ 적중 | 165회 (54.1%) | 142회 (46.6%) | 158회 (51.8%) |
| 첫수 적중 | - | 52.1% | **58.4%** |
| 끝수 적중 | - | 30.8% | **41.6%** |

## 핵심 전략

### v3 (혼합전략 + firstend 50/50)
1. **혼합 전략**: Top24에서 3-4개 + Rest21에서 2-3개
2. **50/50 전략**: 학습 데이터 출현 쌍 50% + 미출현 가능성 쌍 50%
3. **firstend 우선**: ball1/ball6 후보에 firstend 확률 반영

## 출력 파일

| 파일 | 설명 |
|------|------|
| combinations_round{N}.csv | 생성된 조합 |
| backtest_combination_results.csv | 백테스트 상세 결과 |
| backtest_summary.csv | 백테스트 요약 |
