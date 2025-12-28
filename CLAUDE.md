# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

로또 6/45 당첨번호 예측 시스템 - 최근 10회 빈도 기반 Top24/Mid14/Rest7 세그먼트 예측

## 중요: 실행 전 항상 확인 요청

## 실행 방법

```bash
# 예측 실행
python 2_predict/predict.py --round 1205

# 누적 백테스트 (826회차부터)
python 3_backtest/backtest.py

# 특정 회차 백테스트
python 3_backtest/backtest.py --round 1204

# 인사이트 생성
python 4_insights/sum/generate.py
python 4_insights/range/generate.py
python 4_insights/1_firstend/generate.py
python 4_insights/2_third/generate.py

# 첫수끝수 백테스트
python 4_insights/1_firstend/backtest_pairs.py
python 4_insights/1_firstend/backtest_threshold.py
```

## 아키텍처

```
1_data/        → DB에서 데이터 수집 (winning_numbers.csv)
2_predict/     → 예측 알고리즘 (최근 10회 빈도 기반)
3_backtest/    → 백테스트 및 결과 검증
4_insights/    → 패턴 분석 및 통계
  ├── 1_firstend/  → 첫수/끝수 분석 (75% threshold, 88.1% 적중)
  ├── 2_third/     → 3번째 번호 분석
  ├── sum/         → 6개 번호 합계 분석 (권장 118-160)
  ├── range/       → 구간 코드 분석 (0-4)
  ├── prime/       → 소수 패턴 분석
  └── shortcode/   → 세그먼트 코드 분석
5_combination/ → 조합 수 계산
```

**데이터 흐름**: 1_data → 2_predict → 3_backtest → 4_insights

## DB 연결

MCP MySQL 도구 사용 (fetch.py 참조):
- 서버: `192.168.45.113`
- DB: `lottoda_wp.lotto_data`

```python
from 1_data.fetch import save
save(data)  # data: [{"회차": 1, "ord1": 10, "ball1": 5, "o1": 5, ...}, ...]
```

## 핵심 개념

| 용어 | 설명 |
|------|------|
| ord | 출현빈도 순위 (1~45, 1이 가장 빈번) |
| ball | 실제 로또 공 번호 (1~45) |
| o값 | ord→ball 변환 테이블 (회차별 상이) |
| Top24/Mid14/Rest7 | 45개 번호를 점수순으로 24/14/7개 분할 |
| range 구간 | 0(1-9), 1(10-19), 2(20-29), 3(30-39), 4(40-45) |

## 요약 코드 형식

`ABCDEF` (예: 321411)
- ABC: Ord 세그먼트별 적중 (Top24/Mid14/Rest7)
- DEF: Ball 세그먼트별 적중 (Top24/Mid14/Rest7)

예: `321` = Top24에서 3개, Mid14에서 2개, Rest7에서 1개 적중

## 데이터 형식

**winning_numbers.csv**: round, ord1-6, ball1-6, o1-o45 컬럼

## 예측 전략

**핵심 원칙**: 필터링이 아닌 점수 기반 (필터 10개 중 1개만 벗어나도 당첨번호 제외되는 문제 방지)

### 0단계: 첫수/끝수 점수 기반 선택
1. 첫수 빈도 점수 (first_distribution.csv)
2. 끝수 빈도 점수 (end_distribution.csv)
3. (첫수,끝수) 쌍 빈도 점수 (pair_distribution.csv)
4. 최근 3회 출현 번호는 점수 차감
5. **상위 75% 조합 선택** (적중률 88.1%, 평균 318개)

### 1단계: ord1, ord6 픽업
- 가장 빈번한 번호(ord1)와 가장 드문 번호(ord6) 먼저 선택
- 이것만으로도 조합 수 대폭 감소

### 2단계: shortcode 인사이트 검증
- ord1, ord6 + 나머지 4개 조합 생성
- shortcode 패턴(예: 321, 411 등)과 매칭 확인
- 자주 나오는 shortcode 패턴에 해당하는 조합만 유지

### 3단계: 앙상블
- 여러 인사이트(sum, range, prime 등)로 점수 부여
- 필터가 아닌 가중치 방식으로 랭킹
- 상위 N개 조합 선택
