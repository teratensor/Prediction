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
```

## 아키텍처

```
1_data/        → DB에서 데이터 수집 (winning_numbers.csv)
2_predict/     → 예측 알고리즘 (최근 10회 빈도 기반)
3_backtest/    → 백테스트 및 결과 검증
4_insights/    → 패턴 분석 및 통계
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

## 요약 코드 형식

`ABCDEF` (예: 321411)
- ABC: Ord 세그먼트별 적중 (Top24/Mid14/Rest7)
- DEF: Ball 세그먼트별 적중 (Top24/Mid14/Rest7)

예: `321` = Top24에서 3개, Mid14에서 2개, Rest7에서 1개 적중

## 데이터 형식

**winning_numbers.csv**: round, ord1-6, ball1-6, o1-o45 컬럼
