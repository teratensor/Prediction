# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

로또 6/45 당첨번호 예측 시스템 - 빈도 기반 세그먼트 예측 + 인사이트 점수화

## 실행 방법

```bash
# 예측 파이프라인 (순차 실행)
python 3_predict/1_firstend/export_result.py     # ord1, ord6 쌍 생성
python 3_predict/2_146/fill_ord4.py              # ord4 계산 (공식 A, B, C)
python 3_predict/3_235/generate_v2.py --round 1205 --count 500  # 조합 생성

# 백테스트
python 4_backtest/backtest.py --round 1204       # 단일 회차
python 3_predict/3_235/backtest_v3.py --start 900  # 조합 백테스트 (최고 성능)
```

## 아키텍처

```
1_data/        → DB에서 데이터 수집 (winning_numbers.csv)
2_insights/    → 패턴 분석 및 통계 (13개 인사이트)
3_predict/     → 예측 파이프라인
  ├── 1_firstend/  → ord1, ord6 선택 (477개 쌍)
  ├── 2_146/       → ord4 계산 (1,431개 행)
  └── 3_235/       → ord2, ord3, ord5 채우기 + 조합 생성
4_backtest/    → 백테스트 및 결과 검증
result/        → 예측 결과 저장 (result.csv)
```

**데이터 흐름**: 1_data → 2_insights → 3_predict → 4_backtest

## 예측 파이프라인 (3_predict)

| 단계 | 폴더 | 입력 | 출력 |
|------|------|------|------|
| 1 | 1_firstend | winning_numbers.csv | result.csv (ord1, ord6 쌍 477개) |
| 2 | 2_146 | result.csv | result.csv (ord4 추가, 1,431개 행) |
| 3 | 3_235 | result.csv | 완성된 6개 번호 조합 |

### ord4 예측 공식 (2_146)
| 공식 | 비율 | 적중률 |
|------|------|--------|
| A | 0.60 | 74.7% |
| B | 0.31 | 15.3% |
| C | 0.87 | 12.9% |

계산식: `ord4 = ord1 + (ord6 - ord1) × 비율`

## DB 연결

MCP MySQL 도구 사용 (1_data/fetch.py 참조):
- 서버: `192.168.45.113`
- DB: `lottoda_wp.lotto_data`

## 핵심 개념

### ball vs ord (중요)

| 구분 | ball | ord |
|------|------|-----|
| 정의 | 실제 로또 공 번호 (1~45) | 출현빈도 순위 (1~45) |
| 정렬 | 오름차순 (ball1 < ball6) | 빈도순 (ord1이 가장 빈번) |
| 고정성 | 불변 | 회차마다 변동 |

### 용어 정리

| 용어 | 설명 |
|------|------|
| o값 | ord→ball 변환 테이블 (회차별 상이) |
| Top24/Mid14/Rest7 | 45개 번호를 빈도순 24/14/7개 분할 |
| shortcode | 6자리 코드 (예: 321411) - 세그먼트별 적중 수 |

## 백테스트 결과 비교 (305회차)

| 전략 | 5개+ 적중 | 4개+ 적중 |
|------|----------|----------|
| 기존 방식 | 0% | 8.9% |
| 혼합 전략 | 0.7% | 10.2% |
| **v3 (최고)** | **1.3%** | **11.1%** |

v3 핵심: 혼합 전략(Top24+Rest21) + firstend 50/50 전략

## 데이터 형식

**winning_numbers.csv**: round, ord1-6, ball1-6, o1-o45 컬럼

**result/result.csv**: ord1-6, 빈도수, 회차, 공식 컬럼
