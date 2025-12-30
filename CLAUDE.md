# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

로또 6/45 당첨번호 예측 시스템 - ML 기반 파이프라인

**핵심 목표: 6개 일치** - 5개 이하 적중은 의미 없음.

## 실행 방법

```bash
# 단일 회차 예측 (기본: Top-100 firstend, Top-10 inner)
python main.py --round 1205

# 파라미터 조정
python main.py --round 1205 --top_firstend 150 --top_inner 15
```

## 아키텍처

```
main.py              ← ML 예측 파이프라인 (진입점)
1_data/              → winning_numbers.csv (당첨번호 827~1204회차)
2_insights/          → 8개 통계 인사이트
3_predict/           → ML 모델 (ord별 예측)
  ├─ 1_ord1ord6_ml/  → (ord1, ord6) 쌍 예측 (앙상블)
  ├─ 2_ord2_ml/      → ord2 예측
  ├─ 3_ord3_ml/      → ord3 예측
  ├─ 4_ord4_ml/      → ord4 예측
  └─ 5_ord5_ml/      → ord5 예측
```

## ML 예측 파이프라인

| 순서 | 모델 | Top-K | 비고 |
|------|------|-------|------|
| 1 | ord1, ord6 | Top-100 | v2 앙상블 (XGBoost 70% + 빈도 30%) |
| 2 | ord4 | Top-10 | ord4 = ord1 + (ord6-ord1) × 0.60 기준 |
| 3 | ord2 | Top-10 | ord1 < ord2 < ord4 |
| 4 | ord3 | Top-10 | ord2 < ord3 < ord4 |
| 5 | ord5 | Top-10 | ord4 < ord5 < ord6 |

### 의존성 구조

```
ord1, ord6 (820개 쌍) → ord4 → ord2 → ord3 → ord5
```

## 모델 성능 (378회차 백테스트)

| 모델 | Top-10 적중률 | 비고 |
|------|---------------|------|
| ord1,ord6 | 59.9% (Top-100) | 가장 중요, v2 앙상블 사용 |
| ord2 | 68.4% | |
| ord3 | 70.2% | |
| ord4 | 49.8% | 가장 어려움 |
| ord5 | 79.0% | 가장 쉬움 (30-39 구간 집중) |

## 핵심 개념

| 용어 | 설명 |
|------|------|
| ord | 오름차순 정렬된 위치 (ord1=최소, ord6=최대) |
| ball | 실제 로또 공 번호 (1~45) |
| firstend | (ord1, ord6) 쌍 - 820개 가능 |
| span | ord6 - ord1 (범위) |

## 인사이트 (2_insights/)

ML 피처로 사용되는 통계:

| 폴더 | 용도 |
|------|------|
| 4_range | 포지션별 최빈 구간 |
| 5_prime | 소수 빈도 |
| 7_onehot | HOT/COLD 비트 |

## 포지션별 최빈 구간

| 포지션 | 최빈 구간 | 비율 |
|--------|-----------|------|
| ord2 | 10-19 | 48.3% |
| ord3 | 15-24 | - |
| ord5 | 30-39 | 54.6% |

## 데이터 형식

**winning_numbers.csv**: round, ord1-6, ball1-6, o1-o45

## DB 연결

MCP MySQL: `192.168.45.113` / `lottoda_wp.lotto_data`
