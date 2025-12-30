# CLAUDE.md (v1.01)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

로또 6/45 당첨번호 예측 시스템 - 3단계 파이프라인 기반 조합 생성

**핵심 목표: 6개 일치** - 5개 이하 적중은 의미 없음. 모든 개선은 6개 일치 확률 향상에 집중.

## 실행 방법

```bash
# 단일 회차 예측
python main.py --round 1205

# 백테스트 (범위)
python main.py --backtest --start 900 --end 1000

# 매치 분포 분석 (NEW in v1.01)
python main.py --distribution --start 1195 --end 1204
```

## 출력 파일

| 파일 | 설명 |
|------|------|
| result/result.csv | 단일 회차 예측 조합 |
| result/backtest.csv | 백테스트 결과 (6개 일치 여부) |
| result/match_distribution.csv | 매치 분포 분석 (0~6개 적중 개수/퍼센트) |

## 아키텍처

```
main.py           ← 통합 예측 파이프라인 (진입점)
1_data/           → winning_numbers.csv (당첨번호)
2_insights/       → 8개 인사이트 (통계 생성)
3_predict/        → 개별 단계 스크립트 (레거시)
4_backtest/       → 분석 도구
5_finetune/       → 필터링 최적화
result/           → 예측 결과 저장
```

## 3단계 예측 파이프라인

| 단계 | 기능 | 현재 설정 | 출력 |
|------|------|-----------|------|
| 1_firstend | (ord1, ord6) 쌍 생성 | 전체 477개 | 477개 쌍 |
| 2_ord4 | ord4 계산 | ±10 범위 | ~9,000개 행 |
| 3_ord235 | ord2, ord3, ord5 채우기 | top_n=15 | ~560만개 조합 |

### ord4 공식

```python
ord4 = round(ord1 + (ord6 - ord1) × 0.60) ± 10
```

## 인사이트 동적 로드

`main.py`의 `load_insights()`가 아래 파일에서 자동 로드:

| 인사이트 | 파일 | 용도 |
|----------|------|------|
| HOT/COLD bits | `2_insights/7_onehot/statistics/hot_cold_bits.csv` | 점수 가감 |
| 소수 | `2_insights/5_prime/statistics/prime_frequency.csv` | 소수 보너스 |
| 최빈 구간 | `2_insights/4_range/statistics/position_range_distribution.csv` | 구간 보너스 |

## 점수화 기준 (ord235)

| 요소 | 점수 | 비고 |
|------|------|------|
| seen 빈도 | ×10 | |
| unseen 빈도 | ×2 | |
| 최빈 구간 (seen) | +15 | |
| 최빈 구간 (unseen) | +20 | |
| 핫 비트 | +5 (+8 for ord5) | ord5 강화 |
| 콜드 비트 | -3 (-5 for ord5) | ord5 강화 |
| 최근 3회 | -5 | |
| 소수 | +3 | ord5 제외 |

## 포지션별 최빈 구간

| 포지션 | 최빈 구간 | 비율 |
|--------|-----------|------|
| ord2 | 10-19 | 48.3% |
| ord3 | 15-24 | 조정됨 (평균 19.4) |
| ord5 | 30-39 | 54.6% |
| ord6 | 40-45 | 57.8% |

## 핵심 개념

| 용어 | 설명 |
|------|------|
| ball | 실제 로또 공 번호 (1~45), 오름차순 정렬 |
| ord | ball의 인덱스 (ord1=최소, ord6=최대) |
| firstend | (ord1, ord6) 쌍 - 첫수와 끝수 |
| seen/unseen | 학습 데이터에 출현한/안 한 번호 |

## 최근 성능 (1195~1204회차)

| 지표 | 값 |
|------|-----|
| 6개 일치 포함 | 7/10회 (70%) |
| 평균 조합수 | 562만개 |
| 5개+ 적중 조합 | 0.0% (평균 163개/회차) |

## DB 연결

MCP MySQL 도구 사용:
- 서버: `192.168.45.113`
- DB: `lottoda_wp.lotto_data`

## 데이터 형식

**winning_numbers.csv**: round, ord1-6, ball1-6, o1-o45

**result/backtest.csv**: round, winning, best_match, has_6_match, total_combinations, best_prediction

**result/match_distribution.csv**: round, winning, total_combinations, match_0~6, pct_0~6, best_match, best_rank, best_prediction
