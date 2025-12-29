# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

로또 6/45 당첨번호 예측 시스템 - 3단계 파이프라인 기반 조합 생성

**중요: 6개 일치만 가치가 있음** - 5개 이하 적중은 의미 없음. 모든 개선은 6개 일치 확률 향상에 집중할 것.

## 실행 방법

```bash
# 통합 예측 실행 (권장)
python main.py --round 1205

# 백테스트
python 4_backtest/backtest_full.py --start 900 --end 1000
python 4_backtest/backtest_full.py --round 1205

# 병목 분석
python 4_backtest/why.py --start 900 --end 1000
```

## 아키텍처

```
main.py           ← 통합 예측 파이프라인 (권장 진입점)
1_data/           → 당첨번호 데이터 (winning_numbers.csv)
2_insights/       → 8개 인사이트 (1_consecutive ~ 8_ac)
3_predict/        → 개별 단계 스크립트 (레거시)
4_backtest/       → 백테스트 및 분석 도구
result/           → 예측 결과 저장
```

## 3단계 예측 파이프라인 (main.py)

| 단계 | 기능 | 출력 |
|------|------|------|
| 1_firstend | (ord1, ord6) 쌍 생성 | 477개 쌍 |
| 2_ord4 | ord4 계산 (공식 A + ±10 오프셋) | ~10,000개 행 |
| 3_ord235 | ord2, ord3, ord5 채우기 (Top-15) | ~5,600,000개 조합 |

### ord4 예측 공식

`ord4 = ord1 + (ord6 - ord1) × 0.60`

## 인사이트 시스템

인사이트는 **동적으로 로드**됨 (하드코딩 아님):

| 인사이트 | 로드 파일 | 사용 |
|----------|-----------|------|
| 7_onehot | `statistics/hot_cold_bits.csv` | HOT/COLD 비트 점수 |
| 5_prime | `statistics/prime_frequency.csv` | 소수 보너스 |
| 4_range | `statistics/position_range_distribution.csv` | 포지션별 최빈 구간 |

```python
# main.py의 load_insights()가 자동으로 로드
insights = load_insights()  # HOT, COLD, primes, optimal_ranges
```

## 핵심 개념

| 용어 | 설명 |
|------|------|
| ball | 실제 로또 공 번호 (1~45), 오름차순 정렬 |
| ord | 출현빈도 순위 (1~45), 1이 가장 빈번 |
| firstend | (ord1, ord6) 쌍 - 첫수와 끝수 |
| seen/unseen | 학습 데이터에 출현한/안 한 번호 |

## DB 연결

MCP MySQL 도구 사용:
- 서버: `192.168.45.113`
- DB: `lottoda_wp.lotto_data`

## 데이터 형식

**winning_numbers.csv**: round, ord1-6, ball1-6, o1-o45

**result.csv**: ord1-6, 빈도수, 회차, offset
