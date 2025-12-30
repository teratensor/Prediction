# 2_insights - 인사이트 분석

당첨번호 패턴 분석 및 통계 생성

## 8개 인사이트

| # | 폴더 | 설명 | main.py 적용 |
|---|------|------|-------------|
| 1 | 1_consecutive | 연속수 패턴 | - |
| 2 | 2_lastnum | ball6 간격 | - |
| 3 | 3_sum | 합계 분포 | - |
| 4 | 4_range | 포지션별 구간 분포 | optimal_ranges |
| 5 | 5_prime | 소수 분포 | primes |
| 6 | 6_shortcode | Top24/Rest21 분석 | - |
| 7 | 7_onehot | HOT/COLD 비트 | hot_bits, cold_bits |
| 8 | 8_ac | AC값 분포 | - |

## main.py 인사이트 로드

`load_insights()` 함수가 아래 파일에서 자동 로드:

| 인사이트 | 파일 |
|----------|------|
| HOT/COLD bits | `7_onehot/statistics/hot_cold_bits.csv` |
| 소수 | `5_prime/statistics/prime_frequency.csv` |
| 최빈 구간 | `4_range/statistics/position_range_distribution.csv` |

## 점수화 적용 (ord235)

| 요소 | 점수 | 비고 |
|------|------|------|
| 핫 비트 | +5 (+8 for ord5) | 7_onehot |
| 콜드 비트 | -3 (-5 for ord5) | 7_onehot |
| 소수 | +3 | 5_prime (ord5 제외) |
| 최빈 구간 (seen) | +15 | 4_range |
| 최빈 구간 (unseen) | +20 | 4_range |

## 포지션별 최빈 구간 (v1.01)

| 포지션 | 최빈 구간 | 비율 |
|--------|-----------|------|
| ord2 | 10-19 | 48.3% |
| ord3 | 15-24 | 조정됨 (평균 19.4) |
| ord5 | 30-39 | 54.6% |
| ord6 | 40-45 | 57.8% |

## 구조

각 인사이트는 `{name}/` 폴더로 구성:
- `generate.py`: 인사이트 생성 로직
- `statistics/`: CSV 통계 파일
- `CLAUDE.md`: 설명 문서
