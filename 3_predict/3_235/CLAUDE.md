# 235 - ord2, ord3, ord5 채우기

result.csv의 ord1, ord4, ord6이 결정된 상태에서 ord2, ord3, ord5를 채움

**참고**: 현재 main.py에서 통합 처리됨. 이 폴더의 fill_ord235.py는 레거시 버전.

## 파이프라인 위치

```
main.py (통합)
├── 1_firstend   # ord1, ord6 선택 (477개 쌍)
├── 2_ord4       # ord4 계산 (~9,000개 행)
└── 3_ord235     # ord2, ord3, ord5 채우기 ← 현재 위치
```

## 현재 설정 (main.py v1.01)

| 파라미터 | 값 |
|----------|-----|
| top_n | 15 |
| 조합수 | ~560만개 |

## 방식 (점수화 + Top-N)

### 포지션별 최빈 구간

| 포지션 | 최빈 구간 | 비율 |
|--------|-----------|------|
| ord2 | 10-19 | 48.3% |
| ord3 | 15-24 | 조정됨 (평균 19.4) |
| ord5 | 30-39 | 54.6% |
| ord6 | 40-45 | 57.8% |

### 점수화 기준

| 요소 | 점수 | 비고 |
|------|------|------|
| seen 빈도 | ×10 | 해당 포지션에서 출현한 번호 |
| unseen 빈도 | ×2 | 전체 빈도 기반 |
| 최빈 구간 (seen) | +15 | 해당 구간 내 번호 |
| 최빈 구간 (unseen) | +20 | 해당 구간 내 번호 |
| 핫 비트 | +5 (+8 for ord5) | ord5 강화 |
| 콜드 비트 | -3 (-5 for ord5) | ord5 강화 |
| 최근 3회 | -5 | 최근 출현 번호 페널티 |
| 소수 | +3 | ord5 제외 (ball5 소수 18.5%로 낮음) |

### 범위 제약

```
ord1 < ord2 < ord3 < ord4 < ord5 < ord6

ord2 후보: ord1+1 ~ ord4-2
ord3 후보: ord2+1 ~ ord4-1
ord5 후보: ord4+1 ~ ord6-1
```

## 인사이트 동적 로드

main.py에서 아래 파일 자동 로드:
- `2_insights/7_onehot/statistics/hot_cold_bits.csv`
- `2_insights/5_prime/statistics/prime_frequency.csv`
- `2_insights/4_range/statistics/position_range_distribution.csv`
