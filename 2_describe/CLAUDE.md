# 2_describe - 회차별 데이터 분석

각 회차의 당첨번호를 115개 컬럼으로 분석한 describe.csv 생성

## 실행

```bash
python 2_describe/main.py
```

## 출력

- `2_describe/describe.csv` - 379개 회차 × 115개 컬럼
- `2_describe/insights.csv` - 108개 통계

## 컬럼 목록

### 기본 정보 (8개)
| 컬럼 | 설명 |
|------|------|
| round | 회차 |
| ord1~ord6 | 정렬된 번호 |
| ord_bonus | 보너스 번호 |

### 범위/구간 (10개)
| 컬럼 | 설명 |
|------|------|
| span | ord6 - ord1 |
| sum6 | 6개 합계 |
| avg6 | 6개 평균 |
| range_code | 구간 코드 (A=1-9, B=10-19, C=20-29, D=30-39, E=40-45) |
| ord1_range~ord6_range | 각 위치별 구간 |

### 소수 (3개)
| 컬럼 | 설명 |
|------|------|
| prime_count | 소수 개수 |
| prime_positions | 소수 위치 |
| prime_numbers | 소수 번호들 |

### 홀짝 (3개)
| 컬럼 | 설명 |
|------|------|
| odd_count | 홀수 개수 |
| even_count | 짝수 개수 |
| oddeven_pattern | 홀짝 패턴 (O/E) |

### 연속수 (4개)
| 컬럼 | 설명 |
|------|------|
| consecutive_count | 연속수 그룹 수 |
| consecutive_max_len | 최대 연속 길이 |
| consecutive_groups | 연속수 그룹들 |
| consecutive_positions | 연속수 위치 |

### 간격 (6개)
| 컬럼 | 설명 |
|------|------|
| gap_12~gap_56 | 인접 번호 간격 |
| gap_pattern | 간격 패턴 (S=1-2, M=3-5, L=6-10, X=11+) |

### 이월수 (4개)
| 컬럼 | 설명 |
|------|------|
| carryover_count | 이월수 개수 |
| carryover_numbers | 이월된 번호들 |
| carryover_positions_prev | 전회차 위치 |
| carryover_positions_curr | 현회차 위치 |

### AC값 (1개)
| 컬럼 | 설명 |
|------|------|
| ac_value | Arithmetic Complexity (고유 차이값 수 - 5) |

### 끝수 (4개)
| 컬럼 | 설명 |
|------|------|
| last_digit_pattern | 끝자리 패턴 |
| last_digit_unique | 고유 끝자리 수 |
| same_lastdigit_pairs | 동일 끝수 쌍 개수 |
| same_lastdigit_numbers | 동일 끝수 번호들 |

### 구간별 분포 (5개)
| 컬럼 | 설명 |
|------|------|
| count_1to9 ~ count_40to45 | 각 구간별 개수 |

### 보너스 (7개)
| 컬럼 | 설명 |
|------|------|
| bonus_range | 보너스 구간 |
| bonus_is_prime | 보너스가 소수인지 |
| bonus_gap_to_nearest | 가장 가까운 당첨번호와 거리 |
| bonus_odd | 보너스 홀짝 (O/E) |
| bonus_last_digit | 보너스 끝자리 |
| bonus_in_main | 보너스가 본번호 범위 내 |
| bonus_position | 보너스 정렬 위치 |

### 구간 이월 (2개)
| 컬럼 | 설명 |
|------|------|
| range_carryover_count | 전회차와 동일 구간 분포 수 |
| range_carryover_detail | 동일 구간 상세 |

### 보너스 이월 (1개)
| 컬럼 | 설명 |
|------|------|
| prev_bonus_in_curr | 전회차 보너스가 현회차에 포함 |

### 구간 패턴 (4개)
| 컬럼 | 설명 |
|------|------|
| empty_ranges | 비어있는 구간들 |
| empty_range_count | 비어있는 구간 수 |
| dominant_range | 최다 구간 |
| dominant_range_count | 최다 구간 개수 |

### 번호 분포 (3개)
| 컬럼 | 설명 |
|------|------|
| low_count | 저번호(1-22) 개수 |
| high_count | 고번호(23-45) 개수 |
| median | 중앙값 |

### 끝수 이월 (2개)
| 컬럼 | 설명 |
|------|------|
| lastdigit_carryover_count | 전회차와 동일 끝수 개수 |
| lastdigit_carryover_digits | 이월된 끝자리들 |

### 통계적 분석 (6개)
| 컬럼 | 설명 |
|------|------|
| stat_variance | 분산 |
| stat_std | 표준편차 |
| stat_q1 | 1사분위수 |
| stat_q3 | 3사분위수 |
| stat_iqr | 사분위 범위 |
| stat_cv | 변동계수 |

### 왜도 분석 (2개)
| 컬럼 | 설명 |
|------|------|
| stat_skewness | 왜도 값 |
| stat_skew_type | 왜도 유형 (low/balanced/high) |

### 간격 균등도 (3개)
| 컬럼 | 설명 |
|------|------|
| stat_gap_std | 간격 표준편차 |
| stat_gap_uniformity | 균등도 지수 |
| stat_gap_type | 균등도 유형 (uniform/moderate/varied) |

### 피보나치/황금비 (3개)
| 컬럼 | 설명 |
|------|------|
| stat_fib_count | 피보나치 수 개수 |
| stat_fib_numbers | 피보나치 번호들 |
| stat_golden_pairs | 황금비 근사 쌍 개수 |

### 모듈러 패턴 (5개)
| 컬럼 | 설명 |
|------|------|
| stat_mod3_0/1/2 | mod3 분포 |
| stat_mod5_unique | mod5 고유값 수 |
| stat_mod5_complete | mod5 완전성 |

### Shortcode 분석 (11개)
| 컬럼 | 설명 |
|------|------|
| shortcode | 6자리 요약코드 (ord_code + ball_code) |
| ord_code | Ord 3자리: 당첨번호가 Ord 세그먼트에서 어디에 속하는지 |
| ball_code | Ball 3자리: 당첨번호가 Ball 세그먼트에서 어디에 속하는지 |
| sc_top24 | Ord 세그먼트 Top24 개수 |
| sc_mid14 | Ord 세그먼트 Mid14 개수 |
| sc_rest7 | Ord 세그먼트 Rest7 개수 |
| ball_top24 | Ball 세그먼트 Top24 개수 |
| ball_mid14 | Ball 세그먼트 Mid14 개수 |
| ball_rest7 | Ball 세그먼트 Rest7 개수 |
| sc_ord_eq_ball | Ord코드와 Ball코드 일치 여부 (8.7%) |
| sc_pattern | 패턴 분류 (top_heavy/balanced/spread) |

#### Shortcode 세그먼트 정의
- **Ord 세그먼트**: 최근 10회차 ord1~ord6 빈도 기준
- **Ball 세그먼트**: 최근 10회차 ball1~ball6 빈도 기준
- **Top24**: 빈도 상위 24개 번호
- **Mid14**: 중간 빈도 14개 번호
- **Rest7**: 하위 빈도 7개 번호
- 첫 10회차(826~835)는 데이터 부족으로 빈 값

#### Shortcode 계산 예시
예: `420321` = Ord(4,2,0) + Ball(3,2,1)
- Ord 코드 420: 당첨번호 중 4개가 Ord Top24, 2개가 Mid14, 0개가 Rest7
- Ball 코드 321: 당첨번호 중 3개가 Ball Top24, 2개가 Mid14, 1개가 Rest7

#### 패턴 분류 기준
- **top_heavy**: Top24에서 4개 이상 선택
- **spread**: Top24에서 2개 이하 + Rest7에서 2개 이상
- **balanced**: 그 외

### 플래그 및 이상치 (18개)
| 컬럼 | 설명 |
|------|------|
| flag_consecutive_normal | 연속수 정상 (0~1개) |
| flag_prime_normal | 소수 정상 (1~3개) |
| flag_odd_normal | 홀수 정상 (2~4개) |
| flag_carryover_normal | 이월수 정상 (0~1개) |
| flag_ac_normal | AC값 정상 (7~10) |
| flag_sum_normal | 합계 정상 (100~159) |
| flag_samedigit_normal | 동끝수 정상 (0~1쌍) |
| flag_skew_normal | 왜도 정상 (balanced) |
| flag_gaptype_normal | 간격 정상 (uniform/moderate) |
| flag_std_normal | 표준편차 정상 (8~14) |
| flag_fib_normal | 피보나치 정상 (0~2개) |
| flag_golden_normal | 황금비 정상 (0~2쌍) |
| flag_sctop24_normal | Top24 정상 (2~4개) |
| flag_screst7_normal | Rest7 정상 (0~1개) |
| flag_scpattern_normal | SC패턴 정상 (balanced/top_heavy) |
| normal_score | 정상 항목 수 (15개 중) |
| outlier_count | 이상치 개수 |
| outlier_list | 이상치 목록 |

## 이상치 탐지 기준

| 항목 | 정상 범위 | 정상 비율 | 이상치 표기 |
|------|-----------|----------|-------------|
| 연속수 | 0~1개 | 91.0% | 연속수:N개 |
| 소수 | 1~3개 | 83.6% | 소수:N개 |
| 홀수 | 2~4개 | 84.2% | 홀수:N개 |
| 이월수 | 0~1개 | 80.5% | 이월:N개 |
| AC값 | 7~10 | 83.1% | AC:N |
| 합계 | 100~159 | 69.9% | 합계:N |
| 동끝수 | 0~1쌍 | 79.2% | 동끝수:N쌍 |
| 왜도 | balanced | 43.0% | 왜도:저편중/고편중 |
| 간격 | uniform/moderate | 74.9% | 간격:불균등 |
| std | 8~14 | 71.2% | std:집중/분산(값) |
| 피보나치 | 0~2개 | 93.4% | 피보나치:N개 |
| 황금비 | 0~2쌍 | 93.9% | 황금비:N쌍 |
| Top24 | 2~4개 | 79.9% | Top24:N개 |
| Rest7 | 0~1개 | 78.0% | Rest7:N개 |
| SC패턴 | balanced/top_heavy | 90.5% | SC패턴:spread |

## 이상치 분포

- 0개 이상치: 4.0%
- 1개 이상치: 13.5%
- 2개 이상치: 21.1%
- 3개 이상치: 27.4%
- 4개 이상치: 16.6%
- 5개 이상치: 10.6%
- 6개+ 이상치: 6.9%

## Shortcode 인사이트 통계 (108개)

### Ord 세그먼트 통계

#### ord_code 분포 (상위 5개)
| 코드 | 비율 |
|------|------|
| 321 | 14.6% |
| 411 | 13.6% |
| 420 | 13.0% |
| 231 | 10.0% |
| 312 | 8.9% |

#### sc_pattern 분포
| 패턴 | 비율 |
|------|------|
| balanced | 48.2% |
| top_heavy | 42.3% |
| spread | 9.5% |

#### sc_top24 분포 (Ord)
| 개수 | 비율 |
|------|------|
| 2개 | 18.4% |
| 3개 | 32.0% |
| 4개 | 29.5% |
| 5개 | 11.7% |

### Ball 세그먼트 통계

#### ball_code 분포 (상위 5개)
| 코드 | 비율 |
|------|------|
| 321 | 16.8% |
| 420 | 11.1% |
| 411 | 10.8% |
| 231 | 8.4% |
| 312 | 8.4% |

#### ball_top24 분포
| 개수 | 비율 |
|------|------|
| 1개 | 7.9% |
| 2개 | 20.6% |
| 3개 | 34.4% |
| 4개 | 26.0% |
| 5개 | 8.7% |
| 6개 | 2.4% |

#### ball_rest7 분포
| 개수 | 비율 |
|------|------|
| 0개 | 32.2% |
| 1개 | 42.0% |
| 2개 | 22.5% |
| 3개 | 3.0% |

### Ord/Ball 일치율
- Ord코드 = Ball코드: **8.7%**
