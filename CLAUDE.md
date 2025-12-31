# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

로또 6/45 당첨번호 예측 시스템 - ML 기반 파이프라인

**핵심 목표: 6개 일치** - 5개 이하 적중은 의미 없음.

## 테스트 결과 보고 규칙

테스트 실행 후 결과를 보고할 때 반드시 다음을 포함:

1. **6개 일치 여부** 및 결과 요약
2. **실패 원인 분석** (6개 일치 실패 시):
   - 실제 당첨번호 vs Top-K 예측 비교
   - 어떤 위치(ord1~ord6)에서 누락되었는지
   - 누락된 번호가 해당 위치의 Top-K에 포함되었는지
   - ord1,ord6 쌍이 Top-100에 포함되었는지
3. **개선 방향 제안**: 어떤 모듈을 개선해야 할지

예시:
```
❌ 6개 일치 실패 분석:
- 실제: (6, 17, 18, 23, 27, 31)
- (ord1=6, ord6=31) 쌍이 Top-100에 미포함 → ord1ord6 모델 개선 필요
- ord4=23이 Top-10에 미포함 (예측: [25,26,27,...]) → ord4 모델 개선 필요
```

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
1_data/              → winning_numbers.csv (당첨번호 826~1204회차, 379개)
  └─ fetch.py        → DB에서 데이터 가져오기 (pymysql)
2_describe/          → 회차별 데이터 분석 (describe.csv 101개 컬럼, insights.csv 71개 통계)
  └─ main.py         → python 2_describe/main.py
3_insights/          → 8개 통계 인사이트
4_predict/           → ML 모델 (ord별 예측)
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

## 인사이트 (3_insights/)

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

## 이상치 탐지 (2_describe)

12개 항목에 대해 정상/이상치 판정:

| 항목 | 정상 범위 | 정상 비율 |
|------|-----------|----------|
| 연속수 | 0~1개 | 91.0% |
| 소수 | 1~3개 | 83.6% |
| 홀수 | 2~4개 | 84.2% |
| 이월수 | 0~1개 | 80.5% |
| AC값 | 7~10 | 83.1% |
| 합계 | 100~159 | 69.9% |
| 동일끝수 | 0~1쌍 | 79.2% |
| 왜도 | balanced | 43.0% |
| 간격균등도 | uniform/moderate | 74.9% |
| 표준편차 | 8~14 | 71.2% |
| 피보나치 | 0~2개 | 93.4% |
| 황금비쌍 | 0~2쌍 | 93.9% |

`describe.csv`의 `outlier_list` 컬럼에 이상치 목록 기록됨.

## 데이터 형식

**winning_numbers.csv** (81개 컬럼):
- 기본: round
- 정렬번호: ord1~6, 추첨번호: ball1~6
- 보너스: bonus, ord_bonus
- ordball: ordball1~6, ordball_bonus
- 원핫: o1~o45
- 당첨정보: draw_date, prize1~5_amount/winners, total_sales
- 메타: created_at, updated_at

## 데이터 갱신

```bash
python 1_data/fetch.py  # DB에서 최신 데이터 가져오기
```

## DB 연결

- Host: `192.168.45.113`
- Database: `lottoda_wp.lotto_data`
- User: `teratensor`
