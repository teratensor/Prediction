# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

로또 6/45 당첨번호 예측 시스템 - ML 기반 파이프라인

**핵심 목표: 100개 조합 중 1개라도 6개 완전 일치**

## 실행 방법

```bash
# 다음 회차 예측 (인자 없이)
python 3_predict/main.py

# 특정 회차 예측 (숫자만)
python 3_predict/main.py 1205

# 파라미터 조정
python 3_predict/main.py 1205 --top_pairs 600 --top_inner 5

# 백테스트 실행
python 3_predict/main.py --backtest --start 1150 --end 1204
```

## 아키텍처

```
1_data/              → winning_numbers.csv (826~1204회차, 379개)
  └─ fetch.py        → DB에서 데이터 가져오기
2_describe/          → 회차별 분석 (115개 컬럼)
  ├─ describe.csv    → 379개 회차 × 115개 컬럼
  ├─ insights.csv    → 108개 통계
  └─ main.py         → python 2_describe/main.py
3_predict/           → ML 예측 파이프라인 (메인)
  ├─ common.py       → 클러스터링, 이상치, 다양성 필터
  ├─ 1_ord1ord6_ml/  → (ord1, ord6) 쌍 예측 (v2 앙상블)
  ├─ 2_ord2_ml/      → ord2 예측
  ├─ 3_ord3_ml/      → ord3 예측
  ├─ 4_ord4_ml/      → ord4 예측
  ├─ 5_ord5_ml/      → ord5 예측
  └─ main.py         → 파이프라인 진입점
```

## 예측 파이프라인

```
1. (ord1, ord6) 쌍 후보 생성 (Top-400, v2 개별예측)
2. ord4 예측 (ord1, ord6 → Top-3)
3. ord2 예측 (ord1, ord4 → Top-3)
4. ord3 예측 (ord2, ord4 → Top-3)
5. ord5 예측 (ord4, ord6 → Top-3)
6. 약 30,000개 조합 생성
7. 이상치 점수 계산
8. 다양성 필터링 (3자리 이상 중복 제거)
9. 최종 100개 선택
```

### 의존성 구조

```
ord1, ord6 → ord4 → ord2 → ord3 → ord5
```

## 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| --top | 100 | 최종 조합 수 |
| --top_pairs | 400 | (ord1, ord6) 상위 개수 |
| --top_inner | 3 | ord2~ord5 각 상위 개수 |

## 핵심 제약

1. **다양성**: 조합 간 3자리 이상 중복 금지
2. **클러스터**: 5개 구간 (A:1-9, B:10-18, C:19-27, D:28-36, E:37-45)
3. **이상치 허용**: 3개까지 (전체의 66% 커버)

## 포지션별 최빈 구간

| 포지션 | 최빈 구간 | 비율 |
|--------|-----------|------|
| ord1 | 1-5 | 38.4% |
| ord2 | 10-19 | 48.3% |
| ord5 | 30-39 | 54.6% |
| ord6 | 43-45 | 31.4% |

## 백테스트 결과 (55회차, 1150-1204)

- 평균 최대 일치: 3.04개
- 4개 일치: 20.0%
- 3개 일치: 63.6%
- 6개 완전 일치: 0회 (개선 필요)

## 출력 형식

| 표시 | 의미 |
|------|------|
| ★★★ | 5~6개 일치 |
| ★★ | 4개 일치 |
| ★ | 3개 일치 |
| (N개) | 0~2개 일치 |

## 테스트 결과 보고 규칙

6개 일치 실패 시 반드시 분석:

```
❌ 6개 일치 실패:
- 실제: (6, 17, 18, 23, 27, 31)
- 가장 근접: (6, 17, 19, 24, 27, 31) → 4개 일치
- 누락: 18, 23
- 분석: ord3, ord4 모델 개선 필요
```

## 핵심 개념

| 용어 | 설명 |
|------|------|
| ord | 오름차순 정렬된 위치 (ord1=최소, ord6=최대) |
| ball | 실제 로또 공 번호 (1~45) |
| span | ord6 - ord1 (범위) |
| 클러스터 | 5개 구간 (A-E) |

## 데이터 갱신

```bash
python 1_data/fetch.py      # DB에서 최신 데이터 가져오기
python 2_describe/main.py   # describe.csv 재생성
```

## DB 연결

- Host: `192.168.45.113`
- Database: `lottoda_wp.lotto_data`
- User: `teratensor`

## Python 환경

```bash
# Base Python (권장)
/usr/local/bin/python3 3_predict/main.py 1205

# 필수 패키지 설치
/usr/local/bin/pip3 install lightgbm xgboost numpy scipy
```

### 설치된 패키지

| 패키지 | 버전 | 용도 |
|--------|------|------|
| lightgbm | 4.6.0 | ord2~ord5 ML 모델 |
| xgboost | 3.1.2 | ord1ord6 ML 모델 |
| scipy | 1.16.3 | 수치 계산 |
| numpy | 2.3.5 | 배열 연산 |

## Claude Code 작업 규칙

1. **가상환경 유도 금지**: conda, venv 등 가상환경 생성/활성화를 유도하지 않음. Base Python 사용.
2. **코드 삭제 금지**: 기존 코드를 마음대로 삭제하지 않음. 수정 시 사용자 확인 필요.
3. **예외 처리 우회 금지**: 패키지 미설치 시 try-except로 우회하지 말고 패키지 설치로 해결.
