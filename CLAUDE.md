# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

로또 6/45 당첨번호 예측 시스템 - ML 기반 파이프라인

**핵심 목표: 200개 조합 중 1개라도 6개 완전 일치**

## 실행 방법

```bash
# 다음 회차 예측 (인자 없이)
python 3_predict/main.py

# 특정 회차 예측
python 3_predict/main.py 1205

# 범위 예측 (백테스트)
python 3_predict/main.py 1150 1204

# 파라미터 조정
python 3_predict/main.py 1205 --top_pairs 600 --top_inner 5
```

## 아키텍처

```
1_data/              → winning_numbers.csv (826~1204회차)
  └─ fetch.py        → DB에서 데이터 가져오기
2_describe/          → Ord 기반 회차별 분석 (115개 컬럼)
3_predict/           → Ord 기반 ML 예측 파이프라인 (100개 조합)
  ├─ common.py       → 클러스터링, 이상치, 다양성 필터
  ├─ 1_ord1ord6_ml/  → (ord1, ord6) 쌍 예측
  ├─ 2_ord2_ml/ ~ 5_ord5_ml/  → ord2~ord5 예측
  └─ main.py         → 통합 파이프라인 (Ord + Ball 200개 출력)
4_ball_describe/     → Ball 순서 기반 분석
5_ball_predict/      → Ball 기반 ML 예측 파이프라인 (100개 조합)
  ├─ common.py       → Ball→Ord 변환, 이상치 계산
  ├─ 1_ball1ball6_ml/ ~ 5_ball5_ml/  → ball1~ball6 예측
  └─ main.py         → Ball 파이프라인
6_position_frequency/→ positionf.csv (위치별 빈도수 저장)
```

## 예측 파이프라인

### Ord 파이프라인 (3_predict)
```
ord1, ord6 → ord4 → ord2 → ord3 → ord5
→ 30,000개 조합 → 이상치 필터링 → 다양성 필터링 → 100개
```

### Ball 파이프라인 (5_ball_predict)
```
ball1, ball6 → ball2 → ball3 → ball4 → ball5
→ Ball→Ord 변환 → 이상치 필터링 → 다양성 필터링 → 100개
```

### 통합 출력
- 1~100번: Ord 예측
- 101~200번: Ball→Ord 변환 예측
- 위치별 빈도수 표 (ord1~ord6 가로 나열, 빈도순 정렬)
- CSV 저장: 6_position_frequency/positionf.csv (실제 당첨번호의 위치별 빈도수)

## 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| --top | 100 | 각 파이프라인 조합 수 |
| --top_pairs | 400 | (ord1, ord6) 또는 (ball1, ball6) 상위 개수 |
| --top_inner | 3 | 중간 포지션 각각의 상위 개수 |

## 핵심 개념

| 용어 | 설명 |
|------|------|
| ord | 오름차순 정렬된 판매순위 (ord1=최소, ord6=최대) |
| ball | 실제 추첨된 공 번호 순서 (1~45) |
| o1~o45 | 판매순위 1~45위의 공 번호 |
| Ball→Ord | ball 조합을 ord 순서로 변환 |

## 데이터 갱신

```bash
python 1_data/fetch.py      # DB에서 최신 데이터 가져오기
python 2_describe/main.py   # describe.csv 재생성
```

## DB 연결

- Host: `192.168.45.113`
- Database: `lottoda_wp.lotto_data`

## Python 환경

```bash
# Base Python 사용 (권장)
/usr/local/bin/python3 3_predict/main.py 1205

# 필수 패키지
/usr/local/bin/pip3 install lightgbm xgboost numpy scipy
```

## Claude Code 작업 규칙

1. **가상환경 유도 금지**: conda, venv 등 가상환경 생성/활성화를 유도하지 않음. Base Python 사용.
2. **코드 삭제 금지**: 기존 코드를 마음대로 삭제하지 않음. 수정 시 사용자 확인 필요.
3. **예외 처리 우회 금지**: 패키지 미설치 시 try-except로 우회하지 말고 패키지 설치로 해결.

## 출력 형식

| 표시 | 의미 |
|------|------|
| ★★★★ | 6개 일치 |
| ★★★ | 5개 일치 |
| ★★ | 4개 일치 |
| ★ | 3개 일치 |
| (N개) | 0~2개 일치 |
