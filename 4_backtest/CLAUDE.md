# 3_backtest

예측 알고리즘 백테스트

## 요약 코드 형식

`1204,501420` = `회차,ABCDEF`

| 위치 | 의미 | 설명 |
|------|------|------|
| A | Ord Top24 | 실제 당첨 ord 6개 중 예측 ord_top24에 포함된 개수 |
| B | Ord Mid14 | 실제 당첨 ord 6개 중 예측 ord_mid14에 포함된 개수 |
| C | Ord Rest7 | 실제 당첨 ord 6개 중 예측 ord_rest7에 포함된 개수 |
| D | Ball Top24 | 실제 당첨 ord 6개 중 예측 ball_top24에 포함된 개수 |
| E | Ball Mid14 | 실제 당첨 ord 6개 중 예측 ball_mid14에 포함된 개수 |
| F | Ball Rest7 | 실제 당첨 ord 6개 중 예측 ball_rest7에 포함된 개수 |

**예시: 1204,501420**
- Ord: Top24에 5개, Mid14에 0개, Rest7에 1개 적중 (5+0+1=6)
- Ball: Top24에 4개, Mid14에 2개, Rest7에 0개 적중 (4+2+0=6)

## 실행 방법

```bash
# 누적 백테스트 (826회차부터 50회차 학습 → 876회차부터 예측)
python 3_backtest/backtest.py

# 특정 회차만 상세 백테스트
python 3_backtest/backtest.py --round 1204

# 커스텀 설정
python 3_backtest/backtest.py --start 900 --train 30
```

## 누적 학습 방식

1. 826회차부터 50회차 학습 → 876회차 예측
2. 826회차부터 51회차 학습 → 877회차 예측
3. 826회차부터 52회차 학습 → 878회차 예측
4. ... (반복)
5. 826회차부터 378회차 학습 → 1204회차 예측

## 출력 파일

- `backtest_results.csv`: 전체 백테스트 결과
  - 컬럼: round, train_count, summary, ord_top24, ord_mid14, ord_rest7, ball_top24, ball_mid14, ball_rest7

## 검증 로직

**Ord 검증**: 실제 당첨 ord가 예측의 ord 섹션 중 어디에?
- 예측: ord_top24 = [3,12,9,11,...]
- 실제: ord = [6,17,18,23,27,31]
- 결과: 6→Rest7, 17→Top24, 18→Top24, ...

**Ball 검증**: 실제 당첨 ord가 예측의 ball 섹션 중 어디에?
- 예측: ball_top24 = [15,9,35,38,23,...]
- 실제: ord = [6,17,18,23,27,31]
- 결과: 6→Top24, 17→Top24, 18→Mid14, ...
