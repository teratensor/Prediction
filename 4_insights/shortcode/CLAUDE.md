# 요약코드(Shortcode) 인사이트

Top24/Mid14/Rest7 세그먼트별 적중 패턴 분석

## 요약코드 형식

`ABCDEF` (6자리)
- **ABC**: Ord 세그먼트별 적중 수 (Top24/Mid14/Rest7)
- **DEF**: Ball 세그먼트별 적중 수 (Top24/Mid14/Rest7)

예: `321411` = Ord(3,2,1) + Ball(4,1,1)

## 세그먼트 정의

| 세그먼트 | 설명 | 크기 |
|----------|------|------|
| Top24 | 최근 10회 빈도 상위 24개 | 24 |
| Mid14 | 중간 빈도 14개 | 14 |
| Rest7 | 하위 빈도 7개 | 7 |

## 실행

```bash
python generate.py
```

※ 백테스트 결과(backtest_results.csv)가 필요함

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| full_code_distribution.csv | 전체 6자리 코드 분포 |
| ord_code_distribution.csv | Ord 3자리 코드 분포 |
| ball_code_distribution.csv | Ball 3자리 코드 분포 |
| segment_distribution.csv | 각 세그먼트별 개수 분포 |
| summary.csv | 요약 통계 (평균 적중 수) |
