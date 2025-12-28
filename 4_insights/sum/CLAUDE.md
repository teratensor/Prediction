# 합계(Sum) 인사이트

당첨번호 6개의 합계 분석

## 실행

```bash
python generate.py
```

## 출력 파일 (statistics/)

| 파일 | 설명 |
|------|------|
| config.csv | 요약 정보 (평균, 최소, 최대, 표준편차) |
| distribution.csv | 합계별 분포 |

## CSV 컬럼

- **sum**: 합계 값
- **frequency**: 출현 횟수
- **ratio**: 비율 (%)
- **probability**: 확률 (0~1)

## 참고

- 이론적 합계 범위: 21 (1+2+3+4+5+6) ~ 255 (40+41+42+43+44+45)
- 평균 합계: 약 133
