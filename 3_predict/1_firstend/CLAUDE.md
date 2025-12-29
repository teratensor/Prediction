# FirstEnd 인사이트

(ord1, ord6) 쌍 빈도 분석

## 실행

```bash
python export_result.py
```

## 출력

`result/result.csv`

| 컬럼 | 설명 |
|------|------|
| ord1 | 첫수 (1-21) |
| ord2 | (null) |
| ord3 | (null) |
| ord4 | (null) |
| ord5 | (null) |
| ord6 | 끝수 (23-45) |
| 빈도수 | 출현 횟수 |
| 회차 | 출현 회차 목록 |

## 필터 조건

- ord1: 1~21 (22 이상 제외)
- ord6: 23~45 (22 이하 제외)
- 최소 5칸 차이 (ord6 >= ord1 + 5)

## 통계

- 전체 477개 쌍
- 출현 쌍: 185개
- 미출현 쌍: 292개
- 빈도 높은 순 정렬
