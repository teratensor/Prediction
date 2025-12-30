# 1_data - 데이터 수집 및 저장

로또 당첨번호 데이터 관리

## 파일

| 파일 | 설명 |
|------|------|
| winning_numbers.csv | 당첨번호 데이터 (메인) |
| winning_numbers.json | 당첨번호 데이터 (JSON) |
| fetch.py | DB에서 당첨번호 조회 후 저장 |

## winning_numbers.csv 형식

| 컬럼 | 설명 |
|------|------|
| round | 회차 |
| ord1~6 | 오름차순 정렬된 번호 |
| ball1~6 | 원본 번호 |
| o1~o45 | 번호별 출현 여부 (0/1) |

## 사용법

MCP MySQL 도구로 데이터 조회 후:
```python
from fetch import save
save(data)  # data: [{"회차": 1, "ord1": 10, "o1": 5, ...}, ...]
```

## DB 연결

- 서버: `192.168.45.113`
- DB: `lottoda_wp.lotto_data`
