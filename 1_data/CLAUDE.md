# 1_data

데이터 수집 및 저장 모듈

## 파일

- **fetch.py**: DB에서 당첨번호 조회 후 JSON 저장
- **winning_numbers.json**: 당첨번호 데이터

## 사용법

MCP MySQL 도구로 데이터 조회 후:
```python
from fetch import save
save(data)  # data: [{"회차": 1, "ord1": 10, "o1": 5, ...}, ...]
```

## 데이터 형식

`[[회차, ord1, ord2, ord3, ord4, ord5, ord6, {o값}], ...]`

- `row[0]`: 회차
- `row[1:7]`: ord 번호 6개
- `row[7]`: o값 딕셔너리 (ord → ball 변환용)
