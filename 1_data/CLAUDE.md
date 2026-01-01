# 1_data - 데이터 수집 및 저장

로또 당첨번호 데이터 관리 (826~1204회차, 379개)

## 파일

| 파일 | 설명 |
|------|------|
| winning_numbers.csv | 당첨번호 데이터 (메인, 81개 컬럼) |
| winning_numbers.json | 당첨번호 데이터 (JSON) |
| fetch.py | DB에서 당첨번호 조회 후 저장 |

## 실행

```bash
python 1_data/fetch.py  # DB에서 최신 데이터 가져오기
```

## winning_numbers.csv 형식

| 컬럼 그룹 | 컬럼 | 설명 |
|----------|------|------|
| 기본 | round | 회차 |
| 정렬번호 | ord1~ord6 | 오름차순 정렬된 번호 |
| 추첨번호 | ball1~ball6 | 원본 번호 |
| 보너스 | bonus, ord_bonus | 보너스 번호 |
| ordball | ordball1~ordball6, ordball_bonus | 정렬 위치 |
| 원핫 | o1~o45 | 번호별 출현 여부 (0/1) |
| 당첨정보 | prize1~5_amount/winners, total_sales | 상금/당첨자 |
| 메타 | draw_date, created_at, updated_at | 날짜 정보 |

## DB 연결

- 서버: `192.168.45.113`
- DB: `lottoda_wp.lotto_data`
- 사용자: `teratensor`
