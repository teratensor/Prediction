"""
1단계: 당첨 데이터 가져오기

DB: 우분투 서버 lottoda_wp.lotto_data
MCP를 통해 MySQL DB에서 당첨번호를 조회하여 winning_numbers.csv에 저장
"""

import csv
from pathlib import Path

# DB 정보 (우분투 서버: lottoda_wp.lotto_data)
DB_HOST = '192.168.45.113'
DB_USER = 'root'
DB_PASSWORD = '@Thereisno123'
DB_NAME = 'lottoda_wp'
DB_TABLE = 'lotto_data'

OUTPUT_PATH = Path(__file__).parent / "winning_numbers.csv"


def save(data: list):
    """당첨번호 데이터를 CSV로 저장

    Args:
        data: DB 결과 [{"회차": 1, "ord1": 10, "ball1": 5, ..., "o1": 5, ...}, ...]
    """
    # 헤더: 회차, ord1-6, ball1-6, o1-o45
    header = ['round', 'ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6',
              'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6']
    header += [f'o{i}' for i in range(1, 46)]

    with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in data:
            if isinstance(row, dict):
                csv_row = [
                    row.get('회차', 0),
                    row.get('ord1') or 0, row.get('ord2') or 0, row.get('ord3') or 0,
                    row.get('ord4') or 0, row.get('ord5') or 0, row.get('ord6') or 0,
                    row.get('ball1') or 0, row.get('ball2') or 0, row.get('ball3') or 0,
                    row.get('ball4') or 0, row.get('ball5') or 0, row.get('ball6') or 0
                ]
                csv_row += [row.get(f'o{i}') or 0 for i in range(1, 46)]
                writer.writerow(csv_row)

    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"총 {len(data)}회차")


if __name__ == "__main__":
    print("MCP를 통해 DB 데이터를 가져온 후 save(data) 호출 필요")
