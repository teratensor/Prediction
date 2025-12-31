"""
1단계: 당첨 데이터 가져오기

DB: 우분투 서버 lottoda_wp.lotto_data
MCP를 통해 MySQL DB에서 당첨번호를 조회하여 winning_numbers.csv에 저장

전체 81개 컬럼:
- 기본: 회차 (1)
- 추첨번호: ball1~6 (6)
- 정렬번호: ord1~6 (6)
- 보너스: bonus, ord_bonus (2)
- ordball: ordball1~6, ordball_bonus (7)
- 원핫: o1~o45 (45)
- 당첨정보: 당첨일, 등수1~5_당첨금, 등수1~5_당첨자수 (11)
- 판매정보: 총판매금액 (1)
- 메타: created_at, updated_at (2)
"""

import csv
from pathlib import Path

import pymysql

# DB 정보 (우분투 서버: lottoda_wp.lotto_data)
DB_HOST = '192.168.45.113'
DB_USER = 'teratensor'
DB_PASSWORD = '@Thereisno123'
DB_NAME = 'lottoda_wp'
DB_TABLE = 'lotto_data'

OUTPUT_PATH = Path(__file__).parent / "winning_numbers.csv"


def save(data: list):
    """당첨번호 데이터를 CSV로 저장 (전체 81개 컬럼)

    Args:
        data: DB 결과 [{"회차": 1, "ord1": 10, "ball1": 5, ..., "o1": 5, ...}, ...]
    """
    # 헤더 구성 (81개 컬럼)
    header = [
        'round',
        'ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6',
        'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6',
        'bonus', 'ord_bonus',
        'ordball1', 'ordball2', 'ordball3', 'ordball4', 'ordball5', 'ordball6', 'ordball_bonus',
    ]
    header += [f'o{i}' for i in range(1, 46)]
    header += [
        'draw_date',
        'prize1_amount', 'prize1_winners',
        'prize2_amount', 'prize2_winners',
        'prize3_amount', 'prize3_winners',
        'prize4_amount', 'prize4_winners',
        'prize5_amount', 'prize5_winners',
        'total_sales',
        'created_at', 'updated_at'
    ]

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
                    row.get('ball4') or 0, row.get('ball5') or 0, row.get('ball6') or 0,
                    row.get('bonus') or 0, row.get('ord_bonus') or 0,
                    row.get('ordball1') or 0, row.get('ordball2') or 0, row.get('ordball3') or 0,
                    row.get('ordball4') or 0, row.get('ordball5') or 0, row.get('ordball6') or 0,
                    row.get('ordball_bonus') or 0,
                ]
                csv_row += [row.get(f'o{i}') or 0 for i in range(1, 46)]
                csv_row += [
                    row.get('당첨일') or '',
                    row.get('등수1_당첨금') or 0, row.get('등수1_당첨자수') or 0,
                    row.get('등수2_당첨금') or 0, row.get('등수2_당첨자수') or 0,
                    row.get('등수3_당첨금') or 0, row.get('등수3_당첨자수') or 0,
                    row.get('등수4_당첨금') or 0, row.get('등수4_당첨자수') or 0,
                    row.get('등수5_당첨금') or 0, row.get('등수5_당첨자수') or 0,
                    row.get('총판매금액') or 0,
                    row.get('created_at') or '', row.get('updated_at') or ''
                ]
                writer.writerow(csv_row)

    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"총 {len(data)}회차, {len(header)}개 컬럼")


def fetch():
    """pymysql로 MySQL DB에서 데이터를 가져와 CSV로 저장"""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4'
    )

    columns = """회차, ord1, ord2, ord3, ord4, ord5, ord6,
        ball1, ball2, ball3, ball4, ball5, ball6, bonus, ord_bonus,
        ordball1, ordball2, ordball3, ordball4, ordball5, ordball6, ordball_bonus,
        o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
        o16, o17, o18, o19, o20, o21, o22, o23, o24, o25, o26, o27, o28, o29, o30,
        o31, o32, o33, o34, o35, o36, o37, o38, o39, o40, o41, o42, o43, o44, o45,
        당첨일, 등수1_당첨금, 등수1_당첨자수, 등수2_당첨금, 등수2_당첨자수,
        등수3_당첨금, 등수3_당첨자수, 등수4_당첨금, 등수4_당첨자수,
        등수5_당첨금, 등수5_당첨자수, 총판매금액, created_at, updated_at"""

    sql = f"SELECT {columns} FROM {DB_TABLE} ORDER BY 회차 ASC"

    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        print("데이터 없음")
        return

    # 헤더 구성 (81개 컬럼)
    header = [
        'round',
        'ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6',
        'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6',
        'bonus', 'ord_bonus',
        'ordball1', 'ordball2', 'ordball3', 'ordball4', 'ordball5', 'ordball6', 'ordball_bonus',
    ]
    header += [f'o{i}' for i in range(1, 46)]
    header += [
        'draw_date',
        'prize1_amount', 'prize1_winners',
        'prize2_amount', 'prize2_winners',
        'prize3_amount', 'prize3_winners',
        'prize4_amount', 'prize4_winners',
        'prize5_amount', 'prize5_winners',
        'total_sales',
        'created_at', 'updated_at'
    ]

    with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"총 {len(rows)}회차, {len(header)}개 컬럼")


if __name__ == "__main__":
    fetch()
