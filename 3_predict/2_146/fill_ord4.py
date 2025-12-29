"""
firstend의 result.csv를 기반으로 ord4 예측값 채우기

공식 A (기본): ord4 = ord1 + span × 0.60
범위: 공식 A 결과에 ±7 적용 (15개 후보)

입력: result/result.csv (ord1, ord6 쌍)
출력: result/result.csv (ord4 채움, 범위별 행 복제)
"""

import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULT_PATH = BASE_DIR.parent.parent / "result" / "result.csv"

# ord4 예측: 공식 A (0.60) + ±7 범위
FORMULA_RATIO = 0.60
OFFSET_RANGE = range(-7, 8)  # -7 ~ +7 (15개)


def load_result():
    """result.csv 로드"""
    rows = []
    with open(RESULT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def calculate_ord4(ord1, ord6, ratio):
    """공식에 따라 ord4 계산"""
    span = ord6 - ord1
    ord4 = ord1 + span * ratio
    return round(ord4)


def fill_ord4(rows):
    """ord4 채우기 - 공식 A + ±7 범위, 중복 제거"""
    new_rows = []
    seen = set()  # (ord1, ord4, ord6) 중복 체크

    for row in rows:
        ord1 = int(row['ord1'])
        ord6 = int(row['ord6'])
        freq = row['빈도수']
        rounds = row['회차']

        # 공식 A 기준값 계산
        base_ord4 = calculate_ord4(ord1, ord6, FORMULA_RATIO)

        # ±7 범위 적용
        for offset in OFFSET_RANGE:
            ord4 = base_ord4 + offset

            # ord4 유효성 검사 (ord1 < ord4 < ord6)
            if ord4 <= ord1 or ord4 >= ord6:
                continue

            # 중복 체크
            key = (ord1, ord4, ord6)
            if key in seen:
                continue
            seen.add(key)

            new_rows.append({
                'ord1': ord1,
                'ord2': '',
                'ord3': '',
                'ord4': ord4,
                'ord5': '',
                'ord6': ord6,
                '빈도수': freq,
                '회차': rounds,
                'offset': offset,
            })

    return new_rows


def save_result(rows):
    """result.csv 저장"""
    with open(RESULT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6', '빈도수', '회차', 'offset'])

        for row in rows:
            writer.writerow([
                row['ord1'],
                row['ord2'],
                row['ord3'],
                row['ord4'],
                row['ord5'],
                row['ord6'],
                row['빈도수'],
                row['회차'],
                row['offset'],
            ])

    print(f"저장 완료: {RESULT_PATH}")


def main():
    # 1. result.csv 로드
    rows = load_result()
    print(f"입력: {len(rows)}개 행")

    # 2. ord4 채우기 (공식 A + ±7 범위, 중복 제거)
    new_rows = fill_ord4(rows)
    print(f"출력: {len(new_rows)}개 행")

    # 3. 빈도순 정렬
    new_rows.sort(key=lambda x: (-int(x['빈도수']), x['ord1'], x['ord4'], x['ord6']))

    # 4. offset별 통계
    from collections import Counter
    offset_counts = Counter(row['offset'] for row in new_rows)

    print(f"\n[offset별 행 수]")
    for offset in sorted(offset_counts.keys()):
        print(f"  {offset:+d}: {offset_counts[offset]}개")

    # 5. 저장
    save_result(new_rows)


if __name__ == "__main__":
    main()
