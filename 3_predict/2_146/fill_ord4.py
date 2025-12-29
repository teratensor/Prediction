"""
firstend의 result.csv를 기반으로 ord4 예측값 채우기

공식:
- A (기본): ord4 = ord1 + span × 0.60
- B (낮음): ord4 = ord1 + span × 0.31
- C (높음): ord4 = ord1 + span × 0.87

입력: result/result.csv (ord1, ord6 쌍)
출력: result/result.csv (ord4 채움, 공식별 행 복제)
"""

import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULT_PATH = BASE_DIR.parent.parent / "result" / "result.csv"

# ord4 예측 공식 (비율)
FORMULAS = {
    'A': 0.60,  # 기본
    'B': 0.31,  # 낮음
    'C': 0.87,  # 높음
}


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
    """ord4 채우기 - 공식별로 행 복제, 중복 제거"""
    new_rows = []
    seen = set()  # (ord1, ord4, ord6) 중복 체크

    for row in rows:
        ord1 = int(row['ord1'])
        ord6 = int(row['ord6'])
        freq = row['빈도수']
        rounds = row['회차']

        # 각 공식별로 ord4 계산
        for formula, ratio in FORMULAS.items():
            ord4 = calculate_ord4(ord1, ord6, ratio)

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
                '공식': formula,
            })

    return new_rows


def save_result(rows):
    """result.csv 저장"""
    with open(RESULT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6', '빈도수', '회차', '공식'])

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
                row['공식'],
            ])

    print(f"저장 완료: {RESULT_PATH}")


def main():
    # 1. result.csv 로드
    rows = load_result()
    print(f"입력: {len(rows)}개 행")

    # 2. ord4 채우기 (공식별 복제, 중복 제거)
    new_rows = fill_ord4(rows)
    print(f"출력: {len(new_rows)}개 행")

    # 3. 빈도순 정렬
    new_rows.sort(key=lambda x: (-int(x['빈도수']), x['ord1'], x['ord4'], x['ord6']))

    # 4. 공식별 통계
    formula_counts = {'A': 0, 'B': 0, 'C': 0}
    for row in new_rows:
        formula_counts[row['공식']] += 1

    print(f"\n[공식별 행 수]")
    for formula, count in formula_counts.items():
        print(f"  {formula}: {count}개")

    # 5. 저장
    save_result(new_rows)


if __name__ == "__main__":
    main()
