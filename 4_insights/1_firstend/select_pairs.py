"""
첫수/끝수 점수 기반 선택

1. 첫수 빈도 점수
2. 끝수 빈도 점수
3. (첫수,끝수) 쌍 빈도 점수
4. 최근 3회 출현 패널티
5. 상위 90% 조합 선택
"""

import csv
from pathlib import Path

STATS_DIR = Path(__file__).parent / "statistics"
DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"


def load_distribution(filename: str) -> dict:
    """분포 CSV 로드 -> {value: frequency}"""
    result = {}
    with open(STATS_DIR / filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'first' in row and 'end' in row:
                # pair_distribution.csv
                key = (int(row['first']), int(row['end']))
            elif 'value' in row:
                key = int(row['value'])
            elif 'span' in row:
                key = int(row['span'])
            else:
                continue
            result[key] = int(row['frequency'])
    return result


def get_recent_firstend(n: int = 3) -> set:
    """최근 n회차의 첫수/끝수 가져오기"""
    recent = set()
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in rows[-n:]:
            balls = sorted([int(row[f'ball{i}']) for i in range(1, 7)])
            recent.add(balls[0])   # 첫수
            recent.add(balls[-1])  # 끝수
    return recent


def calculate_scores():
    """모든 (첫수,끝수) 조합의 점수 계산"""
    # 분포 로드
    first_dist = load_distribution("first_distribution.csv")
    end_dist = load_distribution("end_distribution.csv")
    pair_dist = load_distribution("pair_distribution.csv")

    # 최근 3회 첫수/끝수
    recent = get_recent_firstend(3)
    print(f"최근 3회 첫수/끝수: {sorted(recent)}")

    # 패널티 점수 (최근 출현 시 차감)
    PENALTY = 5

    scores = []

    # 가능한 모든 (첫수, 끝수) 조합 생성
    # 첫수: 1~30 (통계상 max), 끝수: 17~45 (통계상 min~max)
    for first in range(1, 31):
        for end in range(max(first + 7, 17), 46):  # span 최소 7 이상
            # 점수 계산
            first_score = first_dist.get(first, 0)
            end_score = end_dist.get(end, 0)
            pair_score = pair_dist.get((first, end), 0)

            total = first_score + end_score + pair_score

            # 최근 출현 패널티
            if first in recent:
                total -= PENALTY
            if end in recent:
                total -= PENALTY

            scores.append({
                'first': first,
                'end': end,
                'first_score': first_score,
                'end_score': end_score,
                'pair_score': pair_score,
                'penalty': -PENALTY if first in recent else 0 + (-PENALTY if end in recent else 0),
                'total': total
            })

    # 점수순 정렬
    scores.sort(key=lambda x: x['total'], reverse=True)

    return scores


def select_top_percent(scores: list, percent: float = 0.75) -> list:
    """상위 N% 조합 선택"""
    # 총 점수 합계
    total_sum = sum(s['total'] for s in scores if s['total'] > 0)

    # 누적 N% 선택
    selected = []
    cumsum = 0
    for s in scores:
        if s['total'] <= 0:
            continue
        cumsum += s['total']
        selected.append(s)
        if cumsum >= total_sum * percent:
            break

    return selected


def main():
    print("=== 첫수/끝수 점수 기반 선택 ===\n")

    scores = calculate_scores()
    selected = select_top_percent(scores, 0.75)

    print(f"전체 조합 수: {len(scores)}")
    print(f"상위 75% 선택: {len(selected)}개\n")

    # 결과 저장
    output_path = STATS_DIR / "selected_pairs.csv"
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['first', 'end', 'first_score', 'end_score', 'pair_score', 'penalty', 'total'])
        writer.writeheader()
        writer.writerows(selected)

    print(f"저장: {output_path}\n")

    # 상위 20개 출력
    print("상위 20개 조합:")
    print("-" * 70)
    print(f"{'순위':>4} {'첫수':>4} {'끝수':>4} {'첫수점수':>8} {'끝수점수':>8} {'쌍점수':>6} {'패널티':>6} {'총점':>6}")
    print("-" * 70)
    for i, s in enumerate(selected[:20], 1):
        print(f"{i:>4} {s['first']:>4} {s['end']:>4} {s['first_score']:>8} {s['end_score']:>8} {s['pair_score']:>6} {s['penalty']:>6} {s['total']:>6}")

    print("\n" + "=" * 70)
    print(f"선택된 첫수 범위: {min(s['first'] for s in selected)} ~ {max(s['first'] for s in selected)}")
    print(f"선택된 끝수 범위: {min(s['end'] for s in selected)} ~ {max(s['end'] for s in selected)}")


if __name__ == "__main__":
    main()
