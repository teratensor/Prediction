"""
2단계: 합계 인사이트 생성

1_data/winning_numbers.json을 읽어서 합계 확률 분포 생성
→ config.json, distribution.json 저장
"""

import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.json"
OUTPUT_DIR = Path(__file__).parent / "statistics"


def load_data():
    """당첨번호 데이터 로드"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_insight():
    """합계 인사이트 생성"""
    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    # 각 합계별 가능한 조합 수 계산
    sum_combo_counts = defaultdict(int)
    for combo in combinations(range(1, 46), 6):
        s = sum(combo)
        sum_combo_counts[s] += 1

    # 역대 데이터에서 합계 빈도 계산
    sum_freq = defaultdict(int)
    sum_list = []
    for row in data:
        total = sum(row[1:])  # row[0]은 회차, row[1:]은 번호 6개
        sum_freq[total] += 1
        sum_list.append(total)

    total_draws = len(data)
    MIN_PROB = 0.0005  # 최소 확률

    # 확률 분포 생성
    probabilities = {}
    for s in range(21, 256):
        combo_count = sum_combo_counts.get(s, 0)
        if combo_count == 0:
            continue

        if s in sum_freq:
            probabilities[s] = sum_freq[s] / total_draws
        else:
            combo_ratio = combo_count / max(sum_combo_counts.values())
            probabilities[s] = MIN_PROB * (0.5 + 0.5 * combo_ratio)

    # 정규화
    total_prob = sum(probabilities.values())
    distribution = {str(s): p / total_prob for s, p in probabilities.items()}

    # 가장 많이 나온 합계 Top 6
    sorted_freq = sorted(sum_freq.items(), key=lambda x: -x[1])
    most_common = [s for s, _ in sorted_freq[:6]]

    # config.json 생성
    config = {
        "name": "합계 인사이트",
        "description": "6개 번호의 합계 기반 확률 분포",
        "summary": {
            "min_possible": 21,
            "max_possible": 255,
            "mean": round(sum(sum_list) / len(sum_list)),
            "historical_min": min(sum_list),
            "historical_max": max(sum_list),
            "most_common": most_common
        },
        "metadata": {
            "total_draws": total_draws,
            "total_combinations": 8145060,
            "historical_sums": len(sum_freq),
            "all_possible_sums": len(probabilities),
            "coverage": "100%"
        },
        "probability_config": {
            "min_probability": MIN_PROB,
            "method": "frequency_based",
            "normalization": True
        }
    }

    # 저장
    with open(OUTPUT_DIR / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "distribution.json", 'w', encoding='utf-8') as f:
        json.dump(distribution, f, indent=2)

    print(f"config.json 저장 완료")
    print(f"distribution.json 저장 완료")
    print(f"총 {total_draws}회차 분석, {len(sum_freq)}개 합계 발견")


if __name__ == "__main__":
    generate_insight()
