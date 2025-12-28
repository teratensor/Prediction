"""
4단계: 요약코드(Shortcode) 인사이트 생성

백테스트 결과에서 요약코드(ABCDEF) 패턴 분석
- ABC: Ord 세그먼트별 적중 (Top24/Mid14/Rest7)
- DEF: Ball 세그먼트별 적중 (Top24/Mid14/Rest7)

예: 321411 = Ord(3,2,1) + Ball(4,1,1)
"""

import csv
from pathlib import Path
from collections import Counter
import sys

# predict 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "2_predict"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "3_backtest"))

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"
BACKTEST_PATH = Path(__file__).parent.parent.parent / "3_backtest" / "backtest_results.csv"
OUTPUT_DIR = Path(__file__).parent / "statistics"


def load_backtest_results() -> list:
    """백테스트 결과 로드"""
    results = []
    if not BACKTEST_PATH.exists():
        return results

    with open(BACKTEST_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'summary': row['summary'],
                'ord_top24': int(row['ord_top24']),
                'ord_mid14': int(row['ord_mid14']),
                'ord_rest7': int(row['ord_rest7']),
                'ball_top24': int(row['ball_top24']),
                'ball_mid14': int(row['ball_mid14']),
                'ball_rest7': int(row['ball_rest7'])
            })
    return results


def generate_shortcode(ord_counts: tuple, ball_counts: tuple) -> str:
    """요약코드 생성

    Args:
        ord_counts: (top24, mid14, rest7) 적중 수
        ball_counts: (top24, mid14, rest7) 적중 수

    Returns:
        str: 6자리 요약코드 (예: "321411")
    """
    return f"{ord_counts[0]}{ord_counts[1]}{ord_counts[2]}{ball_counts[0]}{ball_counts[1]}{ball_counts[2]}"


def parse_shortcode(code: str) -> dict:
    """요약코드 파싱

    Args:
        code: 6자리 요약코드

    Returns:
        dict: 파싱된 결과
    """
    if len(code) != 6:
        return None

    return {
        'code': code,
        'ord_top24': int(code[0]),
        'ord_mid14': int(code[1]),
        'ord_rest7': int(code[2]),
        'ball_top24': int(code[3]),
        'ball_mid14': int(code[4]),
        'ball_rest7': int(code[5]),
        'ord_code': code[:3],
        'ball_code': code[3:]
    }


def generate_insight():
    """요약코드 인사이트 생성"""
    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = load_backtest_results()

    if not results:
        print("백테스트 결과가 없습니다. 먼저 backtest.py를 실행하세요.")
        return

    # 요약코드 빈도 분석
    code_freq = Counter(r['summary'] for r in results)
    ord_code_freq = Counter(r['summary'][:3] for r in results)
    ball_code_freq = Counter(r['summary'][3:] for r in results)

    # 가장 빈번한 패턴
    most_common_codes = code_freq.most_common(20)
    most_common_ord = ord_code_freq.most_common(10)
    most_common_ball = ball_code_freq.most_common(10)

    # 통계 계산
    total_rounds = len(results)

    # Ord 세그먼트별 평균
    avg_ord_top24 = sum(r['ord_top24'] for r in results) / total_rounds
    avg_ord_mid14 = sum(r['ord_mid14'] for r in results) / total_rounds
    avg_ord_rest7 = sum(r['ord_rest7'] for r in results) / total_rounds

    # Ball 세그먼트별 평균
    avg_ball_top24 = sum(r['ball_top24'] for r in results) / total_rounds
    avg_ball_mid14 = sum(r['ball_mid14'] for r in results) / total_rounds
    avg_ball_rest7 = sum(r['ball_rest7'] for r in results) / total_rounds

    # 세그먼트별 분포
    ord_top24_dist = Counter(r['ord_top24'] for r in results)
    ord_mid14_dist = Counter(r['ord_mid14'] for r in results)
    ord_rest7_dist = Counter(r['ord_rest7'] for r in results)
    ball_top24_dist = Counter(r['ball_top24'] for r in results)
    ball_mid14_dist = Counter(r['ball_mid14'] for r in results)
    ball_rest7_dist = Counter(r['ball_rest7'] for r in results)

    # full_code_distribution.csv 저장
    with open(OUTPUT_DIR / "full_code_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['code', 'frequency', 'ratio', 'probability'])
        for c, n in code_freq.most_common():
            writer.writerow([c, n, round(n/total_rounds*100, 1), n/total_rounds])

    # ord_code_distribution.csv 저장
    with open(OUTPUT_DIR / "ord_code_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['code', 'frequency', 'ratio', 'probability'])
        for c, n in ord_code_freq.most_common():
            writer.writerow([c, n, round(n/total_rounds*100, 1), n/total_rounds])

    # ball_code_distribution.csv 저장
    with open(OUTPUT_DIR / "ball_code_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['code', 'frequency', 'ratio', 'probability'])
        for c, n in ball_code_freq.most_common():
            writer.writerow([c, n, round(n/total_rounds*100, 1), n/total_rounds])

    # segment_distribution.csv 저장
    with open(OUTPUT_DIR / "segment_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['segment', 'count', 'frequency', 'ratio', 'probability'])
        for seg_name, seg_dist in [('ord_top24', ord_top24_dist), ('ord_mid14', ord_mid14_dist),
                                    ('ord_rest7', ord_rest7_dist), ('ball_top24', ball_top24_dist),
                                    ('ball_mid14', ball_mid14_dist), ('ball_rest7', ball_rest7_dist)]:
            for k, v in sorted(seg_dist.items()):
                writer.writerow([seg_name, k, v, round(v/total_rounds*100, 1), v/total_rounds])

    # summary.csv 저장
    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['total_rounds', total_rounds])
        writer.writerow(['unique_codes', len(code_freq)])
        writer.writerow(['unique_ord_codes', len(ord_code_freq)])
        writer.writerow(['unique_ball_codes', len(ball_code_freq)])
        writer.writerow(['avg_ord_top24', round(avg_ord_top24, 2)])
        writer.writerow(['avg_ord_mid14', round(avg_ord_mid14, 2)])
        writer.writerow(['avg_ord_rest7', round(avg_ord_rest7, 2)])
        writer.writerow(['avg_ball_top24', round(avg_ball_top24, 2)])
        writer.writerow(['avg_ball_mid14', round(avg_ball_mid14, 2)])
        writer.writerow(['avg_ball_rest7', round(avg_ball_rest7, 2)])

    print(f"=== 요약코드 인사이트 생성 완료 ===\n")
    print(f"총 {total_rounds}회차 분석")
    print(f"고유 요약코드: {len(code_freq)}개")
    print()

    print("[평균 적중 수]")
    print(f"  Ord  - Top24: {avg_ord_top24:.2f}, Mid14: {avg_ord_mid14:.2f}, Rest7: {avg_ord_rest7:.2f}")
    print(f"  Ball - Top24: {avg_ball_top24:.2f}, Mid14: {avg_ball_mid14:.2f}, Rest7: {avg_ball_rest7:.2f}")
    print()

    print("[가장 빈번한 요약코드 Top 10]")
    for code, count in most_common_codes[:10]:
        ratio = count / total_rounds * 100
        parsed = parse_shortcode(code)
        print(f"  {code}: {count}회 ({ratio:.1f}%) - Ord({parsed['ord_top24']},{parsed['ord_mid14']},{parsed['ord_rest7']}) Ball({parsed['ball_top24']},{parsed['ball_mid14']},{parsed['ball_rest7']})")

    print()
    print("CSV 파일 저장 완료")


def get_shortcode_for_round(round_num: int) -> str:
    """특정 회차의 요약코드 반환"""
    results = load_backtest_results()
    for r in results:
        if r['round'] == round_num:
            return r['summary']
    return None


if __name__ == "__main__":
    generate_insight()
