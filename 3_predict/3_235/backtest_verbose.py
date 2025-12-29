"""
백테스트 1회차 상세 분석

단계별로 어떻게 조합이 생성되고 평가되는지 상세히 출력
"""

import csv
from pathlib import Path
from collections import Counter

DATA_PATH = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"

# 소수 목록 (1~45)
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


def load_data():
    """당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'ords': [int(row[f'ord{i}']) for i in range(1, 7)],
                'balls': sorted([int(row[f'ball{i}']) for i in range(1, 7)]),
                'o_table': {i: int(row[f'o{i}']) for i in range(1, 46)}
            })
    return results


def get_recent_stats(data, n=50):
    """최근 N회차 통계"""
    recent = data[-n:]

    pos_freq = {pos: Counter() for pos in range(6)}
    for r in recent:
        for pos, ball in enumerate(r['balls']):
            pos_freq[pos][ball] += 1

    all_freq = Counter()
    for r in recent:
        for ball in r['balls']:
            all_freq[ball] += 1

    return pos_freq, all_freq


def get_range(n):
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4


def is_prime(n):
    return n in PRIMES


def score_combination_verbose(balls, o_table):
    """조합에 대한 점수 계산 (상세 출력)"""
    score = 0
    details = []

    ball1, ball2, ball3, ball4, ball5, ball6 = balls

    # 1. firstend
    if 1 <= ball1 <= 10:
        score += 10
        details.append(f"  firstend(ball1): +10 (ball1={ball1}, 1-10 범위)")
    if 38 <= ball6 <= 45:
        score += 10
        details.append(f"  firstend(ball6): +10 (ball6={ball6}, 38-45 범위)")

    # 2. second
    if 10 <= ball2 <= 19:
        score += 8
        details.append(f"  second: +8 (ball2={ball2}, 10-19 범위)")
    elif 1 <= ball2 <= 9:
        score += 6
        details.append(f"  second: +6 (ball2={ball2}, 1-9 범위)")

    if not is_prime(ball1) and not is_prime(ball2):
        score += 5
        details.append(f"  second_prime: +5 (ball1={ball1}, ball2={ball2} 둘 다 비소수)")

    # 3. third
    if 10 <= ball3 <= 19:
        score += 8
        details.append(f"  third: +8 (ball3={ball3}, 10-19 범위)")
    elif 20 <= ball3 <= 29:
        score += 7
        details.append(f"  third: +7 (ball3={ball3}, 20-29 범위)")

    # 4. fourth
    if 20 <= ball4 <= 29:
        score += 8
        details.append(f"  fourth: +8 (ball4={ball4}, 20-29 범위)")
    elif 30 <= ball4 <= 39:
        score += 7
        details.append(f"  fourth: +7 (ball4={ball4}, 30-39 범위)")

    prime_count_123 = sum(1 for b in [ball1, ball2, ball3] if is_prime(b))
    if prime_count_123 <= 1:
        score += 3
        details.append(f"  fourth_prime123: +3 (ball1-3 소수 {prime_count_123}개 ≤1)")

    # 5. fifth
    if 30 <= ball5 <= 39:
        score += 10
        details.append(f"  fifth: +10 (ball5={ball5}, 30-39 범위)")
    elif 20 <= ball5 <= 29:
        score += 5
        details.append(f"  fifth: +5 (ball5={ball5}, 20-29 범위)")

    sum14 = ball1 + ball2 + ball3 + ball4
    if sum14 <= 40 and 18 <= ball5 <= 33:
        score += 5
        details.append(f"  fifth_sum: +5 (sum1-4={sum14}≤40, ball5={ball5} 18-33)")
    elif 41 <= sum14 <= 70 and 22 <= ball5 <= 38:
        score += 5
        details.append(f"  fifth_sum: +5 (sum1-4={sum14} 41-70, ball5={ball5} 22-38)")
    elif sum14 >= 71 and 30 <= ball5 <= 41:
        score += 5
        details.append(f"  fifth_sum: +5 (sum1-4={sum14}≥71, ball5={ball5} 30-41)")

    # 6. consecutive
    consecutive_pairs = 0
    for i in range(5):
        if balls[i+1] - balls[i] == 1:
            consecutive_pairs += 1

    if consecutive_pairs == 1:
        score += 8
        details.append(f"  consecutive: +8 (연속수 1쌍)")
    elif consecutive_pairs == 0:
        score += 5
        details.append(f"  consecutive: +5 (연속수 없음)")
    elif consecutive_pairs == 2:
        score += 3
        details.append(f"  consecutive: +3 (연속수 2쌍)")

    # 7. lastnum
    if 40 <= ball6 <= 45:
        score += 10
        details.append(f"  lastnum: +10 (ball6={ball6}, 40-45 범위)")
    elif 30 <= ball6 <= 39:
        score += 5
        details.append(f"  lastnum: +5 (ball6={ball6}, 30-39 범위)")

    # 8. sum
    total_sum = sum(balls)
    if 121 <= total_sum <= 160:
        score += 10
        details.append(f"  sum: +10 (합계={total_sum}, 121-160 범위)")
    elif 100 <= total_sum <= 170:
        score += 5
        details.append(f"  sum: +5 (합계={total_sum}, 100-170 범위)")

    # 9. range
    ranges_used = len(set(get_range(b) for b in balls))
    if ranges_used == 4:
        score += 8
        details.append(f"  range: +8 (4개 구간 사용)")
    elif ranges_used == 5:
        score += 6
        details.append(f"  range: +6 (5개 구간 사용)")
    elif ranges_used == 3:
        score += 4
        details.append(f"  range: +4 (3개 구간 사용)")

    # 10. prime
    prime_count = sum(1 for b in balls if is_prime(b))
    if prime_count == 1 or prime_count == 2:
        score += 10
        details.append(f"  prime: +10 (소수 {prime_count}개)")
    elif prime_count == 3:
        score += 5
        details.append(f"  prime: +5 (소수 {prime_count}개)")

    # 11. shortcode
    ord_top24_count = 0
    for ball in balls:
        for ord_val, ball_val in o_table.items():
            if ball_val == ball and ord_val <= 24:
                ord_top24_count += 1
                break

    if 3 <= ord_top24_count <= 4:
        score += 10
        details.append(f"  shortcode: +10 (Top24에서 {ord_top24_count}개)")
    elif ord_top24_count == 5:
        score += 5
        details.append(f"  shortcode: +5 (Top24에서 {ord_top24_count}개)")

    # 12. onehot
    hot_bits = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
    cold_bits = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}

    hot_count = sum(1 for b in balls if b in hot_bits)
    cold_count = sum(1 for b in balls if b in cold_bits)

    if hot_count >= 2:
        score += hot_count * 2
        details.append(f"  onehot_hot: +{hot_count * 2} (핫 번호 {hot_count}개)")
    if cold_count >= 2:
        score -= cold_count
        details.append(f"  onehot_cold: -{cold_count} (콜드 번호 {cold_count}개)")

    return score, details


def run_verbose_backtest(target_round):
    """상세 백테스트 실행"""
    print("=" * 80)
    print(f"백테스트 상세 분석 - {target_round}회차")
    print("=" * 80)

    # 1. 데이터 로드
    print("\n[STEP 1] 데이터 로드")
    print("-" * 40)
    data = load_data()
    print(f"  총 데이터: {len(data)}회차 ({data[0]['round']} ~ {data[-1]['round']})")

    # 타겟 회차 인덱스 찾기
    target_idx = None
    for i, d in enumerate(data):
        if d['round'] == target_round:
            target_idx = i
            break

    if target_idx is None:
        print(f"  Error: {target_round}회차를 찾을 수 없습니다.")
        return

    # 해당 회차 정보
    current = data[target_idx]
    actual_balls = current['balls']
    print(f"\n  ★ {target_round}회차 실제 당첨번호: {actual_balls}")

    # 이전 데이터만 사용
    past_data = data[:target_idx]
    print(f"  사용 가능 데이터: {len(past_data)}회차 (~ {past_data[-1]['round']}회차)")

    # 2. 최근 50회 통계 계산
    print("\n[STEP 2] 최근 50회 빈도 통계")
    print("-" * 40)

    if len(past_data) < 50:
        print(f"  Error: 최소 50회 데이터 필요 (현재 {len(past_data)}회)")
        return

    pos_freq, all_freq = get_recent_stats(past_data, 50)

    print(f"  분석 범위: {past_data[-50]['round']}회차 ~ {past_data[-1]['round']}회차")

    # 전체 빈도 상위 10개
    print("\n  [전체 빈도 Top 10]")
    top10 = all_freq.most_common(10)
    for rank, (num, freq) in enumerate(top10, 1):
        in_actual = "★" if num in actual_balls else ""
        print(f"    {rank}. 번호 {num:2d}: {freq}회 {in_actual}")

    # 3. 포지션별 후보 생성
    print("\n[STEP 3] 포지션별 후보 번호 생성")
    print("-" * 40)

    latest = past_data[-1]
    o_table = latest['o_table']

    # ball1: 1-15 범위
    candidates = {}

    print("\n  [ball1 후보] 범위: 1-15")
    candidates['ball1'] = []
    for num in range(1, 16):
        score = pos_freq[0].get(num, 0) + all_freq.get(num, 0)
        candidates['ball1'].append((num, score))
    candidates['ball1'].sort(key=lambda x: -x[1])

    actual_ball1 = actual_balls[0]
    top10_b1 = [x[0] for x in candidates['ball1'][:10]]
    print(f"    상위 10개: {top10_b1}")
    if actual_ball1 in top10_b1:
        print(f"    ✓ 실제 ball1({actual_ball1}) 포함됨!")
    else:
        rank = None
        for i, (num, _) in enumerate(candidates['ball1']):
            if num == actual_ball1:
                rank = i + 1
                break
        if rank:
            print(f"    ✗ 실제 ball1({actual_ball1}) 미포함 (순위: {rank}위)")
        else:
            print(f"    ✗ 실제 ball1({actual_ball1}) 범위(1-15) 밖!")

    # ball2: 2-25 범위
    print("\n  [ball2 후보] 범위: 2-25")
    candidates['ball2'] = []
    for num in range(2, 26):
        score = pos_freq[1].get(num, 0) + all_freq.get(num, 0)
        candidates['ball2'].append((num, score))
    candidates['ball2'].sort(key=lambda x: -x[1])

    actual_ball2 = actual_balls[1]
    top15_b2 = [x[0] for x in candidates['ball2'][:15]]
    print(f"    상위 15개: {top15_b2}")
    if actual_ball2 in top15_b2:
        print(f"    ✓ 실제 ball2({actual_ball2}) 포함됨!")
    else:
        rank = None
        for i, (num, _) in enumerate(candidates['ball2']):
            if num == actual_ball2:
                rank = i + 1
                break
        if rank:
            print(f"    ✗ 실제 ball2({actual_ball2}) 미포함 (순위: {rank}위)")
        else:
            print(f"    ✗ 실제 ball2({actual_ball2}) 범위(2-25) 밖!")

    # ball3: 5-35 범위
    print("\n  [ball3 후보] 범위: 5-35")
    candidates['ball3'] = []
    for num in range(5, 36):
        score = pos_freq[2].get(num, 0) + all_freq.get(num, 0)
        candidates['ball3'].append((num, score))
    candidates['ball3'].sort(key=lambda x: -x[1])

    actual_ball3 = actual_balls[2]
    top18_b3 = [x[0] for x in candidates['ball3'][:18]]
    print(f"    상위 18개: {top18_b3}")
    if actual_ball3 in top18_b3:
        print(f"    ✓ 실제 ball3({actual_ball3}) 포함됨!")
    else:
        rank = None
        for i, (num, _) in enumerate(candidates['ball3']):
            if num == actual_ball3:
                rank = i + 1
                break
        if rank:
            print(f"    ✗ 실제 ball3({actual_ball3}) 미포함 (순위: {rank}위)")
        else:
            print(f"    ✗ 실제 ball3({actual_ball3}) 범위(5-35) 밖!")

    # ball4: 12-42 범위
    print("\n  [ball4 후보] 범위: 12-42")
    candidates['ball4'] = []
    for num in range(12, 43):
        score = pos_freq[3].get(num, 0) + all_freq.get(num, 0)
        candidates['ball4'].append((num, score))
    candidates['ball4'].sort(key=lambda x: -x[1])

    actual_ball4 = actual_balls[3]
    top18_b4 = [x[0] for x in candidates['ball4'][:18]]
    print(f"    상위 18개: {top18_b4}")
    if actual_ball4 in top18_b4:
        print(f"    ✓ 실제 ball4({actual_ball4}) 포함됨!")
    else:
        rank = None
        for i, (num, _) in enumerate(candidates['ball4']):
            if num == actual_ball4:
                rank = i + 1
                break
        if rank:
            print(f"    ✗ 실제 ball4({actual_ball4}) 미포함 (순위: {rank}위)")
        else:
            print(f"    ✗ 실제 ball4({actual_ball4}) 범위(12-42) 밖!")

    # ball5: 20-44 범위
    print("\n  [ball5 후보] 범위: 20-44")
    candidates['ball5'] = []
    for num in range(20, 45):
        score = pos_freq[4].get(num, 0) + all_freq.get(num, 0)
        candidates['ball5'].append((num, score))
    candidates['ball5'].sort(key=lambda x: -x[1])

    actual_ball5 = actual_balls[4]
    top15_b5 = [x[0] for x in candidates['ball5'][:15]]
    print(f"    상위 15개: {top15_b5}")
    if actual_ball5 in top15_b5:
        print(f"    ✓ 실제 ball5({actual_ball5}) 포함됨!")
    else:
        rank = None
        for i, (num, _) in enumerate(candidates['ball5']):
            if num == actual_ball5:
                rank = i + 1
                break
        if rank:
            print(f"    ✗ 실제 ball5({actual_ball5}) 미포함 (순위: {rank}위)")
        else:
            print(f"    ✗ 실제 ball5({actual_ball5}) 범위(20-44) 밖!")

    # ball6: 30-45 범위
    print("\n  [ball6 후보] 범위: 30-45")
    candidates['ball6'] = []
    for num in range(30, 46):
        score = pos_freq[5].get(num, 0) + all_freq.get(num, 0)
        candidates['ball6'].append((num, score))
    candidates['ball6'].sort(key=lambda x: -x[1])

    actual_ball6 = actual_balls[5]
    top12_b6 = [x[0] for x in candidates['ball6'][:12]]
    print(f"    상위 12개: {top12_b6}")
    if actual_ball6 in top12_b6:
        print(f"    ✓ 실제 ball6({actual_ball6}) 포함됨!")
    else:
        rank = None
        for i, (num, _) in enumerate(candidates['ball6']):
            if num == actual_ball6:
                rank = i + 1
                break
        if rank:
            print(f"    ✗ 실제 ball6({actual_ball6}) 미포함 (순위: {rank}위)")
        else:
            print(f"    ✗ 실제 ball6({actual_ball6}) 범위(30-45) 밖!")

    # 후보 포함 여부 요약
    print("\n  [후보 포함 요약]")
    included = []
    excluded = []
    for i, ball in enumerate(actual_balls, 1):
        cand_list = {
            1: top10_b1, 2: top15_b2, 3: top18_b3,
            4: top18_b4, 5: top15_b5, 6: top12_b6
        }[i]
        if ball in cand_list:
            included.append(ball)
        else:
            excluded.append(ball)

    print(f"    포함된 당첨번호: {included} ({len(included)}개)")
    print(f"    제외된 당첨번호: {excluded} ({len(excluded)}개)")

    if excluded:
        print(f"\n    ⚠️  제외된 번호가 있어 완전한 적중(6개)은 불가능!")

    # 4. 조합 생성
    print("\n[STEP 4] 조합 생성")
    print("-" * 40)

    b1_cand = [x[0] for x in candidates['ball1'][:10]]
    b2_cand = [x[0] for x in candidates['ball2'][:15]]
    b3_cand = [x[0] for x in candidates['ball3'][:18]]
    b4_cand = [x[0] for x in candidates['ball4'][:18]]
    b5_cand = [x[0] for x in candidates['ball5'][:15]]
    b6_cand = [x[0] for x in candidates['ball6'][:12]]

    total_possible = len(b1_cand) * len(b2_cand) * len(b3_cand) * len(b4_cand) * len(b5_cand) * len(b6_cand)
    print(f"  최대 조합 수 (중복 포함): {total_possible:,}")

    # 유효 조합만 생성
    all_combos = []

    for b1 in b1_cand:
        for b2 in b2_cand:
            if b2 <= b1:
                continue
            for b3 in b3_cand:
                if b3 <= b2:
                    continue
                for b4 in b4_cand:
                    if b4 <= b3:
                        continue
                    for b5 in b5_cand:
                        if b5 <= b4:
                            continue
                        for b6 in b6_cand:
                            if b6 <= b5:
                                continue

                            balls = [b1, b2, b3, b4, b5, b6]
                            score, _ = score_combination_verbose(balls, o_table)
                            all_combos.append({
                                'balls': balls,
                                'score': score
                            })

    print(f"  생성된 유효 조합: {len(all_combos):,}개")

    # 5. 점수순 정렬
    print("\n[STEP 5] 점수순 정렬 및 상위 500개 선정")
    print("-" * 40)

    all_combos.sort(key=lambda x: -x['score'])
    top500 = all_combos[:500]

    # 점수 분포
    score_dist = Counter(c['score'] for c in top500)
    print("  [상위 500개 점수 분포]")
    for score in sorted(score_dist.keys(), reverse=True)[:10]:
        cnt = score_dist[score]
        print(f"    {score}점: {cnt}개")

    # 6. 적중 수 계산
    print("\n[STEP 6] 적중 수 계산")
    print("-" * 40)

    match_counts = []
    for combo in top500:
        matches = len(set(combo['balls']) & set(actual_balls))
        match_counts.append(matches)
        combo['matches'] = matches

    match_dist = Counter(match_counts)
    print("  [적중 수 분포 (500개 중)]")
    for m in sorted(match_dist.keys(), reverse=True):
        cnt = match_dist[m]
        pct = cnt / 500 * 100
        print(f"    {m}개 적중: {cnt:3d}개 ({pct:5.1f}%)")

    # 최고 적중 조합
    best_match = max(match_counts)
    print(f"\n  ★ 최고 적중: {best_match}개")

    # 7. 상위 10개 조합 상세
    print("\n[STEP 7] 상위 10개 조합 상세")
    print("-" * 40)

    for i, combo in enumerate(top500[:10], 1):
        balls = combo['balls']
        score = combo['score']
        matches = combo['matches']
        matched_balls = set(balls) & set(actual_balls)

        balls_str = "-".join(f"{b:2d}" for b in balls)
        matched_str = ",".join(str(b) for b in sorted(matched_balls)) if matched_balls else "없음"

        print(f"\n  [{i}위] {balls_str} (점수: {score})")
        print(f"       적중: {matches}개 ({matched_str})")

        # 점수 상세 (1위만)
        if i == 1:
            print("       [점수 상세]")
            _, details = score_combination_verbose(balls, o_table)
            for d in details:
                print(f"       {d}")

    # 8. 3개 이상 적중 조합 예시
    print("\n[STEP 8] 3개 이상 적중 조합")
    print("-" * 40)

    match3_plus = [c for c in top500 if c['matches'] >= 3]
    print(f"  3개+ 적중 조합: {len(match3_plus)}개")

    if match3_plus:
        print("\n  [3개 이상 적중 조합 예시 (상위 5개)]")
        for i, combo in enumerate(match3_plus[:5], 1):
            balls = combo['balls']
            score = combo['score']
            matches = combo['matches']
            matched_balls = set(balls) & set(actual_balls)
            rank = top500.index(combo) + 1

            balls_str = "-".join(f"{b:2d}" for b in balls)
            matched_str = ",".join(str(b) for b in sorted(matched_balls))

            print(f"    {i}. [{rank}위] {balls_str} → {matches}개 적중 ({matched_str})")

    # 9. 실제 당첨번호 점수
    print("\n[STEP 9] 실제 당첨번호 분석")
    print("-" * 40)

    actual_score, actual_details = score_combination_verbose(actual_balls, o_table)
    print(f"  당첨번호: {actual_balls}")
    print(f"  점수: {actual_score}점")
    print("  [점수 상세]")
    for d in actual_details:
        print(f"  {d}")

    # 당첨번호가 500개 안에 있는지
    actual_in_top500 = False
    for i, combo in enumerate(top500):
        if combo['balls'] == actual_balls:
            actual_in_top500 = True
            print(f"\n  ★ 당첨번호가 상위 500개에 포함됨! (순위: {i+1}위)")
            break

    if not actual_in_top500:
        # 전체 조합에서 순위 찾기
        for i, combo in enumerate(all_combos):
            if combo['balls'] == actual_balls:
                print(f"\n  당첨번호 순위: {i+1}위 / {len(all_combos):,}개")
                break
        else:
            print(f"\n  ⚠️  당첨번호가 후보 조합에 포함되지 않음!")
            print(f"      (후보에서 제외된 번호: {excluded})")

    # 10. 결론
    print("\n" + "=" * 80)
    print(f"[결론] {target_round}회차 백테스트 결과")
    print("=" * 80)
    print(f"  당첨번호: {actual_balls}")
    print(f"  후보 포함: {len(included)}/6개")
    print(f"  최고 적중: {best_match}개 (상위 500개 중)")
    print(f"  3개+ 적중 조합: {len(match3_plus)}개")

    if best_match >= 4:
        print(f"\n  ✓ 4개 이상 적중 성공!")
    elif best_match == 3:
        print(f"\n  △ 3개 적중 (4등)")
    else:
        print(f"\n  ✗ 3개 미만 적중")

    if excluded:
        print(f"\n  개선 필요: 후보 범위/개수 확장 필요")
        print(f"    제외된 번호: {excluded}")


def main():
    import sys

    if len(sys.argv) > 1:
        target_round = int(sys.argv[1])
    else:
        target_round = 1204  # 기본값

    run_verbose_backtest(target_round)


if __name__ == "__main__":
    main()
