# 5_cluster - 5개 공유 그룹 클러스터링

당첨번호 없이 500만 조합에서 유력 조합군 식별

## 핵심 아이디어

```
500만 조합 / 150 (평균 5개 일치) = ~33,000개 그룹

[a,b,c,d,e,f] → 6개의 5-키 생성:
- (b,c,d,e,f) : 1번 제외
- (a,c,d,e,f) : 2번 제외
- (a,b,d,e,f) : 3번 제외
- (a,b,c,e,f) : 4번 제외
- (a,b,c,d,f) : 5번 제외
- (a,b,c,d,e) : 6번 제외
```

## 파일

| 파일 | 설명 |
|------|------|
| cluster.py | 클러스터링 로직 |
| result/clusters.csv | 상위 클러스터 결과 |

## 사용법

```bash
# 단일 회차 클러스터링
python main.py --cluster --round 1205

# 백테스트
python main.py --cluster --start 1195 --end 1204
```

## 출력 (clusters.csv)

| 컬럼 | 설명 |
|------|------|
| rank | 클러스터 순위 (크기 내림차순) |
| five_key | 5개 공유 번호 |
| cluster_size | 클러스터 크기 |
| has_5match | 당첨번호 5개 포함 여부 |
| has_6match | 6개 일치 조합 포함 여부 |
| top_excluded | 가장 많이 제외된 번호들 |

## 핵심 함수

| 함수 | 설명 |
|------|------|
| generate_5keys() | 6개 조합에서 6개 5-키 생성 |
| cluster_combinations() | 전체 조합 클러스터링 |
| get_top_clusters() | 상위 N개 클러스터 추출 |
| check_winning_in_cluster() | 당첨번호 포함 여부 확인 |
