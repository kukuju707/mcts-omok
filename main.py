import numpy as np
import math
import random as r
from typing import List, Tuple, Dict, Optional

# ====== (Windows) 폰트/백엔드 ======
try:
    import matplotlib
    try:
        matplotlib.use("TkAgg", force=True)  # Windows에서 대화형 GUI 백엔드 강제
    except Exception:
        pass
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# ====== 전역 상수 ======
EMPTY, AI, OP = 0, 1, -1
DIRS = [(1,0), (0,1), (1,1), (-1,1)]

# (옵션) 탐색/롤아웃 안전장치 파라미터 — 기본값은 원래 동작과 동일하게 영향 없도록 설정
ENABLE_DIRICHLET_AT_ROOT = False   # 루트 Dirichlet 노이즈(개시 다양성) — 기본 OFF
DIRICHLET_ALPHA = 0.30
DIRICHLET_EPS = 0.25

ROLLOUT_DEPTH_CAP = None           # 롤아웃 최대 길이(정수). None이면 제한 없음
LATE_GAME_EPS_DECAY = False        # 후반부(빈칸 적을 때) ε 자동 감소 — 기본 OFF

# --------------------------- 기본 유틸 ---------------------------
def board_size(board: np.ndarray) -> Tuple[int,int]:
    """
    [목적] 보드의 세로/가로 크기(H, W)를 반환
    [입력] board: np.ndarray (정수 행렬)
    [출력] (H, W)
    """
    return board.shape

def in_bounds(H: int, W: int, x: int, y: int) -> bool:
    """
    [목적] 좌표(x,y)가 보드 범위 내인지 검사
    [입력] H,W: 보드 크기, x,y: 좌표
    [출력] bool
    """
    return 0 <= x < H and 0 <= y < W

def list_candidate_positions(board: np.ndarray, radius: int = 1) -> List[Tuple[int,int]]:
    """
    [목적] 현재 돌 주변 radius 이내의 빈칸 후보 좌표를 나열
    [전략] 초반엔 중앙부터, 이후엔 '가까운 근방만' 탐색해 분지 폭을 줄임
    [입력] board, radius(기본 1)
    [출력] (i,j) 후보 리스트(정렬)
    """
    H, W = board.shape
    stones = np.argwhere(board != EMPTY)
    if len(stones) == 0:
        # 첫 수는 중앙 추천
        return [(H//2, W//2)]
    cand = set()
    for x, y in stones:
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = int(x+dx), int(y+dy)
                if in_bounds(H,W,nx,ny) and board[nx,ny] == EMPTY:
                    cand.add((nx,ny))
    return sorted(cand)

def is_win_last(board: np.ndarray, who: int, i: int, j: int) -> bool:
    """
    [목적] (i,j)에 who가 마지막으로 둔 직후 '5목 완성' 여부 판정
    [입력] board, who(1/-1), i,j
    [출력] bool
    [주의] board[i,j]에 이미 who가 둔 상태라고 가정하지 않음 → 함수 내부에서 검사
    """
    H, W = board_size(board)
    for dx, dy in DIRS:
        cnt = 1
        # +방향 연속 카운트
        x, y = i+dx, j+dy
        while in_bounds(H,W,x,y) and board[x,y] == who:
            cnt += 1; x += dx; y += dy
        # -방향 연속 카운트
        x, y = i-dx, j-dy
        while in_bounds(H,W,x,y) and board[x,y] == who:
            cnt += 1; x -= dx; y -= dy
        if cnt >= 5:
            return True
    return False

def is_win_full(board: np.ndarray, who: int) -> bool:
    """
    [목적] '전체 보드'에서 who의 5목 완성 여부를 완전 탐색으로 확인
    [입력] board, who
    [출력] bool
    [비용] O(N^2) 완전탐색 — GUI 폴링 시 과도 호출되지 않도록 주의
    """
    H, W = board.shape
    for i in range(H):
        for j in range(W-4):
            if all(board[i, j+k] == who for k in range(5)): return True
    for i in range(H-4):
        for j in range(W):
            if all(board[i+k, j] == who for k in range(5)): return True
    for i in range(H-4):
        for j in range(W-4):
            if all(board[i+k, j+k] == who for k in range(5)): return True
    for i in range(4, H):
        for j in range(W-4):
            if all(board[i-k, j+k] == who for k in range(5)): return True
    return False

def max_line_len_after(board: np.ndarray, i: int, j: int, who: int) -> int:
    """
    [목적] (i,j)에 who가 뒀다고 가정할 때 형성되는 '최대 연속 길이'를 계산
    [입력] board, i,j, who
    [출력] 최대 연속 길이(int)
    [주의] 함수 내부에서 착수/철회 수행
    """
    board[i,j] = who
    best = 1
    H, W = board.shape
    for dx, dy in DIRS:
        c1 = 0; x, y = i+dx, j+dy
        while in_bounds(H,W,x,y) and board[x,y] == who:
            c1 += 1; x += dx; y += dy
        c2 = 0; x, y = i-dx, j-dy
        while in_bounds(H,W,x,y) and board[x,y] == who:
            c2 += 1; x -= dx; y -= dy
        best = max(best, 1 + c1 + c2)
    board[i,j] = EMPTY
    return best

def count_open_len_after(board: np.ndarray, i: int, j: int, who: int, need: int, steps: int) -> int:
    """
    [목적] (i,j)에 who가 둔 경우, 한 방향으로 steps 칸 내에서
           '막히지 않은(양끝이 열려있는) 길이 need' 패턴이 몇 개나 생기는지 근사 카운트
    [입력] need: 목표 길이(예: 3,4), steps: 한 방향 탐색 제한
    [출력] open 패턴 개수(int)
    [의미] 열린3/열린4, 이중3/이중4 판정에 활용
    """
    H, W = board_size(board)
    board[i, j] = who
    cnts = 0
    for dx, dy in DIRS:
        for sign in (1, -1):
            x, y = i, j
            cnt, block = 1, 0
            for _ in range(steps):
                x += dx*sign; y += dy*sign
                if in_bounds(H,W,x,y):
                    v = board[x,y]
                    if v == who: cnt += 1
                    elif v != EMPTY: block += 1; break
                else:
                    block += 1; break
            if cnt == need and block == 0:
                cnts += 1
    board[i, j] = EMPTY
    return cnts

# --------------------------- 규칙/위협 레이어 ---------------------------
def immediate_win_check(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] AI(AI=1)가 '한 수로 즉시 승리' 가능한 수가 있는지 검사
    [출력] 그러한 좌표 (i,j) 또는 None
    """
    for (i,j) in list_candidate_positions(board, 1):
        if board[i,j] != EMPTY: continue
        board[i,j] = AI
        ok = is_win_last(board, AI, i, j)
        board[i,j] = EMPTY
        if ok: return (i,j)
    return None

def block_opponent_immediate_win(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] 상대(OP=-1)가 '한 수로 즉시 승리' 가능한 수를 차단할 좌표 탐색
    [출력] 차단 좌표 (i,j) 또는 None
    """
    for (i,j) in list_candidate_positions(board, 1):
        if board[i,j] != EMPTY: continue
        board[i,j] = OP
        ok = is_win_last(board, OP, i, j)
        board[i,j] = EMPTY
        if ok: return (i,j)
    return None

def block_opponent_makes_four(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] 상대가 4를 만들 여지가 큰 수를 선제 차단(막힌4/열린4 포함 근사)
    [출력] 차단 좌표 또는 None
    """
    for (i,j) in list_candidate_positions(board, 1):
        if board[i,j] != EMPTY: continue
        if max_line_len_after(board, i, j, OP) >= 4:
            return (i,j)
    return None

def creates_one_move_win(board: np.ndarray, who: int) -> Optional[Tuple[int,int]]:
    """
    [목적] '지금 한 수 + 다음 한 수' 조합으로 즉승 가능한 '첫 수'를 찾음
    [설명] (i) 지금 (x,y) 두고 → (ii) 다음 (i,j) 두면 승리인지 검사
    [출력] 그러한 첫 수 (x,y) 또는 None
    """
    for (x,y) in list_candidate_positions(board, 1):
        if board[x,y] != EMPTY: continue
        board[x,y] = who
        for (i,j) in list_candidate_positions(board, 1):
            if board[i,j] != EMPTY: continue
            board[i,j] = who
            ok = is_win_last(board, who, i, j)
            board[i,j] = EMPTY
            if ok:
                board[x,y] = EMPTY
                return (x,y)
        board[x,y] = EMPTY
    return None

def block_opponent_forced_in_two(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] 상대의 '2수 강제승(한 수 선행)' 진입 수를 차단
    [구현] 상대에 대해 creates_one_move_win 재사용
    """
    return creates_one_move_win(board, OP)

def create_forced_in_two(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] AI가 '2수 강제승' 시퀀스에 진입하는 첫 수를 탐색
    [출력] 해당 좌표 또는 None
    """
    for (i,j) in list_candidate_positions(board, 1):
        if board[i,j] != EMPTY: continue
        board[i,j] = AI
        nxt = creates_one_move_win(board, AI)
        board[i,j] = EMPTY
        if nxt is not None: return (i,j)
    return None

def create_double_four(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] AI가 '이중4(동시에 4 두 줄)'을 만드는 수를 탐색
    [출력] 좌표 또는 None
    """
    for (i,j) in list_candidate_positions(board, 1):
        if board[i,j] != EMPTY: continue
        if count_open_len_after(board, i,j, AI, 4, 3) >= 2: return (i,j)
    return None

def block_opponent_double_four(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] 상대의 '이중4' 형성을 사전 차단
    """
    for (i,j) in list_candidate_positions(board, 1):
        if board[i,j] != EMPTY: continue
        if count_open_len_after(board, i,j, OP, 4, 3) >= 2: return (i,j)
    return None

def extend_my_open_three(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] AI의 '열린3'을 '열린4/승리 가능한 모양'으로 확장시키는 수 탐색
    """
    for (i,j) in list_candidate_positions(board, 1):
        if board[i,j] != EMPTY: continue
        a = count_open_len_after(board, i,j, AI, 3, 2)
        b = count_open_len_after(board, i,j, AI, 4, 3)
        if a >= 1 and b >= 1: return (i,j)
    return None

def block_extend_opponent_open_three(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] 상대의 '열린3' 확장을 차단(열린4로 진행 방지)
    """
    for (i,j) in list_candidate_positions(board, 1):
        if board[i,j] != EMPTY: continue
        a = count_open_len_after(board, i,j, OP, 3, 2)
        b = count_open_len_after(board, i,j, OP, 4, 3)
        if a >= 1 and b >= 1: return (i,j)
    return None

def create_double_three(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] AI의 '이중3' 형성(두 방향의 열린3 동시)을 추구하는 수 탐색
    """
    for (i,j) in list_candidate_positions(board, 1):
        if board[i,j] != EMPTY: continue
        if count_open_len_after(board, i,j, AI, 3, 2) >= 2: return (i,j)
    return None

def block_opponent_double_three(board: np.ndarray) -> Optional[Tuple[int,int]]:
    """
    [목적] 상대의 '이중3' 형성을 차단하는 수 탐색
    """
    for (i,j) in list_candidate_positions(board, 1):
        if board[i,j] != EMPTY: continue
        if count_open_len_after(board, i,j, OP, 3, 2) >= 2: return (i,j)
    return None

# 규칙/위협 적용 순서 — 전술적 우선순위 반영
RULE_ORDER = [
    immediate_win_check,
    block_opponent_immediate_win,
    block_opponent_makes_four,
    block_opponent_forced_in_two,
    create_forced_in_two,
    block_opponent_double_four,
    create_double_four,
    block_extend_opponent_open_three,
    extend_my_open_three,
    block_opponent_double_three,
    create_double_three,
]

# --------------------------- 휴리스틱/플레이아웃 ---------------------------
def move_score(board: np.ndarray, i: int, j: int, who: int) -> float:
    """
    [목적] 후보 수(i,j)에 대한 휴리스틱 점수(우선도)를 계산
    [전략] 즉승/즉사 차단에 큰 가중치, 다음으로 4/이중4/열린3 가중
    [출력] 실수 점수(크면 우선)
    """
    if board[i,j] != EMPTY:
        return -1e12
    score = 0.0

    # 즉승
    board[i,j] = who
    if is_win_last(board, who, i, j):
        board[i,j] = EMPTY
        return 1e9

    # 상대 즉승 차단
    board[i,j] = -who
    opp_win = is_win_last(board, -who, i, j)
    board[i,j] = who
    if opp_win:
        score += 1e7

    # 4 / 이중4 / 열린3
    mk4 = (max_line_len_after(board, i, j, who) >= 4)
    if mk4: score += 50_000

    d4  = count_open_len_after(board, i, j, who, 4, 3)
    d3  = count_open_len_after(board, i, j, who, 3, 2)
    score += 30_000 * max(0, d4-1)   # 이중4
    score +=   600 * max(0, d3-1)    # 이중3
    score +=   120 * d3 + 220 * d4

    board[i,j] = EMPTY
    return score

def _epsilon_dynamic(board: np.ndarray, base_eps: float = 0.10) -> float:
    """
    [목적] 후반부일수록 ε(무작위 비율)를 낮춰 '막판 실수'를 줄임(옵션)
    [설정] LATE_GAME_EPS_DECAY 플래그가 True일 때만 적용
    """
    if not LATE_GAME_EPS_DECAY:
        return base_eps
    empties = int(np.sum(board == EMPTY))
    if empties <= 20:
        return max(0.02, base_eps * 0.5)
    if empties <= 40:
        return max(0.04, base_eps * 0.75)
    return base_eps

def random_playout(board: np.ndarray, turn: int) -> int:
    """
    [목적] ε-greedy 휴리스틱 롤아웃으로 끝(승/무/패)까지 시뮬레이션
    [전략]
      - 즉승/차단 우선
      - 그 외엔 휴리스틱 최고점 or ε 확률로 랜덤
      - 동적 후보폭: 초반엔 radius=2, 이후엔 radius=1 위주
      - (옵션) 깊이 제한/후반 ε 감소
    [출력] 승자(AI=1/OP=-1) 또는 0(무승부)
    """
    last: Optional[Tuple[int,int]] = None
    empties_remaining = int(np.sum(board == EMPTY))
    EPS = _epsilon_dynamic(board, base_eps=0.10)

    step = 0
    while True:
        # 직전 수가 승리로 끝났는지 확인
        if last is not None:
            li, lj = last
            if is_win_last(board, -turn, li, lj):
                return -turn
        # 무승부(빈칸 없음)
        if empties_remaining == 0:
            return 0

        # (옵션) 롤아웃 깊이 제한 
        if (ROLLOUT_DEPTH_CAP is not None) and (step >= ROLLOUT_DEPTH_CAP):
            return 0  # 보수적으로 무승부 처리
        step += 1

        # 동적 후보폭
        stones = int(np.sum(board != EMPTY))
        if stones < 8:
            cand = list_candidate_positions(board, 2)
        else:
            cand = list_candidate_positions(board, 1) or list_candidate_positions(board, 2)

        # 즉승
        picked = None
        for (i, j) in cand:
            board[i, j] = turn
            w = is_win_last(board, turn, i, j)
            board[i, j] = EMPTY
            if w: picked = (i, j); break

        if picked is None:
            # ε-greedy: (1-ε) 휴리스틱 최고, ε 랜덤
            if r.random() > EPS:
                best_s, picked = -1e18, None
                for (i, j) in cand:
                    s = move_score(board, i, j, turn)
                    if s > best_s:
                        best_s, picked = s, (i, j)
            else:
                picked = r.choice(cand)

        # 착수
        i, j = picked
        board[i, j] = turn
        last = (i, j)
        empties_remaining -= 1
        turn = -turn

# --------------------------- PUCT MCTS ---------------------------
CPUCT = 1.6      # 탐색 상수 (높을수록 탐험↑)
BIAS_K = 0.002   # 진입 편향 강도 (prior 스케일링)

def _apply_dirichlet(priors: List[float]) -> List[float]:
    """
    [목적] 루트에서 prior에 Dirichlet 노이즈를 섞어 개시 다양성 확보(옵션)
    [출력] 혼합된 priors
    """
    if not priors or not ENABLE_DIRICHLET_AT_ROOT:
        return priors
    noise = np.random.default_rng().dirichlet([DIRICHLET_ALPHA] * len(priors))
    eps = DIRICHLET_EPS
    mixed = [(1 - eps) * p + eps * float(n) for p, n in zip(priors, noise)]
    s = sum(mixed)
    return [m/s for m in mixed] if s > 0 else priors

def create_children(board: np.ndarray, who: int, is_root: bool=False) -> List[Tuple[np.ndarray, Tuple[int,int], float]]:
    """
    [목적] 현재 보드에서 착수 후보/자식 노드 생성 + prior 계산
    [전략]
      - 초반은 radius=2, 이후엔 radius=1 위주
      - prior = max(0, move_score) 정규화
      - (루트 한정) Dirichlet 노이즈 옵션
    [출력] 리스트[(자식보드, 수좌표, prior)]
    """
    stones = int(np.sum(board != EMPTY))
    if stones < 8:
        cand = list_candidate_positions(board, 2)
    else:
        cand = list_candidate_positions(board, 1) or list_candidate_positions(board, 2)

    triples: List[Tuple[np.ndarray, Tuple[int,int], float]] = []
    priors = []
    for (i,j) in cand:
        if board[i,j] != EMPTY: continue
        prior = max(0.0, move_score(board, i, j, who))
        priors.append(prior)

    if len(priors) == 0:
        return []
    s = sum(priors)
    if s == 0:
        norm = [1.0/len(priors)]*len(priors)
    else:
        norm = [p/s for p in priors]

    # 루트에서만 Dirichlet 적용(옵션)
    if is_root:
        norm = _apply_dirichlet(norm)

    idx = 0
    for (i,j) in cand:
        if board[i,j] != EMPTY: continue
        nb = board.copy(); nb[i,j] = who
        triples.append((nb, (i,j), norm[idx]))
        idx += 1
    return triples

def mcts_decide_rollouts(board: np.ndarray, rollouts: int = 1200) -> Optional[Tuple[int,int]]:
    """
    [목적] 루트에서 PUCT-MCTS를 rollouts 횟수만큼 수행하여 최종 수 선택
    [구현]
      - 방문/승수/부모방문/priors 테이블로 통계 유지
      - 선택: Q + c*P*sqrt(Nparent)/(1+N) + bias
      - 시뮬레이션: ε-greedy 휴리스틱 롤아웃
      - 역전파: 루트(AI) 관점 보상(+1/0/-1) 누적
    [출력] (i,j) 또는 None
    """
    visits: Dict[Tuple[bytes,int,int], int] = {}
    wins: Dict[Tuple[bytes,int,int], float] = {}
    parent_visits: Dict[Tuple[bytes,int], int] = {}
    priortable: Dict[Tuple[bytes,int,int], float] = {}

    def pkey(b: np.ndarray) -> bytes:
        """[목적] 보드 상태를 바이트로 직렬화해 해시 키로 사용(간단한 상태키)"""
        return b.tobytes()

    root_key = pkey(board)
    root_children = create_children(board, AI, is_root=True)
    if not root_children:
        return None

    for _ in range(rollouts):
        path = []
        depth = 0
        cur = board.copy()
        pk = root_key
        turn = AI
        parent_visits[(pk, depth)] = parent_visits.get((pk, depth), 0) + 1

        # ===== 선택/확장 루프 =====
        while True:
            children = create_children(cur, turn, is_root=False)
            if not children:
                break

            pN = parent_visits[(pk, depth)]
            best_idx, best_score = None, -1e18

            for idx, (nb, mv, prior) in enumerate(children):
                key = (pk, depth, idx)
                if key not in visits:
                    visits[key] = 0; wins[key] = 0.0; priortable[key] = prior

                N = visits[key]
                W = wins[key]
                Q = (W / N) if N > 0 else 0.0

                # PUCT: Q + c * P * sqrt(Nparent)/(1+N)
                U = CPUCT * priortable[key] * math.sqrt(max(1, pN)) / (1 + N)

                # 추가 편향(서서히 약화) — prior가 완전 고사되지 않도록
                bias = BIAS_K * priortable[key] / (1 + N)

                score = Q + U + bias
                if score > best_score:
                    best_score = score; best_idx = idx

            # 선택된 자식으로 전개
            visits[(pk, depth, best_idx)] = visits.get((pk, depth, best_idx), 0) + 1
            path.append((pk, depth, best_idx, children))
            cur, _mv, _prior = children[best_idx]
            pk = pkey(cur)
            depth += 1
            parent_visits[(pk, depth)] = parent_visits.get((pk, depth), 0) + 1
            turn = -turn

        # ===== 시뮬레이션(롤아웃) =====
        result = random_playout(cur.copy(), turn)
        reward = 1.0 if result == AI else (0.0 if result == 0 else -1.0)

        # ===== 역전파 =====
        for (ppk, dd, idx, _) in path:
            wins[(ppk, dd, idx)] = wins.get((ppk, dd, idx), 0.0) + reward

    # ===== 루트에서 최고 승률 수 반환 =====
    best_mv, best_val = None, -1e18
    for idx, (_nb, mv, _p) in enumerate(root_children):
        key = (root_key, 0, idx)
        v = visits.get(key, 0)
        w = wins.get(key, 0.0)
        val = (w / v) if v else -1e18
        if val > best_val:
            best_val, best_mv = val, mv
    return best_mv

# --------------------------- 미니 위협 탐색(2-ply) ---------------------------
def mini_threat_search(board: np.ndarray, who: int) -> Optional[Tuple[int,int]]:
    """
    [목적] 값싼 2-ply 위협 탐색으로 전술 실수를 줄이고 압박 수를 선호
    [전개]
      1) 즉승/즉사 우선 처리
      2) 내 수 → 상대 최선 응수 → 내 수에서 즉승/이중4/열린3 최대치 근사
    [출력] 위협 가치가 양(>0)인 최선 수 또는 None
    """
    # 1) 즉승/즉사 먼저
    win_now = immediate_win_check(board)
    if win_now: return win_now
    block_now = block_opponent_immediate_win(board)
    if block_now: return block_now

    # 2) 후보 중 위협 수 탐색
    cand = list_candidate_positions(board, 1) or list_candidate_positions(board, 2)
    best_mv, best_score = None, -1e18
    for (i,j) in cand:
        if board[i,j] != EMPTY: continue
        board[i,j] = who
        if is_win_last(board, who, i, j):
            board[i,j] = EMPTY
            return (i,j)

        # 상대 응수 후보
        opp_cand = list_candidate_positions(board, 1) or list_candidate_positions(board, 2)
        worst_for_me = +1e18
        for (u,v) in opp_cand:
            if board[u,v] != EMPTY: continue
            board[u,v] = -who
            if is_win_last(board, -who, u, v):
                sc = -1e12 
            else:
                # 다음 내 차례: 즉승/이중4/열린3 최대치 평가
                next_cand = list_candidate_positions(board, 1) or list_candidate_positions(board, 2)
                good = 0.0
                for (a,b) in next_cand:
                    if board[a,b] != EMPTY: continue
                    board[a,b] = who
                    if is_win_last(board, who, a, b):
                        good = max(good, 1e8)
                    else:
                        d4 = count_open_len_after(board, a, b, who, 4, 3)
                        d3 = count_open_len_after(board, a, b, who, 3, 2)
                        good = max(good, 5000*d4 + 300*d3)
                    board[a,b] = EMPTY
                sc = good
            board[u,v] = EMPTY
            worst_for_me = min(worst_for_me, sc)

        board[i,j] = EMPTY
        if worst_for_me > best_score:
            best_score = worst_for_me; best_mv = (i,j)

    if best_mv is not None and best_score > 0:
        return best_mv
    return None

# --------------------------- 메인 의사결정 ---------------------------
def MCTS_engine(board: np.ndarray) -> Tuple[int,int]:
    """
    [목적] 한 수를 결정하는 엔진
    [전략]
      A) 규칙 레이어(즉승/즉사 등) 즉시 처리
      B) 2-ply 위협 탐색으로 값싼 전술 수읽기
      C) 남는 경우 PUCT-MCTS(1200롤아웃)로 결정
      D) 그래도 없으면 랜덤 폴백(주변 후보)
    [출력] (i,j)
    """
    # 첫 수는 중앙
    if not np.any(board != EMPTY):
        H, W = board_size(board)
        return (H//2, W//2)

    # (A) 규칙 레이어
    for fn in RULE_ORDER:
        mv = fn(board)
        if mv:
            return mv

    # (B) 미니 위협 탐색(2-ply)
    mv = mini_threat_search(board, AI)
    if mv:
        return mv

    # (C) PUCT-MCTS 
    mv = mcts_decide_rollouts(board, rollouts=1200)
    if mv:
        return mv

    # (D) 폴백
    cand = list_candidate_positions(board, 1) or list_candidate_positions(board, 2)
    return r.choice(cand)

# --------------------------- GUI ---------------------------
def run_gui(size: int = 15, human_color: int = OP):
    """
    [목적] 간단한 대국 GUI 실행(사람이 두고 → AI 응수)
    [안정화]
      - 타이머 1회(poll)로 백그라운드 결과를 안전하게 소비
      - 스레드 락으로 보드 동시 접근 방지
      - 중복 착수/다수 착 방지
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import threading

    board = np.zeros((size, size), dtype=int)
    ai_color = AI if human_color == OP else OP
    turn = ai_color if human_color == OP else human_color
    board_lock = threading.Lock()

    fig, ax = plt.subplots(figsize=(6.8,6.8))
    ax.set_xticks(range(size)); ax.set_yticks(range(size))
    ax.set_xlim(-0.5, size-0.5); ax.set_ylim(-0.5, size-0.5)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.grid(True)

    ai_label = '검정' if ai_color == AI else '파랑'
    human_label = '검정' if human_color == AI else '파랑'
    default_title = f"오목 — (AI: {ai_label}, 사람: {human_label})  | sims=1200 + threat"
    ax.set_title(default_title)

    stones: Dict[Tuple[int,int], object] = {}
    last_circle = None
    ai_thinking = False
    ai_result = {"move": None}  # 백그라운드 스레드 ↔ 타이머 간 결과 전달 버퍼

    def refresh():
        """[목적] 캔버스 즉시 갱신(레이턴시 최소화)"""
        fig.canvas.draw(); fig.canvas.flush_events()

    def draw_stone(i: int, j: int, who: int):
        """
        [목적] (i,j)에 시각적 돌 표기
        [주의] 동일 좌표 중복 그리기 방지
        """
        if (i,j) in stones: return
        ch = '●'
        color = 'black' if who == AI else 'blue'
        t = ax.text(j, i, ch, ha='center', va='center', fontsize=18, color=color)
        stones[(i,j)] = t

    def mark_last(i: int, j: int):
        """
        [목적] 마지막 수를 빨간 원으로 강조
        """
        nonlocal last_circle
        if last_circle is not None:
            try: last_circle.remove()
            except Exception: pass
        last_circle = Circle((j,i), 0.40, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(last_circle)
        refresh()

    def end_if_finished_full() -> bool:
        """
        [목적] 승패/무승부 판정 후 타이틀 업데이트 및 종료 이벤트 트리거
        [출력] True면 게임 종료 상태
        """
        with board_lock:
            ai_w = is_win_full(board, ai_color)
            hm_w = is_win_full(board, human_color)
            draw = not np.any(board == EMPTY)
        if ai_w:
            ax.set_title("AI 승리! (창 닫기)"); refresh(); return True
        if hm_w:
            ax.set_title("사람 승리! (창 닫기)"); refresh(); return True
        if draw:
            ax.set_title("무승부! (창 닫기)"); refresh(); return True
        return False

    # 타이머 1회 — 백그라운드 탐색 결과를 안전하게 수거
    ai_timer = fig.canvas.new_timer(interval=70)

    def poll():
        """
        [목적] (주기적) 백그라운드 스레드가 계산한 AI 수를 수거하여 보드에 반영
        [주의] 결과 소비 후 버퍼 초기화, 중복착수 시 안전 재탐색
        """
        nonlocal ai_thinking, turn
        mv = ai_result["move"]
        if mv is None: return

        # 결과 소비(한 번만 반영)
        ai_result["move"] = None

        need_retry = False
        with board_lock:
            i, j = mv
            if board[i,j] != EMPTY:
                need_retry = True  # 중복 착수 방어
            else:
                board[i,j] = ai_color
                turn = -turn

        if need_retry:
            ai_thinking = False
            ax.set_title(default_title); refresh()
            start_ai_search()  # 안전 재탐색
            return

        draw_stone(i,j,ai_color)
        mark_last(i,j)
        ax.set_title(default_title); refresh()
        ai_thinking = False
        end_if_finished_full()

    ai_timer.add_callback(poll)
    ai_timer.start()

    def start_ai_search():
        """
        [목적] AI 탐색 스레드를 기동(중복 기동 방지)
        [절차] 보드 스냅샷을 락 내에서 복사 → 백그라운드에서 MCTS_engine 호출
        """
        nonlocal ai_thinking
        if ai_thinking: return
        ai_thinking = True
        ai_result["move"] = None
        ax.set_title("AI 계산 중… (느리게 생각 중)"); refresh()

        def worker():
            with board_lock:
                snap = board.copy()
            mv = MCTS_engine(snap)
            ai_result["move"] = mv

        import threading as th
        th.Thread(target=worker, daemon=True).start()

    def on_click(event):
        """
        [목적] 사람의 클릭 입력 처리
        [로직] 범위/중복 검사 → 사람 착수 → 승리 판정 → AI 탐색 시작
        """
        nonlocal turn
        if event.inaxes != ax: return
        if ai_thinking: return
        if end_if_finished_full(): return
        if turn != human_color: return

        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        j = int(round(x)); i = int(round(y))
        j = max(0, min(size-1, j)); i = max(0, min(size-1, i))

        with board_lock:
            if board[i,j] != EMPTY:
                return
            board[i,j] = human_color
            my_win = is_win_last(board, human_color, i, j)
            turn = -turn

        draw_stone(i,j,human_color)
        mark_last(i,j)
        if my_win:
            ax.set_title("사람 승리! (창 닫기)"); refresh(); return
        if end_if_finished_full():
            return
        start_ai_search()

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)

    def on_close(_evt):
        """
        [목적] 창 닫힘 이벤트 시 타이머/핸들러 정리
        """
        try: ai_timer.stop()
        except Exception: pass
        try: fig.canvas.mpl_disconnect(cid_click)
        except Exception: pass

    fig.canvas.mpl_connect('close_event', on_close)

    # 선공이 AI이면 중앙 선착
    if turn == ai_color:
        with board_lock:
            H, W = board_size(board)
            ci, cj = (H//2, W//2)
            if board[ci, cj] == EMPTY:
                board[ci, cj] = ai_color
                turn = -turn
        draw_stone(ci, cj, ai_color)
        mark_last(ci, cj)

    import matplotlib.pyplot as plt
    plt.show()

# --------------------------- Entry ---------------------------
if __name__ == "__main__":
    # 기본: 사람이 파랑(OP=-1), AI(검정=1) 선공
    run_gui(size=15, human_color=OP)
    # 사람이 선공(검정)으로 두고 싶다면:
    # run_gui(size=15, human_color=AI)

