# main.py — Gomoku (Omok) with Rule-based Heuristics + Light MCTS + Matplotlib Click GUI
# Keeps your original structure; adds threading + speed tweaks
# Requirements: pip install numpy matplotlib

import numpy as np
import random as r
import math
from typing import List, Tuple, Optional, Dict

# ====== (Windows) 한글 폰트 설정 ======
try:
    import matplotlib
    # matplotlib.use('TkAgg')  # <- 백엔드 이슈 있으면 주석 해제
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

EMPTY, AI, OP = 0, 1, -1
DIRS = [(1,0), (0,1), (1,1), (-1,1)]

# ---------------- Board / Utils ----------------
def board_size(board: np.ndarray) -> Tuple[int, int]:
    return board.shape

def in_bounds(H, W, x, y) -> bool:
    return 0 <= x < H and 0 <= y < W

def is_win_full(board: np.ndarray, who: int) -> bool:
    """전체 스캔(결과 표시/보수용)"""
    H, W = board_size(board)
    for i in range(H):
        for j in range(W - 4):
            if all(board[i, j + k] == who for k in range(5)): return True
    for i in range(H - 4):
        for j in range(W):
            if all(board[i + k, j] == who for k in range(5)): return True
    for i in range(H - 4):
        for j in range(W - 4):
            if all(board[i + k, j + k] == who for k in range(5)): return True
    for i in range(4, H):
        for j in range(W - 4):
            if all(board[i - k, j + k] == who for k in range(5)): return True
    return False

def is_win_last(board: np.ndarray, who: int, i: int, j: int) -> bool:
    """마지막 착수(i,j) 기준 빠른 승리 판정"""
    H, W = board_size(board)
    for dx, dy in DIRS:
        cnt = 1
        x, y = i + dx, j + dy
        while in_bounds(H, W, x, y) and board[x, y] == who:
            cnt += 1; x += dx; y += dy
        x, y = i - dx, j - dy
        while in_bounds(H, W, x, y) and board[x, y] == who:
            cnt += 1; x -= dx; y -= dy
        if cnt >= 5: return True
    return False

# -------- 후보 생성 (캐시) --------
_CAND_CACHE: Dict[bytes, List[Tuple[int,int]]] = {}
def _cache_key(board: np.ndarray) -> bytes:
    return board.tobytes()

def get_candidate_positions(board: np.ndarray, radius: int = 1) -> List[Tuple[int,int]]:
    """착점 인접 반경 내 빈칸만 후보. 캐시로 가속."""
    key = _cache_key(board)
    if key in _CAND_CACHE:
        return _CAND_CACHE[key]

    H, W = board.shape
    stones = np.argwhere(board != EMPTY)
    if len(stones) == 0:
        cands = [(H // 2, W // 2)]
        _CAND_CACHE[key] = cands
        return cands

    cand = set()
    for x, y in stones:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if in_bounds(H, W, nx, ny) and board[nx, ny] == EMPTY:
                    cand.add((nx, ny))
    if not cand and radius == 1:
        # 반경 2로 확장
        for x, y in stones:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if in_bounds(H, W, nx, ny) and board[nx, ny] == EMPTY:
                        cand.add((nx, ny))

    out = sorted(cand) if cand else []
    _CAND_CACHE[key] = out
    return out

# --------------- Rule-based layer (네 순서 유지) ---------------
def immediate_win_check(board: np.ndarray):
    for (i, j) in get_candidate_positions(board, 1):
        if board[i, j] != EMPTY: continue
        board[i, j] = AI
        ok = is_win_last(board, AI, i, j)
        board[i, j] = EMPTY
        if ok: return (i, j)
    return None

def block_opponent_immediate_win(board: np.ndarray):
    for (i, j) in get_candidate_positions(board, 1):
        if board[i, j] != EMPTY: continue
        board[i, j] = OP
        ok = is_win_last(board, OP, i, j)
        board[i, j] = EMPTY
        if ok: return (i, j)
    return None

def count_open_len_after(board: np.ndarray, i: int, j: int, who: int, need: int, steps: int) -> int:
    """(i,j) 착수 시 길이 need(3/4) '열린' 형태 근사 카운트."""
    H, W = board_size(board)
    board[i, j] = who
    cnts = 0
    for dx, dy in DIRS:
        for sign in (1, -1):
            x, y = i, j
            cnt, block = 1, 0
            for _ in range(steps):
                x += dx * sign; y += dy * sign
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

def create_double_four(board: np.ndarray):
    for (i, j) in get_candidate_positions(board, 1):
        if board[i, j] != EMPTY: continue
        if count_open_len_after(board, i, j, AI, 4, 3) >= 2: return (i, j)
    return None

def block_opponent_double_four(board: np.ndarray):
    for (i, j) in get_candidate_positions(board, 1):
        if board[i, j] != EMPTY: continue
        if count_open_len_after(board, i, j, OP, 4, 3) >= 2: return (i, j)
    return None

def create_double_three(board: np.ndarray):
    for (i, j) in get_candidate_positions(board, 1):
        if board[i, j] != EMPTY: continue
        if count_open_len_after(board, i, j, AI, 3, 2) >= 2: return (i, j)
    return None

def block_opponent_double_three(board: np.ndarray):
    for (i, j) in get_candidate_positions(board, 1):
        if board[i, j] != EMPTY: continue
        if count_open_len_after(board, i, j, OP, 3, 2) >= 2: return (i, j)
    return None

def extend_my_open_three(board: np.ndarray):
    for (i, j) in get_candidate_positions(board, 1):
        if board[i, j] != EMPTY: continue
        a = count_open_len_after(board, i, j, AI, 3, 2)
        b = count_open_len_after(board, i, j, AI, 4, 3)
        if a >= 1 and b >= 1: return (i, j)
    return None

def block_extend_opponent_open_three(board: np.ndarray):
    for (i, j) in get_candidate_positions(board, 1):
        if board[i, j] != EMPTY: continue
        a = count_open_len_after(board, i, j, OP, 3, 2)
        b = count_open_len_after(board, i, j, OP, 4, 3)
        if a >= 1 and b >= 1: return (i, j)
    return None

RULE_ORDER = [
    immediate_win_check,
    block_opponent_immediate_win,
    block_opponent_double_four,
    create_double_four,
    block_extend_opponent_open_three,
    extend_my_open_three,
    block_opponent_double_three,
    create_double_three,
]

# ---------------- MCTS (얕은 트리 + 가변 rollouts) ----------------
def ucb1(wins: float, visits: int, parent_visits: int, c: float = 1.4) -> float:
    if visits == 0: return float('inf')
    return wins / visits + c * math.sqrt(max(1e-9, math.log(parent_visits) / visits))

def create_children(board: np.ndarray, who: int) -> List[Tuple[np.ndarray, Tuple[int,int]]]:
    cand = get_candidate_positions(board, 1)
    if not cand:
        cand = get_candidate_positions(board, 2)
    children = []
    for (i, j) in cand:
        nb = board.copy()
        nb[i, j] = who
        children.append((nb, (i, j)))
    return children

def random_playout(board: np.ndarray, turn: int) -> int:
    """인접 후보 위주 랜덤 시뮬. 마지막 수 승리만 빠르게 체크."""
    last = None
    empties = int(np.sum(board == EMPTY))
    while True:
        if last is not None:
            li, lj = last
            if is_win_last(board, -turn, li, lj):
                return -turn
        if empties == 0:
            return 0
        cand = get_candidate_positions(board, 1)
        if not cand:
            return 0
        i, j = r.choice(cand)
        board[i, j] = turn
        last = (i, j)
        empties -= 1
        turn = -turn

def smart_rollouts(board: np.ndarray) -> int:
    """후보 수에 따라 샘플 수 가변화."""
    c = len(get_candidate_positions(board, 1))
    if c >= 25:  return 350   # 초반
    if c >= 12:  return 600   # 중반
    return 900                 # 후반

def mcts_decide(board: np.ndarray, rollouts: int = 600, max_depth: int = 3) -> Optional[Tuple[int,int]]:
    visits: Dict[Tuple[bytes,int,int], int] = {}
    wins: Dict[Tuple[bytes,int,int], float] = {}
    parent_visits: Dict[Tuple[bytes,int], int] = {}

    def pkey(b: np.ndarray) -> bytes:
        return b.tobytes()

    root_children = create_children(board, AI)
    if not root_children:
        cand = get_candidate_positions(board, 1) or get_candidate_positions(board, 2)
        return r.choice(cand) if cand else None

    root_key = pkey(board)

    for _ in range(rollouts):
        path = []
        depth = 0
        cur = board.copy()
        pk = root_key
        turn = AI
        parent_visits[(pk, depth)] = parent_visits.get((pk, depth), 0) + 1

        while True:
            if depth >= max_depth:
                break
            children = create_children(cur, turn)
            if not children: break
            pN = parent_visits[(pk, depth)]
            # UCB1
            best_idx, best_score = None, -1e9
            for idx, (nb, _mv) in enumerate(children):
                key = (pk, depth, idx)
                if key not in visits:
                    visits[key] = 0; wins[key] = 0.0
                w = wins[key]; n = visits[key]
                score = ucb1(w, n, pN)
                if score > best_score:
                    best_score, best_idx = score, idx
            visits[(pk, depth, best_idx)] += 1
            path.append((pk, depth, best_idx, children))
            cur, _picked = children[best_idx]
            pk = pkey(cur)
            depth += 1
            parent_visits[(pk, depth)] = parent_visits.get((pk, depth), 0) + 1
            turn = -turn

        res = random_playout(cur.copy(), turn)
        reward = 1.0 if res == AI else (0.0 if res == 0 else -1.0)
        for (ppk, dd, idx, _children) in path:
            wins[(ppk, dd, idx)] = wins.get((ppk, dd, idx), 0.0) + reward

    # 루트에서 최고 승률 수
    best_mv, best_val = None, -1e9
    ch0 = create_children(board, AI)
    for idx, (_nb, mv) in enumerate(ch0):
        key = (root_key, 0, idx)
        v = visits.get(key, 0); w = wins.get(key, 0.0)
        val = (w / v) if v else -1e9
        if val > best_val:
            best_val, best_mv = val, mv
    return best_mv

def MCTS_engine(board: np.ndarray) -> Tuple[int,int]:
    # 1) 룰 레이어
    for fn in RULE_ORDER:
        mv = fn(board)
        if mv: return mv
    # 2) 첫 수 중앙
    if not np.any(board != EMPTY):
        H, W = board_size(board)
        return (H // 2, W // 2)
    # 3) MCTS (가변 rollouts / 얕은 트리)
    mv = mcts_decide(board, rollouts=smart_rollouts(board), max_depth=3)
    if mv: return mv
    # 4) 폴백
    cand = get_candidate_positions(board, 1) or get_candidate_positions(board, 2)
    return r.choice(cand) if cand else (board.shape[0]//2, board.shape[1]//2)

# ---------------- Matplotlib Click GUI (스레드 + 타이머) ----------------
def run_gui(size: int = 15, human_color: int = OP):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import threading

    board = np.zeros((size, size), dtype=int)
    ai_color = AI if human_color == OP else OP
    turn = AI if human_color == OP else OP  # 사람=OP면 AI 선공

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xticks(range(size)); ax.set_yticks(range(size))
    ax.set_xlim(-0.5, size-0.5); ax.set_ylim(-0.5, size-0.5)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.grid(True)
    default_title = "오목 — 보드를 클릭해 착수 (창 닫으면 종료)"
    ax.set_title(default_title)

    # 증분 그리기
    stones: Dict[Tuple[int,int], object] = {}
    last_circle = None
    ai_thinking = False
    ai_result = {"move": None}

    def draw_stone(i, j, who):
        if (i, j) in stones: return
        ch = '●'  # 사람/AI 모두 꽉 찬 원 문자
        color = 'black' if who == AI else 'blue'  # AI=검정, 사람=파랑
        t = ax.text(j, i, ch, ha='center', va='center', fontsize=18, color=color)
        stones[(i, j)] = t

    def mark_last(i, j):
        nonlocal last_circle
        if last_circle is not None:
            try: last_circle.remove()
            except: pass
        last_circle = Circle((j, i), 0.40, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(last_circle)
        fig.canvas.draw_idle()

    def end_if_finished_full() -> bool:
        # 최종 확인은 전체 스캔으로 안정성 확보
        if is_win_full(board, AI):
            ax.set_title("AI 승리! (창 닫기)"); fig.canvas.draw_idle(); return True
        if is_win_full(board, OP):
            ax.set_title("사람 승리! (창 닫기)"); fig.canvas.draw_idle(); return True
        if not np.any(board == EMPTY):
            ax.set_title("무승부! (창 닫기)"); fig.canvas.draw_idle(); return True
        return False

    # ---- AI 비동기 계산 ----
    def start_ai_search():
        nonlocal ai_thinking
        if ai_thinking: return
        ai_thinking = True
        ai_result["move"] = None
        ax.set_title("AI 계산 중…"); fig.canvas.draw_idle()

        def worker():
            mv = MCTS_engine(board.copy())  # 상태 복사
            ai_result["move"] = mv
        threading.Thread(target=worker, daemon=True).start()

        timer = fig.canvas.new_timer(interval=50)
        def poll():
            nonlocal ai_thinking, turn
            mv = ai_result["move"]
            if mv is None:
                timer.start(); return
            i, j = mv
            board[i, j] = ai_color
            draw_stone(i, j, ai_color)
            mark_last(i, j)
            ax.set_title(default_title); fig.canvas.draw_idle()
            ai_thinking = False
            if end_if_finished_full(): return
            turn = -turn
        timer.add_callback(poll)
        timer.start()
    # -----------------------

    def on_click(event):
        nonlocal turn
        if event.inaxes != ax: return
        if ai_thinking: return
        # 종료 체크
        if end_if_finished_full(): return
        if turn != human_color: return

        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        # 스냅: 가장 가까운 격자점
        j = int(round(x)); i = int(round(y))
        j = max(0, min(size - 1, j)); i = max(0, min(size - 1, i))
        if board[i, j] != EMPTY: return

        # 사람 즉시 표시
        board[i, j] = human_color
        draw_stone(i, j, human_color)
        mark_last(i, j)
        fig.canvas.draw_idle()
        # 빠른 승리 판정(마지막 수 기준)
        if is_win_last(board, human_color, i, j):
            ax.set_title("사람 승리! (창 닫기)"); fig.canvas.draw_idle(); return
        if not np.any(board == EMPTY):
            ax.set_title("무승부! (창 닫기)"); fig.canvas.draw_idle(); return

        # 턴 넘기고 AI 탐색 시작
        turn = -turn
        start_ai_search()

    fig.canvas.mpl_connect('button_press_event', on_click)

    # 선공이 AI면 시작하자마자 계산 시작
    if turn == (AI if human_color == OP else OP):
        start_ai_search()

    plt.show()

# ---------------- Entry ----------------
if __name__ == "__main__":
    # 기본: 사람이 파랑(OP=●), AI가 검정(AI=●) → AI 선공
    run_gui(size=15, human_color=OP)
    # 사람이 선공(검정) 원하면:
    # run_gui(size=15, human_color=AI)
