# Nguyễn Phan Nhựt Trường N20DCCN082
# Bùi Tuấn Anh N20DCCN002
# Nguyễn Tấn Phát N20DCCN054



import numpy as np
from queue import PriorityQueue

def heuristic(a, b):
    return np.sum(a != b)

def astar(start, goal):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()[1]

        if current == goal:
            break

        for next in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put((priority, next))
                came_from[next] = current

    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def neighbors(current):
    x = int(np.where(current == 0)[0])
    y = int(np.where(current == 0)[1])
    candidates = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
    valid_candidates = []
    for r,c in candidates:
        if r >= 0 and r < 3 and c >= 0 and c < 3:
            valid_candidates.append((r,c))
    results = []
    for r,c in valid_candidates:
        temp_board = np.copy(current)
        temp_board[x][y], temp_board[r][c] = temp_board[r][c], temp_board[x][y]
        results.append(temp_board)
    return results

start_state = np.array([[2,8,3],[1,6,4],[7,0,5]])
goal_state = np.array([[1,2,3],[8,0,4],[7,6,5]])

came_from, cost_so_far = astar(start_state.tobytes(), goal_state.tobytes())
path = reconstruct_path(came_from, start_state.tobytes(), goal_state.tobytes())

for p in path:
    print(p.reshape(3,3))