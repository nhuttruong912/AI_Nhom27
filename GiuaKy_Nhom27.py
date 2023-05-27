# Nguyễn Phan Nhựt Trường N20DCCN082
# Bùi Tuấn Anh N20DCCN002
# Nguyễn Tấn Phát N20DCCN054



import numpy as np
from queue import PriorityQueue
#Hàm `heuristic` tính toán khoảng cách Manhattan giữa hai trạng thái của bài toán 8 ô chữ. 
#Khoảng cách Manhattan là tổng khoảng cách giữa mỗi ô và vị trí của nó trong trạng thái mục tiêu.
def heuristic(a, b):
    return np.sum(a != b)


#Hàm `astar` thực hiện thuật toán A* để tìm đường đi ngắn nhất từ trạng thái ban đầu đến trạng thái mục tiêu. 
#Hàm này sử dụng một hàng đợi ưu tiên để lưu trữ các trạng thái được xét đến và tính toán chi phí để đi từ trạng thái ban đầu đến
#các trạng thái khác.
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

#Hàm `reconstruct_path` tạo ra một danh sách các trạng thái để đi từ trạng thái ban đầu đến trạng thái mục tiêu.

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

#Hàm `neighbors` tìm tất cả các trạng thái kế tiếp có thể được tạo ra từ một trạng thái hiện tại bằng cách 
#di chuyển ô trống lên, xuống, sang trái hoặc sang phải.
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
