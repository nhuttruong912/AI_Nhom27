from queue import PriorityQueue
#Tạo một hàng đợi ưu tiên để lưu trữ các nút
class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    # Định nghĩa lớp Node để lưu trữ trạng thái của bảng và các thông tin khác như cha của nó,
    # bước di chuyển để đến trạng thái hiện tại và độ sâu của nó trong cây tìm kiếm
    def __lt__(self, other):
        return (self.depth + self.heuristic()) < (other.depth + other.heuristic())

    # Định nghĩa phương thức lt để so sánh hai nút với nhau dựa trên tổng chi phí của chúng

    def __eq__(self, other):
        return self.state == other.state

    # Định nghĩa phương thức eq để so sánh hai nút với nhau dựa trên trạng thái của chúng

    def __hash__(self):
        return hash(str(self.state))
    # Định nghĩa phương thức hash để tính toán giá trị băm cho một nút.
    def heuristic(self):
        # Tính hàm heuristic cho trạng thái hiện tại
        # Hàm heuristic được tính bằng khoảng cách Manhattan giữa các ô vuông và vị trí đích của chúng
        h = 0
        for i in range(3):
            for j in range(3):
                if self.state[i][j] != 'x':
                    x, y = divmod(int(self.state[i][j]) - 1, 3)
                    h += abs(x - i) + abs(y - j)
        return h
    #Định nghĩa hàm heuristic để tính toán khoảng cách Manhattan giữa các ô vuông và vị trí đích của chúng.

    def get_successors(self):
        # Trả về danh sách các trạng thái kế tiếp có thể đạt được từ trạng thái hiện tại
        successors = []
        i, j = next((i, j) for i in range(3) for j in range(3) if self.state[i][j] == 'x')
        for ni, nj in ((i-1,j), (i+1,j), (i,j-1), (i,j+1)):
            if 0 <= ni < 3 and 0 <= nj < 3:
                new_state = [row[:] for row in self.state]
                new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                successors.append(Node(new_state, parent=self))
        return successors
    #Định nghĩa hàm get_successors để tìm kiếm các trạng thái kế tiếp có thể đạt được từ trạng thái hiện tại.
    def get_path(self):
        # Trả về đường đi từ trạng thái ban đầu đến trạng thái hiện tại
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return reversed(path)
    # Định nghĩa hàm get_path để tìm kiếm đường đi từ trạng thái ban đầu đến trạng thái hiện tại
def solve_puzzle(start_state, goal_state):
    start_node = Node(start_state)
    goal_node = Node(goal_state)

    frontier = PriorityQueue()
    frontier.put(start_node)

    explored = set()

    while not frontier.empty():
        node = frontier.get()

        if node == goal_node:
            return node.get_path()

        explored.add(node)

        for child_node in node.get_successors():
            if child_node not in explored:
                frontier.put(child_node)

    return None
    #Định nghĩa hàm solve_puzzle để giải quyết bài toán trò chơi 8 ô vuông bằng giải thuật A*.
start_state = [
    ['2', '8', '3'],
    ['1', '6', '4'],
    ['7', 'x', '5']
]

goal_state = [
    ['1', '2', '3'],
    ['8', 'x', '4'],
    ['7', '6', '5']
]

path = solve_puzzle(start_state, goal_state)
# Khởi tạo trạng thái ban đầu và trạng thái đích cho bài toán và gọi hàm solve_puzzle để giải quyết bài toán.
if path is None:
    print("Không tìm được đường đi từ trạng thái ban đầu đến trạng thái đích.")
else:
    print("Đường đi ngắn nhất từ trạng thái ban đầu đến trạng thái đích:")
    for node in path:
        print(node.move)
