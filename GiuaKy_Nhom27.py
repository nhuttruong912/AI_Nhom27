#!/usr/bin/env python3
import argparse


class Node:

    # ----- Khởi tạo nút -----#
    def __init__(self, pattern, gfunc, move='start'):
        self.pattern = pattern
        self.gfunc = gfunc
        self.move = move
        for (row, i) in zip(pattern, range(3)):
            if 0 in row:
                self.blankloc = [i, row.index(0)] #lấy vị trí phần tử 0
            # print(self.blankloc[0],self.blankloc[1])

    # ----- kiểm tra xem các trạng thái có bằng nhau hay không----- #
    def __eq__(self, other):
        if other == None:
            return False

        if isinstance(other, Node) != True:
            raise TypeError

        for i in range(3):
            for j in range(3):
                if self.pattern[i][j] != other.pattern[i][j]:
                    return False
        return True

    # -----  để lấy một phần tử đầu tàu giống như một mảng -----#
    def __getitem__(self, key):
        if isinstance(key, tuple) != True:
            raise TypeError
        if len(key) != 2:
            raise KeyError

        return self.pattern[key[0]][key[1]]

    # tính toán hàm heuristic cho trạng thái hiện tại của trò chơi bằng cách đếm số lượng ô không đúng vị trí so với trạng thái mục tiêu.
    # Hàm heuristic được sử dụng để ước tính chi phí để đạt được trạng thái mục tiêu. -----#

    def calc_hfunc(self, goal):
        self.hfunc = 0
        for i in range(3):
            for j in range(3):
                # print (i,j)
                if self.pattern[i][j] != goal.pattern[i][j]:
                    self.hfunc += 1
        if self.blankloc != goal.blankloc:
            self.hfunc -= 1

        self.ffunc = self.hfunc + self.gfunc

        return self.hfunc, self.gfunc, self.ffunc

    # ----- Hàm di chuyển ô trống sang trái nếu có thể -----#
    def moveleft(self):
        if self.blankloc[1] == 0:
            return None

        left = [[self.pattern[i][j] for j in range(3)] for i in range(3)]
        left[self.blankloc[0]][self.blankloc[1]] = left[self.blankloc[0]][self.blankloc[1] - 1]
        left[self.blankloc[0]][self.blankloc[1] - 1] = 0

        return Node(left, self.gfunc + 1, 'left')

    # ----- Hàm di chuyển ô trống sang phải nếu có thể -----#
    def moveright(self):
        if self.blankloc[1] == 2:
            return None

        right = [[self.pattern[i][j] for j in range(3)] for i in range(3)]
        right[self.blankloc[0]][self.blankloc[1]] = right[self.blankloc[0]][self.blankloc[1] + 1]
        right[self.blankloc[0]][self.blankloc[1] + 1] = 0

        return Node(right, self.gfunc + 1, 'right')

    # ----- Hàm di chuyển ô trống lên nếu có thể -----#
    def moveup(self):
        if self.blankloc[0] == 0:
            return None

        up = [[self.pattern[i][j] for j in range(3)] for i in range(3)]
        up[self.blankloc[0]][self.blankloc[1]] = up[self.blankloc[0] - 1][self.blankloc[1]]
        up[self.blankloc[0] - 1][self.blankloc[1]] = 0

        return Node(up, self.gfunc + 1, 'up')

    # ----- Hàm di chuyển ô trống xuống dưới nếu có thể -----#
    def movedown(self):
        if self.blankloc[0] == 2:
            return None

        down = [[self.pattern[i][j] for j in range(3)] for i in range(3)]
        down[self.blankloc[0]][self.blankloc[1]] = down[self.blankloc[0] + 1][self.blankloc[1]]
        down[self.blankloc[0] + 1][self.blankloc[1]] = 0

        return Node(down, self.gfunc + 1, 'down')

    # ----- tạo ra các trạng thái mới bằng cách di chuyển ô trống sang trái, phải, lên hoặc xuống.
    # Sau đó, nó đóng nút hiện tại và mở các nút mới được tạo ra. - ----#
    def moveall(self, game):
        left = self.moveleft()
        left = None if game.isclosed(left) else left
        right = self.moveright()
        right = None if game.isclosed(right) else right
        up = self.moveup()
        up = None if game.isclosed(up) else up
        down = self.movedown()
        down = None if game.isclosed(down) else down

        game.closeNode(self)
        game.openNode(left)
        game.openNode(right)
        game.openNode(up)
        game.openNode(down)

        return left, right, up, down

    # ----- Hàm in mảng ra định dạng đẹp -----#
    def print(self):
        print(self.move + str(self.gfunc))
        print(self.pattern[0])
        print(self.pattern[1])
        print(self.pattern[2])


class Game:

    #khởi tạo một đối tượng trò chơi với trạng thái ban đầu và trạng thái mục tiêu. Nó cũng khởi tạo các biến open và closed để theo dõi các nút đã mở và đóng.
    # Cuối cùng, nó tính toán hàm heuristic cho trạng thái ban đầu và thêm nó vào danh sách các nút mở.#
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.open = {}
        self.closed = {}
        _, _, ffunc = self.start.calc_hfunc(self.goal)
        self.open[ffunc] = [start]

    # kiểm tra xem một nút đã được đóng hay chưa bằng cách tính toán hàm heuristic cho nút đó và kiểm tra xem giá trị hàm heuristic đó có trong danh sách bảng băm đó không.
    # Nếu có, nó sẽ kiểm tra xem nút đã được đóng có trong danh sách các nút đã đóng với giá trị hàm heuristic tương ứng hay không.#
    def isclosed(self, node):
        if node == None:  # trả về True nếu không có nút nào
            return True

        hfunc, _, _ = node.calc_hfunc(self.goal)  # tính toán hfucntion để kiểm tra danh sách bảng băm đó

        if hfunc in self.closed:
            for x in self.closed[hfunc]:
                if x == node:
                    return True

        return False

    #  xóa nút khỏi danh sách các nút mở và thêm nó vào danh sách các nút đã đóng.
    #  Nếu danh sách các nút mở trống, nó sẽ xóa thuộc tính của hàm heuristic đó khỏi bảng băm.#
    def closeNode(self, node):
        if node == None:
            return

        hfunc, _, ffunc = node.calc_hfunc(self.goal)
        self.open[ffunc].remove(node)  # xóa khỏi danh sách ffunc của bảng băm cho các nút mở
        if len(self.open[ffunc]) == 0:
            del self.open[ffunc]  # xóa thuộc tính của hàm nếu danh sách của nó trống

        if hfunc in self.closed:
            self.closed[hfunc].append(node)
        else:
            self.closed[hfunc] = [node]

        return

    # thêm nút vào danh sách các nút mở.
    # Nếu danh sách các nút mở không có thuộc tính ffunc tương ứng, nó sẽ tạo một thuộc tính mới với giá trị ffunc đó.#
    def openNode(self, node):
        if node == None:
            return

        _, _, ffunc = node.calc_hfunc(
            self.goal)  # Tính ffunnc để thêm nút vào danh sách kết quả đó trong bảng băm
        if ffunc in self.open:
            self.open[ffunc].append(node)
        else:
            self.open[ffunc] = [node]

        return

    # ---- Hàm giải game bằng thuật toán A star ----#
    def solve(self):

        presentNode = None

        while (presentNode != self.goal):
            i = 0
            while i not in self.open:
                i += 1  # Kiểm tra danh sách có ít 'ffunction' nhất để chọn một nút từ danh sách đó
            presentNode = self.open[i][-1]
            presentNode.moveall(self)  # Mở rộng nút đó cho các bước di chuyển có thể tiếp theo

        # ---- In giải pháp theo hướng ngược lại, tức là từ mục tiêu đến đầu ----#
        while presentNode.move != 'start':
            presentNode.print()
            # di chuyển ngược lại những gì đã được thực hiện để đạt đến trạng thái quay lại theo giải pháp
            if presentNode.move == 'up':
                presentNode = presentNode.movedown()
            elif presentNode.move == 'down':
                presentNode = presentNode.moveup()
            elif presentNode.move == 'right':
                presentNode = presentNode.moveleft()
            elif presentNode.move == 'left':
                presentNode = presentNode.moveright()

            hfunc, _, _ = presentNode.calc_hfunc(self.goal)
            for i in self.closed[hfunc]:
                if i == presentNode:
                    presentNode = i

        return


if __name__ == '__main__':
    # ----- Phân tích đối số -----#
    parser = argparse.ArgumentParser()
    # parser.add_argument("--hfunc",help='choose 1 for Manhattan distance and 2 for Displaced tiles.',metavar='Heuristic Function', default='1')
    parser.add_argument("--startrow",
                        help='Enter the numbers in sequence for starting arangement starting from row 1 to row 3 space separated (put 0 for blank area).',
                        type=int, nargs=9, metavar=(
        'row1col1', 'row1col2', 'row1col3', 'row2col1', 'row2col2', 'row2col3', 'row3col1', 'row3col2', 'row3col3'),
                        required=True)
    parser.add_argument("--goalrow",
                        help='Enter the numbers in sequence for goal arangement starting from row 1 to row 3 space sepearted (put 0 for blank area).',
                        type=int, nargs=9, metavar=(
        'row1col1', 'row1col2', 'row1col3', 'row2col1', 'row2col2', 'row2col3', 'row3col1', 'row3col2', 'row3col3'),
                        required=True)

    args = parser.parse_args()

    x = [1, 2, 3, 4, 5, 6, 7, 8, 0]

    # ----- Xác nhận nếu Đầu vào đúng -----#

    assert set(x) == set(args.startrow)
    assert set(x) == set(args.goalrow)

    # ----- Định dạng lại Đầu vào -----#

    startloc = [args.startrow[0:3], args.startrow[3:6], args.startrow[6:]]
    goalloc = [args.goalrow[0:3], args.goalrow[3:6], args.goalrow[6:]]

    # ----- Khởi tạo nút bắt đầu và kết thúc -----#

    start = Node(startloc, 0)
    goal = Node(goalloc, 0, 'goal')

    # ----- Khởi tạo trò chơi -----#

    game = Game(start, goal)

    game.solve()  # Giải quyết trò chơi
