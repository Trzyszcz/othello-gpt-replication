import random
from copy import deepcopy
import ast
from datetime import datetime

class oth:
    def __init__(self, board_size):
        self.bs = board_size
        self.board = [['x' for _ in range(self.bs)] for _ in range(self.bs)]
        self.board[(self.bs//2) - 1][(self.bs//2) - 1] = 'w'
        self.board[(self.bs//2) - 1][self.bs//2] = 'b'
        self.board[self.bs//2][(self.bs//2) - 1] = 'b'
        self.board[self.bs//2][self.bs//2] = 'w'

        #self.board[5][0] = 'b'
        #self.board[4][0] = 'w'

    enc_dict = {num:ch for num,ch in enumerate('ABCDEFGH')}
    dec_dict = {ch:num for num,ch in enumerate('ABCDEFGH')}

    def col_print(self, symbol):
        if symbol == 'x':
            print("\033[92m{}\033[00m".format(symbol), end=' ')
        if symbol == 'w':
            print(symbol, end=' ')
        if symbol == 'b':
            print("\033[30m{}\033[00m".format(symbol), end=' ')
           
    def print_board(self):
        for i in range(self.bs):
            print(self.enc_dict[i], end=' ')
            for j in range(self.bs):
                #print(self.board[i][j], end=' ')
                self.col_print(self.board[i][j])
            print('\n', end='')
        print('  ', end='')
        for i in range(self.bs):
            print(i, end=' ')
        print('\n', end='')

    def find_legal_moves(self, col):
        legal_moves_list = []
        for i in range(self.bs):
            for j in range(self.bs):
                if self.board[i][j] == 'x':
                    if self.move(i, j, col, True):
                        legal_moves_list.append((i, j))
        return legal_moves_list

    def comp_move(self, col):
        legal_moves = self.find_legal_moves(col)
        chosen_move = random.choice(legal_moves)
        self.move(chosen_move[0], chosen_move[1], col)
        return self.enc_dict[chosen_move[0]] + str(chosen_move[1])

    def move(self, x, y, col, check=False):
        #check if the move is legal
        #change state of the board
        if self.board[x][y] != 'x':
             #print('Illegal move, try again')
             return False
        if col == 'w':
            other_col = 'b'
        else:
            other_col = 'w'
        res = False
        res = self.vertical_up(x, y, col, other_col, check) or res
        res = self.vertical_down(x, y, col, other_col, check) or res
        res = self.horizontal_right(x, y, col, other_col, check) or res
        res = self.horizontal_left(x, y, col, other_col, check) or res
        res = self.diagonal_slash_up(x, y, col, other_col, check) or res
        res = self.diagonal_slash_down(x, y, col, other_col, check) or res
        res = self.diagonal_backslash_up(x, y, col, other_col, check) or res
        res = self.diagonal_backslash_down(x, y, col, other_col, check) or res
        if res:
            if not check:
                self.board[x][y] = col
            return True
        else:
            #if not check:
                #print('Illegal move, try again')
            return False

    def vertical_up(self, x, y, col, other_col, check):
        #print('vu')
        if x == 0 or x  == 1:
            return False
        #print('coord ok')
        to_change = [] #coordinates to flip if move is legal
        for i in range(1, x + 1):
            #print(f'checking {x+i}, {y}')
            if self.board[x - i][y] == other_col:
                to_change.append((x - i, y))
            if self.board[x - i][y] == col:
                if not check:
                    for piece in to_change:
                        self.board[piece[0]][piece[1]] = col
                if len(to_change) == 0:
                    return False
                else:
                    return True
            if self.board[x - i][y] == 'x':
                return False
        return False

    def vertical_down(self, x, y, col, other_col, check):
        #print('vd')
        if x == self.bs - 2 or x  == self.bs - 1:
            return False
        #print('coord ok')
        to_change = [] #coordinates to flip if move is leagal
        for i in range(1, (self.bs - 1) - x + 1):
            #print(f'checking {x+i}, {y}')
            if self.board[x + i][y] == other_col:
                to_change.append((x + i, y))
            if self.board[x + i][y] == col:
                if not check:
                    for piece in to_change:
                        self.board[piece[0]][piece[1]] = col
                if len(to_change) == 0:
                    return False
                else:
                    return True
            if self.board[x + i][y] == 'x':
                return False
        return False

    def horizontal_right(self, x, y, col, other_col, check):
        #print('hr')
        if y == self.bs - 2 or y  == self.bs - 1:
            return False
        #print('coord ok')
        to_change = [] #coordinates to flip if move is leagal
        for i in range(1, (self.bs - 1) - y + 1):
            #print(f'checking {x+i}, {y}')
            if self.board[x][y + i] == other_col:
                to_change.append((x, y + i))
            if self.board[x][y + i] == col:
                if not check:
                    for piece in to_change:
                        self.board[piece[0]][piece[1]] = col
                if len(to_change) == 0:
                    return False
                else:
                    return True
            if self.board[x][y + i] == 'x':
                return False
        return False

    def horizontal_left(self, x, y, col, other_col, check): 
        #if not check:
        #    print('hl')
        if y == 0 or y  == 1:
            #if not check:
            #    print('y=0 or 1')
            return False
        #print('coord ok')
        to_change = [] #coordinates to flip if move is leagal
        for i in range(1, y + 1):
            #print(f'checking {x+i}, {y}')
            if self.board[x][y - i] == other_col:
                to_change.append((x, y - i))
            if self.board[x][y - i] == col:
                if not check:
                    for piece in to_change:
                        self.board[piece[0]][piece[1]] = col
                if len(to_change) == 0:
                    #if not check:
                    #    print('nothing to change')
                    return False
                else:
                    return True
            if self.board[x][y - i] == 'x':
                #if not check:
                #    print('found x')
                return False
        return False

    def diagonal_slash_up(self, x, y, col, other_col, check):
        if x == 0 or x == 1 or y == self.bs - 2 or y  == self.bs - 1:
            return False
        #print('coord ok')
        to_change = [] #coordinates to flip if move is leagal
        for i in range(1, min((self.bs - 1) - y + 1, x + 1)):
            #print(f'checking {x+i}, {y}')
            if self.board[x - i][y + i] == other_col:
                to_change.append((x - i, y + i))
            if self.board[x - i][y + i] == col:
                if  not check:
                    for piece in to_change:
                        self.board[piece[0]][piece[1]] = col
                if len(to_change) == 0:
                    return False
                else:
                    return True
            if self.board[x - i][y + i] == 'x':
                return False
        return False

    def diagonal_slash_down(self, x, y, col, other_col, check):
        if x == self.bs - 2 or x == self.bs - 1 or y == 0 or y  == 1:
            return False
        #print('coord ok')
        to_change = [] #coordinates to flip if move is leagal
        for i in range(1, min((self.bs - 1) - x + 1, y + 1)):
            #print(f'checking {x+i}, {y}')
            if self.board[x + i][y - i] == other_col:
                to_change.append((x + i, y - i))
            if self.board[x + i][y - i] == col:
                if not check:
                    for piece in to_change:
                        self.board[piece[0]][piece[1]] = col
                if len(to_change) == 0:
                    return False
                else:
                    return True
            if self.board[x + i][y - i] == 'x':
                return False
        return False

    def diagonal_backslash_up(self, x, y, col, other_col, check):
        if x == 0 or x == 1 or y == 0 or y  == 1:
            return False
        #print('coord ok')
        to_change = [] #coordinates to flip if move is leagal
        for i in range(1, min(x + 1, y + 1)):
            #print(f'checking {x+i}, {y}')
            if self.board[x - i][y - i] == other_col:
                to_change.append((x - i, y - i))
            if self.board[x - i][y - i] == col:
                if  not check:
                    for piece in to_change:
                        self.board[piece[0]][piece[1]] = col
                if len(to_change) == 0:
                    return False
                else:
                    return True
            if self.board[x - i][y - i] == 'x':
                return False
        return False

    def diagonal_backslash_down(self, x, y, col, other_col, check):
        if x == self.bs - 2 or x == self.bs - 1 or y == self.bs - 2 or y  == self.bs - 1:
            return False
        #print('coord ok')
        to_change = [] #coordinates to flip if move is leagal
        for i in range(1, min((self.bs - 1) - y + 1, (self.bs - 1) - x + 1)):
            #print(f'checking {x+i}, {y}')
            if self.board[x + i][y + i] == other_col:
                to_change.append((x + i, y + i))
            if self.board[x + i][y + i] == col:
                if not check:
                    for piece in to_change:
                        self.board[piece[0]][piece[1]] = col
                if len(to_change) == 0:
                    return False
                else:
                    return True
            if self.board[x + i][y + i] == 'x':
                return False
        return False

def check_game(board_size, notated_game):
    color = ['b', 'w']
    game = oth(board_size)
    for turn, move in enumerate(notated_game):
        if len(game.find_legal_moves(color[turn%2]))==0:
            for last_move in notated_game[turn:]:
                if last_move != 'p':
                    return False
            return True
        else:
            if move == 'p' or move == 's':
                return False
            
        row = move[0]
        column = move[1]
        if not game.move(int(game.dec_dict[row]), int(column), color[turn%2]):
            #print(move)
            return False
    return True

def gen_board_state_data_set(board_size):
    color = ['b', 'w']
    game = oth(board_size)
    
    cells_state = {game.enc_dict[row]+str(column):[] for row in range(board_size) for column in range(board_size)}
    print(cells_state)

#gen_board_state_data_set(6)
"""
data_file = open('data.txt', 'r')
first_games = []
for i in range(5):
    first_games.append(data_file.readline()) #here was a bug! first_games[i]
    first_games[i] = first_games[i].split(", ")[:-1]
data_file.close()

for game_to_check in first_games:
    print(check_game(4, game_to_check))

print('GEN GAME')
gen_game = ['C3', 'D3', 'B0', 'B3', 'D2', 'C0', 'D0', 'A1', 'A0', 'D1', 'A3']
print(check_game(4, gen_game))
"""
def generate_game(board_size, whole_board_state=False):
    color = ['b', 'w']
    game = oth(board_size)
    move = 0
    moves = []
    if whole_board_state:
        stack_board_states = []
        stack_board_states.append(deepcopy(game.board))
    while len(game.find_legal_moves(color[move%2])) != 0:
        moves.append(game.comp_move(color[move%2]))
        move += 1
        if whole_board_state:
            stack_board_states.append(deepcopy(game.board))
    while len(moves) < ((board_size**2) - 4):
        moves.append('p')
    if not whole_board_state:
        return moves
    else:
        return [moves, stack_board_states]


def generate_games(board_size, number_of_games, whole_board_state=False):
    data_file = open('data6.txt', 'w')
    for i in range(number_of_games):
        moves = generate_game(board_size, whole_board_state)
        for tok in moves:
            data_file.write(tok + ', ')
        data_file.write('\n')
    data_file.close()

def generate_games_boards(board_size, number_of_games, whole_board_state=True):
    data_bw_file = open('data_boards_bw_100k.txt', 'w')
    data_me_file = open('data_boards_me_100k.txt', 'w')
    data_bw = []
    data_me = []
    for i in range(number_of_games):
        moves = generate_game(board_size, whole_board_state)
        data_bw.append(moves)
        data_me.append([moves[0], change_wb_to_my_enemys(moves[1])])
        if i % 1000 == 0:
            print('.', end='')
    data_bw_file.write(data_bw.__str__())
    data_me_file.write(data_me.__str__())
    data_bw_file.close()
    data_me_file.close()
    print('\n')



#on black on even moves, white on odd ones
def change_wb_to_my_enemys(stack_board_states):
    changed_stack_board_states = []
    for move_idx in range(len(stack_board_states)):
        changed_board_state = []
        for row in stack_board_states[move_idx]:
            changed_row = []
            for cell in row:
                changed_cell = 'x'
                if move_idx%2==0:
                    if cell=='b':
                        changed_cell = 'm'
                    if cell=='w':
                        changed_cell = 'e'
                if move_idx%2==1:
                    if cell=='b':
                        changed_cell = 'e'
                    if cell=='w':
                        changed_cell = 'm'
                changed_row.append(changed_cell)
            changed_board_state.append(changed_row)
        changed_stack_board_states.append(changed_board_state)
    return changed_stack_board_states

def generate_data_for_act_learning():
    now = datetime.now()
    current_time = now.strftime('%H:%M:%S')
    print(f'Start generating {current_time}')
    generate_games_boards(6, 100000)
    now = datetime.now()
    current_time = now.strftime('%H:%M:%S')
    print(f'Finished {current_time}')

"""
sbs = generate_game(4, True)
for i in range(len(sbs)):
    for j in range(len(sbs[i])):
        print(sbs[i][j])
    print('\n')

changed_sbs = change_wb_to_my_enemys(sbs)

for i in range(len(changed_sbs)):
    for j in range(len(changed_sbs[i])):
        print(changed_sbs[i][j])
    print('\n')

print(sbs[0:-1, 0, 0])
"""
"""
game1 = oth(4)
game1.print_board()
turn = 0
color = ['b', 'w']
while True:
    if turn == 0:
        #print(game1.find_legal_moves(color[turn]))
        inp = input()
        if inp == 'q':
            break
        x, y = inp.split()
        while not game1.move(int(game1.dec_dict[x]), int(y), color[turn]):
            inp = input()
            if inp == 'q':
                break
            x, y = inp.split()
    else:
        game1.comp_move(color[turn])
    game1.print_board()
    if len(game1.find_legal_moves(color[(turn+1)%2])) == 0:
        print(f"{color[turn]} wins")
        break
    turn = (turn+1)%2
    #print(game1.find_legal_moves(color[turn]))
"""
"""
while inp != 'q':
    x, y = inp.split()
    if(game1.move(int(x), int(y), color[turn])):
        turn = (turn+1)%2
    game1.print_board()
    print(game1.find_legal_moves(color[turn]))
    inp = input()
"""
