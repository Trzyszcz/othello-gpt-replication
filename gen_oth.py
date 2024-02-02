import random

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

    def encode_moves_from_tuples(self, moves_tuples):
        encoded_moves = []
        for move_tuple in moves_tuples:
            encoded_moves.append(self.enc_dict[move_tuple[0]] + str(move_tuple[1]))
        return encoded_moves

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

def how_many_correct_moves(board_size, actual_game, guessed_game):
    #function checks if i move in guessed game is correct in context of actual_game[:i]
    num_correct = 0
    game = oth(board_size)
    color = ['b', 'w']  #TODO add color and moves to game instance
    move = 0
    correct_moves = game.find_legal_moves(color[move%2])
    correct_moves = game.encode_moves_from_tuples(correct_moves)
    for guessed_move in guessed_game:
        if 'p' in correct_moves:
            if guessed_move == 'p':
                num_correct += 1
            #print("we should skip")
            continue
            
        if guessed_move in correct_moves:
            num_correct += 1
        #print(actual_game[0])
        [x, y] = actual_game[move]
        game.move(int(game.dec_dict[x]), int(y), color[move%2])
        move += 1
        correct_moves = game.find_legal_moves(color[move%2])
        correct_moves = game.encode_moves_from_tuples(correct_moves)
        if len(correct_moves) == 0:
            correct_moves = ['p']
    return num_correct

    
        
 


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
"""
color = ['b', 'w']

for i in range(100000):
    game = oth(4)
    move = 0
    moves = []
    while len(game.find_legal_moves(color[move%2])) != 0:
        moves.append(game.comp_move(color[move%2]))
        move += 1
    while len(moves) < 12:
        moves.append('p')
    for tok in moves:
        data_file.write(tok + ', ')
    data_file.write('\n')

data_file.close()


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
def pvp_game(board_size):
    game1 = oth(board_size)
    game1.print_board()
    turn = 0
    color = ['b', 'w']
    inp = input()
    while inp != 'q':
        #x, y = inp.split()
        [x, y] = inp
        if(game1.move(int(game1.dec_dict[x]), int(y), color[turn])):
            turn = (turn+1)%2
        game1.print_board()
        #print(game1.find_legal_moves(color[turn]))
        inp = input()

pvp_game(6)

