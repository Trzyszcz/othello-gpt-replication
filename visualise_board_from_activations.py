import torch
from torch import nn
import transformer_lens
import ast
import gen_oth
from train_linear_probes import lin_prob
import os
from copy import deepcopy
from train_nets import read_enc_dec_dicts

me_enc_dict = {'x':0, 'm':1, 'e':2}
me_dec_list = ['x', 'm', 'e']

def load_probes(board_dim, layer, probes_names_list):
    probes = []
    for i in range(board_dim):
        row_of_probes = []
        for j in range(board_dim):
            probe = find_probe_for_cell([i, j], layer, probes_names_list)
            row_of_probes.append(probe)
        probes.append(deepcopy(row_of_probes))
    return probes

def find_probe_for_cell(coord, layer, probes_names_list):
    for path in probes_names_list:
        if path.endswith(f'{coord[0]}_{coord[1]}_lay{layer}.pt'):
            probe = torch.load(os.path.join(f'probes/lay{layer}', path))
            ##this part is for testing bug
            #probe = lin_prob(1280)
            #if torch.cuda.is_available():
            #    probe.to('cuda')
            return probe

def col_print_me(symbol):
    if symbol == 'x':
        print("\033[92m{}\033[00m".format(symbol), end=' ')
    if symbol == 'm':
        print(symbol, end=' ')
    if symbol == 'e':
        print("\033[30m{}\033[00m".format(symbol), end=' ')

def col_print_bw(symbol, turn):
    if symbol == 'x':
        print("\033[92m{}\033[00m".format(symbol), end=' ')
    if symbol == 'm' and turn == 0:
        print("\033[30m{}\033[00m".format('b'), end=' ')
    if symbol == 'e' and turn == 0:
        print('w', end=' ')
    if symbol == 'm' and turn == 1:
        print('w', end=' ')
    if symbol == 'e' and turn == 1:
        print("\033[30m{}\033[00m".format('b'), end=' ')
 

def print_board(board, turn, bw=True):
    rows_names = 'ABCDEFGHIJKLMNOPRST'
    for i, row in enumerate(board):
        print(rows_names[i], end=' ')
        for symbol in row:
            #print(self.board[i][j], end=' ')
            if bw:
                col_print_bw(symbol, turn)
            else:
                col_print_me(symbol)
        print('\n', end='')
    print('  ', end='')
    for i in range(len(board)):
        print(i, end=' ')
    print('\n', end='')


def show_imaginary_board(activation, probes, turn):
    imaginary_board = []
    for i, probes_row in enumerate(probes):
        imaginary_board_row = []
        for j, probe in enumerate(probes_row):
            cell_state = me_dec_list[torch.argmax(probes[i][j](activation))]
            imaginary_board_row.append(cell_state)
        imaginary_board.append(deepcopy(imaginary_board_row))

    print_board(imaginary_board, turn)

if __name__ == '__main__':

    enc_dict, dec_dict, encode, decode = read_enc_dec_dicts(6)
    print(decode(encode(['A0', 'B0', 'p'])))

    #get game
    
    game_to_inter = ['B2', 'B3', 'C4', 'B5', 'A4', 'A3', 'C5', 'D5', 'B4', 'A5', 'E4', 'F3', 'F5', 'A2', 'D4', 'B1', 'A0', 'C1', 'D1', 'C0', 'A1', 'F4', 'E5', 'D0', 'F2', 'B0', 'E1', 'E2', 'E3', 'F1', 'E0', 'F0']
    
    #game_to_inter = ['E3', 'E4', 'B2', 'B1', 'B0', 'A0', 'C1', 'D1', 'E1', 'A3', 'C0', 'E2', 'B3', 'E0', 'F0', 'C4', 'F4', 'F3', 'C5', 'A1', 'F5', 'A4', 'A2', 'D0', 'F1', 'D4', 'B4', 'D5', 'F2', 'E5', 'A5', 'B5']
    
    game_to_inter_enc = torch.tensor(encode(['s'] + game_to_inter[:-1]))
    #print board state after some move
    #get probes
    
    lay = 4
    
    inside_probes_folder = os.listdir(f'./probes/lay{lay}')
    
    #HERE
    
    probes = load_probes(6, lay, inside_probes_folder)
     
    
    me_enc_dict = {'x':0, 'm':1, 'e':2}
    me_dec_list = ['x', 'm', 'e']
    
    #game_to_inter
    #game_to_inter_enc
    
    oth_mod = torch.load('nets/99_good.pt')
    
    game1 = gen_oth.oth(6)
    turn = 0
    color = ['b', 'w']
    
    #activations = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.{lay}.mlp.hook_pre', device='cuda')[1][f'blocks.{lay}.mlp.hook_pre'][0]
    activations = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.{lay}.hook_resid_post', device='cuda')[1][f'blocks.{lay}.hook_resid_post'][0]
    
    print(activations.shape)
    
    
    for i in range(len(game_to_inter)):
        print('True board:')
        game1.print_board()
        print('Imaginary board:')
        show_imaginary_board(activations[i], probes, turn)
        x, y = tuple(game_to_inter[i])
        game1.move(int(game1.dec_dict[x]), int(y), color[turn])
        turn = (turn+1)%2
    
    
    #show real board
    
    
    #show imaginary board
