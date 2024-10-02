import torch
from torch import nn
import transformer_lens
import ast
import gen_oth
from train_nets import read_enc_dec_dicts
from train_linear_probes import lin_prob
from visualise_board_from_activations import show_imaginary_board, load_probes
import os

def likely_moves_from_tensor(tens, prec=0.0001):
    _, _, _, decode = read_enc_dec_dicts(6)

    moves_dict = {}

    for i in range(len(tens)):
        if tens[i].item() >= prec:
            moves_dict[decode([i])[0]] = tens[i].item()
    return moves_dict

def print_moves_from_dict(new_dict,  prec=0.0001):
    for key in new_dict:
        print('{} : {:.3f}'.format(key, new_dict[key]))

def print_moves_from_dict_with_dif(old_dict, new_dict,  prec=0.0001):
    for key in new_dict:
        if key in old_dict:
            print('{} : {:.3f}'.format(key, new_dict[key]))
        else:
            print('\033[92m{} : {:.3f}\033[00m'.format(key, new_dict[key]))


#get model

oth_mod = torch.load('nets/oth_net.pt')

enc_dict, dec_dict, encode, decode = read_enc_dec_dicts(6)

#print(decode(encode(['A0', 'B0', 'p'])))

layer = 4


#get game

#game_to_inter = ['B2', 'B3', 'C4', 'B5', 'A4', 'A3', 'C5', 'D5', 'B4', 'A5', 'E4', 'F3', 'F5', 'A2', 'D4', 'B1', 'A0', 'C1', 'D1', 'C0', 'A1', 'F4', 'E5', 'D0', 'F2', 'B0', 'E1', 'E2', 'E3', 'F1', 'E0', 'F0']

with open("./games/game_to_inter") as file:
    game_to_inter = file.readline().split(", ")

move_to_change = 23

game_to_inter_enc = torch.tensor(encode(['s'] + game_to_inter[:-1]))

activations = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.{layer}.hook_resid_post', device='cuda')[1][f'blocks.{layer}.hook_resid_post'][0]


#print board state after some move

game1 = gen_oth.oth(6)
turn = 0
color = ['b', 'w']
for move in range(move_to_change):
    x, y = game_to_inter[move]
    game1.move(int(game1.dec_dict[x]), int(y), color[turn])
    turn = (turn+1)%2
game1.print_board()

if turn == 0:
    print("Black to move")
else:
    print("White to move")

#get user input
row = input("Row: ")
column = int(input("Column: "))

row_letters = "ABCDEFGH"
if row in row_letters:
    row = row_letters.index(row)
row = int(row)

inter_type = input("Intervention type:\n1) enemy to mine\n2) mine to enemy\n")

intervention_layers = [int(inp_str) for inp_str in input("Intervention layers (separated by spaces): ").split()]
scaling_parameter = float(input("Scaling parameter: "))
print("")

#get probe

inside_probes_folder = os.listdir(f'./probes/lay{layer}')
probes = load_probes(6, layer, inside_probes_folder)


probe = probes[row][column]


#get vectors from probe

me_enc_dict = {'x':0, 'm':1, 'e':2}
me_dec_list = ['x', 'm', 'e']

#for par in probe.named_parameters():
#    print(par)

weights = [x[1] for x in probe.named_parameters() if x[0]=='innards.0.weight']
#bias = [x[1] for x in probe.named_parameters() if x[0]=='innards.0.bias']

#print(weights.shape)
#print(weights)

enemy_vector = weights[0][2]
empty_vector = weights[0][0]
my_vector = weights[0][1]

if inter_type == "1":
    vect_to_subt = enemy_vector
    vect_to_add = my_vector
else:
    vect_to_subt = my_vector
    vect_to_add = enemy_vector

print("Imaginary board before intervention:")
show_imaginary_board(activations[move_to_change], probes, turn) 
print("")
#print suggested moves

logits = oth_mod.forward(game_to_inter_enc)
#print(logits.shape)
probs = nn.functional.softmax(logits, dim=-1)
print('Legal moves before intervention:')
pre_inter_dict = likely_moves_from_tensor(probs[0][move_to_change], prec=0.01)
print_moves_from_dict(pre_inter_dict, prec=0.01)
print("")

#add hook

def intervention(add_vector, subt_vector, activation_vector):
    return activation_vector + (( (-1) * scaling_parameter)*subt_vector) + (scaling_parameter*add_vector)

def ortho_proj_hook(value, hook):
    #print('hook activated')
    #print(value.shape)
    activ = value[:, move_to_change][0]
    #print('Activation size squared:')
    #print((activ.T@activ).item())
    #print(value[:, 2])
    value[:, move_to_change] = intervention(vect_to_add, vect_to_subt, value[:, move_to_change])
    print("Imaginary board after intervention:")
    show_imaginary_board(value[:, move_to_change], probes, turn)
    print("")
    #print(value[:, 2])
    return value

def print_my(value, hook):
    print( (value[:, move_to_change][0].T @ enemy_vector).item())
    return value

logits = oth_mod.run_with_hooks(
        game_to_inter_enc,
        return_type='logits',
        fwd_hooks=[(
            f'blocks.{i}.hook_resid_post',
            ortho_proj_hook
            #print_my
            ) for i in intervention_layers]
        )
#print(logits.shape)
probs = nn.functional.softmax(logits, dim=-1)
print('Legal moves after intervention:')
post_inter_dict = likely_moves_from_tensor(probs[0][move_to_change], prec=0.01)
print_moves_from_dict_with_dif(pre_inter_dict, post_inter_dict, prec=0.01)


#print suggested moves
