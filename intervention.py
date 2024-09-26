import torch
from torch import nn
import transformer_lens
import ast
import gen_oth
from train_nets import show_moves_from_tensor, read_enc_dec_dicts
from train_linear_probes import lin_prob
from visualise_board_from_activations import show_imaginary_board, load_probes
import os

#get model


oth_mod = torch.load('nets/99_good.pt')

enc_dict, dec_dict, encode, decode = read_enc_dec_dicts(6)

print(decode(encode(['A0', 'B0', 'p'])))

layer = 4

inside_probes_folder = os.listdir(f'./probes/lay{layer}')
probes = load_probes(6, layer, inside_probes_folder)


#probe = torch.load('probes/95_good_prob_1_1_lay7.pt')
probe = torch.load('probes/lay4/95_good_prob_1_2_lay4.pt')
#probe = torch.load('probes/lay4/95_good_prob_4_2_lay4.pt')
#probe = torch.load('probes/lay3/93_good_prob_1_2_lay3.pt')

#probe = torch.load('probes/lay4/94_good_prob_3_4_lay4.pt')


#probe = torch.load('probes/89_good_prob_1_2_lay1.pt')

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
#print(enemy_vector)

print('ACHTUNG')
print((enemy_vector.T@my_vector).item())
print('ACHTUNG')
print((enemy_vector.T@enemy_vector).item())
print((my_vector.T@my_vector).item())

#get game

game_to_inter = ['B2', 'B3', 'C4', 'B5', 'A4', 'A3', 'C5', 'D5', 'B4', 'A5', 'E4', 'F3', 'F5', 'A2', 'D4', 'B1', 'A0', 'C1', 'D1', 'C0', 'A1', 'F4', 'E5', 'D0', 'F2', 'B0', 'E1', 'E2', 'E3', 'F1', 'E0', 'F0']

move_to_change = 10

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

print("imaginary board before intervention:")
show_imaginary_board(activations[move_to_change], probes, turn) 

#print suggested moves

logits = oth_mod.forward(game_to_inter_enc)
#print(logits.shape)
probs = nn.functional.softmax(logits, dim=-1)
print('before intervention:')
show_moves_from_tensor(probs[0][move_to_change], prec=0.01)

#add hook

def ortho_proj(eq_vector, t_vector):
    #takes vector with parameters of equation describing subset of activation space which probe grades with value 1 (interpreted as a vector it is one orthagonal
    #to the n-1-dimentional space defined by that equation) and vector named t_vector. It return orthagonal projection of t_vector on space defined by equation.
    #in other words, it returns vector most close to t_vector which probe will grade with value 1

    non_zero_idx = len(eq_vector) - 1
    for i in range(len(eq_vector)):
        if eq_vector[i] != 0:
            non_zero_idx = i
            break
    non_zero_idx_val = eq_vector[non_zero_idx]
    transpo_vector = torch.zeros(len(eq_vector))
    transpo_vector[non_zero_idx] = -1/non_zero_idx_val
    transpo_vector = transpo_vector.to('cuda')
    #norm_of_eq_vec = eq_vector.T @ eq_vector

    transpo_t_vector = t_vector - transpo_vector

    dot_prod = eq_vector @ transpo_t_vector.T

    #print(eq_vector * (dot_prod/(eq_vector.T @ eq_vector)))

    projected_vector = t_vector - (10*(eq_vector * (dot_prod/(eq_vector.T @ eq_vector))))

    return projected_vector

def ortho_proj_new(eq_vector, t_vector, val):
    #takes vector with parameters of equation describing subset of activation space which probe grades with value 1 (interpreted as a vector it is one orthagonal
    #to the n-1-dimentional space defined by that equation) and vector named t_vector. It return orthagonal projection of t_vector on space defined by equation.
    #in other words, it returns vector most close to t_vector which probe will grade with value 1

    non_zero_idx = len(eq_vector) - 1
    for i in range(len(eq_vector)):
        if eq_vector[i] != 0:
            non_zero_idx = i
            break
    non_zero_idx_val = eq_vector[non_zero_idx]
    transpo_vector = torch.zeros(len(eq_vector))
    transpo_vector[non_zero_idx] = -val/non_zero_idx_val
    transpo_vector = transpo_vector.to('cuda')
    #norm_of_eq_vec = eq_vector.T @ eq_vector

    transpo_t_vector = t_vector - transpo_vector

    dot_prod = eq_vector @ transpo_t_vector.T

    #print(eq_vector * (dot_prod/(eq_vector.T @ eq_vector)))

    projected_vector = t_vector - (2*(eq_vector * (dot_prod/(eq_vector.T @ eq_vector))))

    return projected_vector


def ortho_proj_test():
    a = torch.tensor([-1.0, -1.0])
    t = torch.tensor([4.0, 1.0])
    print(ortho_proj(a, t))

    t2 = torch.tensor([-3, -2])
    print(ortho_proj(a, t2))

def intervention(add_vector, subt_vector, activation_vector):
    return activation_vector + ((-3)*subt_vector) + (3*add_vector)

def ortho_proj_hook(value, hook):
    print('hook activated')
    print(value.shape)
    activ = value[:, move_to_change][0]
    print('Activation size squared:')
    print((activ.T@activ).item())
    #print(value[:, 2])
    #value[:, move_to_change] = ortho_proj(enemy_vector, value[:, move_to_change])
    #value[:, move_to_change] = ortho_proj(enemy_vector, value[:, move_to_change], 5)
    #value[:, move_to_change] = ortho_proj(my_vector, value[:, move_to_change], -5)
    value[:, move_to_change] = intervention(enemy_vector, my_vector, value[:, move_to_change])
    print("imaginary board after intervention:")
    show_imaginary_board(value[:, move_to_change], probes, turn)
    #print(value[:, 2])
    return value

def print_my(value, hook):
    print( (value[:, move_to_change][0].T @ enemy_vector).item())
    return value
#activation_data_part = htransformer.run_with_cache(games_ten[i*100:(i+1)*100], names_filter=f'blocks.{lay_num}.mlp.hook_pre', device='cpu')[1][f'blocks.{lay_num}.mlp.hook_pre']
#run model forward
logits = oth_mod.run_with_hooks(
        game_to_inter_enc,
        return_type='logits',
        fwd_hooks=[(
            f'blocks.{i}.hook_resid_post',
            ortho_proj_hook
            #print_my
            ) for i in [4]]
        )
#print(logits.shape)
probs = nn.functional.softmax(logits, dim=-1)
print('after intervention:')
show_moves_from_tensor(probs[0][move_to_change], prec=0.01)


#print suggested moves
