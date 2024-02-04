import torch
from torch import nn
import transformer_lens
import ast
import gen_oth_act

#get model


oth_mod = torch.load('40_12_90_6.pt')

with open('../enc_dec_dicts/enc_dict6.txt', 'r') as f:
    enc_dict_data = f.read()

with open('../enc_dec_dicts/dec_dict6.txt', 'r') as f:
    dec_dict_data = f.read()

enc_dict = ast.literal_eval(enc_dict_data)
dec_dict = ast.literal_eval(dec_dict_data)

encode = lambda tok_lst: [enc_dict[tok] for tok in tok_lst]
decode = lambda num_lst: [dec_dict[num] for num in num_lst]

print(decode(encode(['A0', 'B0', 'p'])))

#get probe
class lin_prob(nn.Module):
    def __init__(self):
        super().__init__()
        self.innards = nn.Sequential(
                nn.Linear(360, 3)
                )
    def forward(self, x):
        return self.innards(x)


probe = torch.load('lay9/92_good_prob_1_1.pt')

#get vectors from probe

me_enc_dict = {'x':0, 'm':1, 'e':2}
#for par in probe.named_parameters():
#    print(par)

weights = [x[1] for x in probe.named_parameters() if x[0]=='innards.0.weight']
#bias = [x[1] for x in probe.named_parameters() if x[0]=='innards.0.bias']
#print(weights.shape)
print(weights)

enemy_vector = weights[0][2]

#get game

game_to_inter = ['B2', 'B3', 'C4', 'B5', 'A4', 'A3', 'C5', 'D5', 'B4', 'A5', 'E4', 'F3', 'F5', 'A2', 'D4', 'B1', 'A0', 'C1', 'D1', 'C0', 'A1', 'F4', 'E5', 'D0', 'F2', 'B0', 'E1', 'E2', 'E3', 'F1', 'E0', 'F0']

game_to_inter_enc = torch.tensor(encode(['s'] + game_to_inter[:-1]))

#print board state after some move

game1 = gen_oth_act.oth(6)
turn = 0
color = ['b', 'w']
x, y = 'B 2'.split()
game1.move(int(game1.dec_dict[x]), int(y), color[turn])
turn = (turn+1)%2
gx, y = 'B 3'.split()
game1.move(int(game1.dec_dict[x]), int(y), color[turn])
turn = (turn+1)%2
game1.print_board()

#print suggested moves

def show_moves_from_tensor(tens):
    for i in range(len(tens)):
        if tens[i].item() >= 0.0001:
            print('{} : {:.3f}'.format(decode([i])[0], tens[i].item()))


logits = oth_mod.forward(game_to_inter_enc)
#print(logits.shape)
probs = nn.functional.softmax(logits, dim=-1)
print('before intervention:')
show_moves_from_tensor(probs[0][2])

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

    projected_vector = t_vector - 20*(eq_vector * (dot_prod/(eq_vector.T @ eq_vector)))

    return projected_vector

def ortho_proj_test():
    a = torch.tensor([-1.0, -1.0])
    t = torch.tensor([4.0, 1.0])
    print(ortho_proj(a, t))

    t2 = torch.tensor([-3, -2])
    print(ortho_proj(a, t2))

def ortho_proj_hook(value, hook):
    print('hook activated')
    #print(value[:, 2])
    value[:, 2] = ortho_proj(enemy_vector, value[:, 2])
    #print(value[:, 2])
    return value
#activation_data_part = htransformer.run_with_cache(games_ten[i*100:(i+1)*100], names_filter=f'blocks.{lay_num}.mlp.hook_pre', device='cpu')[1][f'blocks.{lay_num}.mlp.hook_pre']
#run model forward
logits = oth_mod.run_with_hooks(
        game_to_inter_enc,
        return_type='logits',
        fwd_hooks=[(
            'blocks.9.mlp.hook_pre',
            ortho_proj_hook
            )]
        )
#print(logits.shape)
probs = nn.functional.softmax(logits, dim=-1)
print('after intervention:')
show_moves_from_tensor(probs[0][2])


#print suggested moves
