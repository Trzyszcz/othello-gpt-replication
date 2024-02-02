import torch
from torch import nn
import transformer_lens
import ast
import gen_oth_act

#get model


oth_mod = torch.load('40_12_90_6.mod')

with open('enc_dict6.txt', 'r') as f:
    enc_dict_data = f.read()

with open('dec_dict6.txt', 'r') as f:
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


probe = torch.load('lay9/92_good_prob_1_1.mod')

#get vectors from probe

me_enc_dict = {'x':0, 'm':1, 'e':2}
for par in probe.named_parameters():
    print(par)

weights = [x[1] for x in probe.named_parameters() if x[0]=='innards.0.weight']
#bias = [x[1] for x in probe.named_parameters() if x[0]=='innards.0.bias']

print(weights)


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
game1.print_board()

#print suggested moves

def show_moves_from_tensor(tens):
    for i in range(len(tens)):
        if tens[i].item() >= 0.001:
            print('{} : {:.3f}'.format(decode([i])[0], tens[i].item()))


logits = oth_mod.forward(game_to_inter_enc)
#print(logits.shape)
probs = nn.functional.softmax(logits, dim=-1)
print('before intervention:')
show_moves_from_tensor(probs[0][1])

#run model to a point

'''
    We are interested in cached 'blocks.9.ln2.hook_normalized', 'blocks.9.mlp.hook_pre', 'blocks.9.mlp.hook_post', 'blocks.9.hook_mlp_out', 'blocks.9.hook_resid_post'
    Assuming hook_resid_post is ln2.hook_normalised plus mlp_out, I keep the first one, project mlp_pre, get it through ReLU and projection back to residual stream
'''

'''
    This code checks if mlp_out plus resid_mid gives us resid_post (I didn't check tutorial on HookedTransformers)

resmid9 = oth_mod.run_with_cache(game_to_inter_enc, names_filter='blocks.9.hook_resid_mid', device='cpu')[1]['blocks.9.hook_resid_mid'][0][1]
mlp_out9 = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.9.hook_mlp_out', device='cpu')[1][f'blocks.9.hook_mlp_out'][0][1]
respost9 = oth_mod.run_with_cache(game_to_inter_enc, names_filter='blocks.9.hook_resid_post', device='cpu')[1]['blocks.9.hook_resid_post'][0][1]
print((resmid9 + mlp_out9) == respost9)
'''

'''
    This code checks if mlp_pre after ReLU equals mlp_post

act_mlp9 = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.9.mlp.hook_pre', device='cpu')[1][f'blocks.9.mlp.hook_pre'][0][1]
relufn = nn.ReLU()
post_mlp9 = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.9.mlp.hook_post', device='cpu')[1][f'blocks.9.mlp.hook_post'][0][1]
print(relufn(act_mlp9) == post_mlp9)
'''

'''
    This code checks if mlp_post afte multiplied by linear layer projecting to residual stream equals mlp_out
'''
post_mlp9 = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.9.mlp.hook_post', device='cuda')[1][f'blocks.9.mlp.hook_post'][0][1]
mlp_out9 = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.9.hook_mlp_out', device='cuda')[1][f'blocks.9.hook_mlp_out'][0][1]

named_params = oth_mod.named_parameters()
#names = []
#for named_param in named_params:
#    names.append(named_param[0])

#print(names)
#'blocks.9.mlp.W_out', 'blocks.9.mlp.b_out'
for named_param in named_params:
    if named_param[0] == 'blocks.9.mlp.W_out':
        linlay_w = named_param[1]
        #print(linlay_w.shape)
    if named_param[0] == 'blocks.9.mlp.b_out':
        linlay_b = named_param[1]
        #print(linlay_b.shape)

print( ((linlay_w.T@post_mlp9) + linlay_b)[1] )
print( ((linlay_w.T@post_mlp9) + linlay_b)[1] == mlp_out9[1])
print( mlp_out9[1] )

'''
innards = oth_mod.run_with_cache(game_to_inter_enc)
print(innards)

act_mlp9 = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.9.mlp.hook_pre', device='cpu')[1][f'blocks.9.mlp.hook_pre'][0][1]
print(f'mlp pre shape: {act_mlp9.shape}')
ln29 = oth_mod.run_with_cache(game_to_inter_enc, names_filter='blocks.9.ln2.hook_normalized', device='cpu')[1]['blocks.9.ln2.hook_normalized'][0][1]
print(f'ln2norm shape: {ln29.shape}')
print(f'respost9 shape: {respost9.shape}')
resmid9 = oth_mod.run_with_cache(game_to_inter_enc, names_filter='blocks.9.hook_resid_mid', device='cpu')[1]['blocks.9.hook_resid_mid'][0][1]
print(f'resmid9 shape: {resmid9.shape}')

mlp_out9 = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.9.hook_mlp_out', device='cpu')[1][f'blocks.9.hook_mlp_out'][0][1]
'''


#do the intervention

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

    #norm_of_eq_vec = eq_vector.T @ eq_vector

    transpo_t_vector = t_vector - transpo_vector

    dot_prod = eq_vector.T @ transpo_t_vector

    projected_vector = t_vector - (eq_vector * (dot_prod/(eq_vector.T @ eq_vector)))

    return projected_vector

def ortho_proj_test():
    a = torch.tensor([-1.0, -1.0])
    t = torch.tensor([4.0, 1.0])
    print(ortho_proj(a, t))

    t2 = torch.tensor([-3, -2])
    print(ortho_proj(a, t2))

#activation_data_part = htransformer.run_with_cache(games_ten[i*100:(i+1)*100], names_filter=f'blocks.{lay_num}.mlp.hook_pre', device='cpu')[1][f'blocks.{lay_num}.mlp.hook_pre']
#run model forward

#print suggested moves
