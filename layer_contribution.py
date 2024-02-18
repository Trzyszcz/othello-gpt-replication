import torch
from torch import nn
import transformer_lens
import ast
import gen_oth
from train_nets import show_moves_from_tensor, read_enc_dec_dicts

#get model

oth_mod = torch.load('nets/99_good.pt')

enc_dict, dec_dict, encode, decode = read_enc_dec_dicts(6)

print(decode(encode(['A0', 'B0', 'p'])))

game_to_inter = ['B2', 'B3', 'C4', 'B5', 'A4', 'A3', 'C5', 'D5', 'B4', 'A5', 'E4', 'F3', 'F5', 'A2', 'D4', 'B1', 'A0', 'C1', 'D1', 'C0', 'A1', 'F4', 'E5', 'D0', 'F2', 'B0', 'E1', 'E2', 'E3', 'F1', 'E0', 'F0']

game_to_inter_enc = torch.tensor(encode(['s'] + game_to_inter[:-1]))

#get decoding layer
#unembed.W_U
#unembed.b_U

unembed_weights = [x[1] for x in oth_mod.named_parameters() if x[0]=='unembed.W_U'][0]
unembed_bias = [x[1] for x in oth_mod.named_parameters() if x[0]=='unembed.b_U'][0]

#get residual stream state after every layer

all_in = oth_mod.run_with_cache(game_to_inter_enc, names_filter='blocks.7.hook_resid_post')
#[1]['blocks.6.hook_resid_post']
print(all_in[0].shape)
print(all_in[0])
print('FINAL probabilities')
all_proba = nn.functional.softmax(all_in[0], dim=-1)
show_moves_from_tensor(all_proba[0][2])

for i in range(8):
    print(f'after layer{i}')
    all_in = oth_mod.run_with_cache(game_to_inter_enc, names_filter=f'blocks.{i}.hook_resid_post')
    res_after = all_in[1][f'blocks.{i}.hook_resid_post'][0][2]
    #print(res_after.shape)
    log_after = (res_after@unembed_weights) + unembed_bias
    proba_after = nn.functional.softmax(log_after, dim=-1)
    show_moves_from_tensor(proba_after)

#blocks.0.hook_resid_pre
#blocks.6.hook_resid_post #TODO what is the difference?
#ln_final.hook_normalized
