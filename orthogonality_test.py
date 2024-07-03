import torch
from torch import nn
import transformer_lens
import ast
import gen_oth
from train_nets import show_moves_from_tensor, read_enc_dec_dicts
from train_linear_probes import lin_prob
from copy import deepcopy
from pprint import pprint
import os
import seaborn as sns
import matplotlib.pyplot as plt

enc_dict, dec_dict, encode, decode = read_enc_dec_dicts(6)

print(decode(encode(['A0', 'B0', 'p'])))

lay = 3

inside_probes_folder = os.listdir(f'./probes/lay{lay}')

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

probes = load_probes(6, lay, inside_probes_folder)

def every_on_every(board_dim, probes_table, probe_type):

    probes_list = []

    for i in range(6):
        probes_list = probes_list + probes[i]

    result_table = []

    for i in range(36):
        res_table_row = []
        for j in range(36):
            probe_i = probes_list[i]
            probe_j = probes_list[j]
            pr_i_vector = probe_i.get_parameter('innards.0.weight')[probe_type]
            pr_j_vector = probe_j.get_parameter('innards.0.weight')[probe_type]
            dot_prod = (pr_i_vector.T@pr_j_vector).item()
            res_table_row.append(dot_prod - (dot_prod%0.01))  
        result_table.append(deepcopy(res_table_row))
    
    return result_table

def every_on_every_board_format(board_dim, probes_table, probe_type):
    res_table = []

    for i in range(board_dim):
        row_res_table = []
        for j in range(board_dim):
            probes_cell = []
            base_probe = probes_table[i][j].get_parameter('innards.0.weight')[probe_type]
            for k in range(board_dim):
                probes_row = []
                for l in range(board_dim):
                    probe_k_l = probes_table[k][l].get_parameter('innards.0.weight')[probe_type]
                    dot_prod = (base_probe.T@probe_k_l).item()
                    probes_row.append(dot_prod - (dot_prod%0.01))
                probes_cell.append(deepcopy(probes_row))
            row_res_table.append(deepcopy(probes_cell))
        res_table.append(deepcopy(row_res_table))

    return res_table

res_table = every_on_every_board_format(6, probes, 2)

#pprint(result_table)
result_ten = torch.tensor(res_table)

with open('list.txt', 'x') as f:
    f.write(result_ten.tolist().__str__())

'''
a = torch.rand(4, 4)
a_lst = a.tolist()
sns.heatmap(a_lst, annot=True, cmap="YlGnBu")
#sns.heatmap(result_ten.tolist(), annot=True, cmap="YlGnBu")
plt.show()
'''

'''
#get vectors from probe

me_enc_dict = {'x':0, 'm':1, 'e':2}
#for par in probe.named_parameters():
#    print(par)

weights = [x[1] for x in probe.named_parameters() if x[0]=='innards.0.weight']
#bias = [x[1] for x in probe.named_parameters() if x[0]=='innards.0.bias']

#print(weights.shape)
#print(weights)

enemy_vector = weights[0][2]
empty_vector = weights[0][0]
my_vector = weights[0][1]
#
'''
