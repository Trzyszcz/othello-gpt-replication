import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import transformer_lens
import ast
from copy import deepcopy
import os

with open('enc_dict6.txt', 'r') as f:
    enc_dict_data = f.read()

with open('dec_dict6.txt', 'r') as f:
    dec_dict_data = f.read()

enc_dict = ast.literal_eval(enc_dict_data)
dec_dict = ast.literal_eval(dec_dict_data)

encode = lambda tok_lst: [enc_dict[tok] for tok in tok_lst]
decode = lambda num_lst: [dec_dict[num] for num in num_lst]

print(decode(encode(['A0', 'B0', 'p'])))

with open('data_boards_me.txt', 'r') as f:
    data_boards_me = f.read()

boards_me = ast.literal_eval(data_boards_me)
print(f'Data set size: {len(boards_me)}')
#cut_out = int((1/50) * len(boards_me))
#boards_me = boards_me[:cut_out]

me_enc_dict = {'x':0, 'm':1, 'e':2}

for i in range(len(boards_me)):
    for board in boards_me[i][1]:
        for row in board:
            for j in range(len(row)):
                row[j] = me_enc_dict[row[j]]


just_boards = [game_board_pair[1] for game_board_pair in boards_me]

for sing_game_boards in just_boards:
    while len(sing_game_boards) < 33:
        sing_game_boards.append(deepcopy(sing_game_boards[-1]))

boards_me_ten = torch.tensor(just_boards)
#if torch.cuda.is_available():
#    boards_me_ten = boards_me_ten.to('cuda')
print(boards_me_ten[0][0])
print(boards_me_ten[0][:, 1, 1])

just_games = [encode(game_board_pair[0]) for game_board_pair in boards_me]
just_games_ten = torch.tensor(just_games)

oth_mod = torch.load('40_12_90_6.pt')
"""
print(just_games[0])
oth_output = oth_mod.run_with_cache(just_games_ten[0])
#shape of hook [batch, pos, mlp_dim]
print(oth_output[1]['blocks.10.mlp.hook_post'].shape)
print(oth_output[1]['blocks.10.mlp.hook_post'])
print(oth_output[1])

oth_output = oth_mod.run_with_cache(just_games_ten[:10])
print(oth_output[1]['blocks.10.mlp.hook_post'].shape)
#print(oth_mod.device)
"""
"""
oth_output = oth_mod.run_with_cache(just_games_ten[0])
print(oth_output[1]['blocks.10.mlp.hook_pre'].shape)
print(oth_output[1]['blocks.10.mlp.hook_pre'])
"""

#oth_mod.to('cpu')
#just_games_ten = just_games_ten.to('cpu')

lay = 4

def create_activation_data(htransformer, lay_num, games_ten):
    activation_data_list = []
    for i in range(100):
        activation_data_part = htransformer.run_with_cache(games_ten[i*100:(i+1)*100], names_filter=f'blocks.{lay_num}.mlp.hook_pre', device='cpu')[1][f'blocks.{lay_num}.mlp.hook_pre']
        activation_data_list.append(activation_data_part)

    activation_data = torch.cat(activation_data_list, dim=0)
    print(activation_data.shape)
    if torch.cuda.is_available():
        activation_data = activation_data.to('cuda')
    return activation_data

activation_data_10 = create_activation_data(oth_mod, lay, just_games_ten)
#print(activation_data_10)

class act_to_cell_state(Dataset):
    def __init__(self, cell_coord_x, cell_coord_y, activations, boards):
        self.size = len(activations)
        self.act_cell_state_pairs = []
        for i in range(self.size):
            cell_states = boards[i][:, cell_coord_y, cell_coord_y]
            for j in range(32):
                act = activations[i][j]
                cell_state = cell_states[j+1]
                if torch.cuda.is_available():
                    cell_state = cell_state.to('cuda')
                self.act_cell_state_pairs.append([act, cell_state])
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        X = self.act_cell_state_pairs[idx][0]
        y = self.act_cell_state_pairs[idx][1]
        return X, y

cutoff = int(0.9*len(boards_me_ten))
train_activation_data_10 = activation_data_10[:cutoff]
train_boards_me_ten = boards_me_ten[:cutoff]
val_activation_data_10 = activation_data_10[cutoff:]
val_boards_me_ten = boards_me_ten[cutoff:]

train_dataset = act_to_cell_state(1, 1, train_activation_data_10, train_boards_me_ten)
vali_dataset = act_to_cell_state(1, 1, val_activation_data_10, val_boards_me_ten)
train_dl = DataLoader(train_dataset, batch_size=100, shuffle=True)
vali_dl = DataLoader(vali_dataset, batch_size=100, shuffle=True)

class lin_prob(nn.Module):
    def __init__(self):
        super().__init__()
        self.innards = nn.Sequential(
                nn.Linear(360, 3, bias=False)
                )
    def forward(self, x):
        return self.innards(x)

prob_1_1 = lin_prob()
if torch.cuda.is_available():
    prob_1_1.to('cuda')

def errorfn(transformer, val_dl):
    dl_len = 0
    num_of_corr = 0
    for val_in, val_tar in val_dl:
        val_pred = transformer(val_in)
        quess = torch.argmax(val_pred, dim=-1)
        #tar = torch.argmax(val_tar, dim=-1)
        for i in range(len(quess)):
            dl_len += 1
            if quess[i] == val_tar[i]:
                num_of_corr += 1
    perc_of_corr = (num_of_corr/dl_len)*100
    return perc_of_corr

lossfn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(prob_1_1.parameters(), lr=5e-6)

epochs = 800

print('start training')

best_loss = 30
best_loss_epoch = 1

best_error = 0
best_error_epoch = 0

for epoch in range(epochs):
    for idx, (train_in, train_tar) in enumerate(train_dl):
        opt.zero_grad()
        train_pred = prob_1_1(train_in)
        train_loss = lossfn(train_pred, train_tar)
        #if idx%10==0:
        #print(train_loss.item())
        #if train_loss.item() < best_loss:
        #    best_loss = train_loss.item()
        #    best_loss_epoch = epoch + 1
        if idx%10==0:
            with torch.no_grad():
                print(train_loss.item())
                error_rate = errorfn(prob_1_1, vali_dl)
                if error_rate > best_error:
                    print('NEW BEST')
                    torch.save(prob_1_1, 'best_prob_1_1.pt')
                    best_error = error_rate
                    best_error_epoch = epoch + 1
        train_loss.backward()
        opt.step()
    print(f'End of epoch {epoch + 1}')

print('end of training')
print(f'Best error: {best_error} during epoch {best_error_epoch}')
best_prob_1_1 = torch.load('best_prob_1_1.pt')
print(errorfn(best_prob_1_1, vali_dl))
torch.save(best_prob_1_1, f'lay{lay}/{int(best_error)}_good_prob_1_1.pt')
#os.mkdir(f'100k/lay{lay}')
#torch.save(best_prob_1_1, f'100k/lay{lay}/{int(best_error)}_good_prob_1_1.pt')
