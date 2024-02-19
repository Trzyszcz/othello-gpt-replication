import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import transformer_lens
import ast
from copy import deepcopy
import os
from train_nets import read_enc_dec_dicts

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

class act_to_cell_state(Dataset):
    def __init__(self, cell_coord_x, cell_coord_y, activations, boards):
        act_len = len(activations)
        self.act_cell_state_pairs = []
        for i in range(act_len):
            cell_states = boards[i][:, cell_coord_x, cell_coord_y]
            for j in range(32):
                act = activations[i][j]
                cell_state = cell_states[j]
                if torch.cuda.is_available():
                    cell_state = cell_state.to('cuda')
                self.act_cell_state_pairs.append([act, cell_state])
        self.size = len(self.act_cell_state_pairs)
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        X = self.act_cell_state_pairs[idx][0]
        y = self.act_cell_state_pairs[idx][1]
        return X, y

class lin_prob(nn.Module):
    def __init__(self, act_space_size):
        super().__init__()
        self.innards = nn.Sequential(
                nn.Linear(act_space_size, 3, bias=False)
                )
    def forward(self, x):
        return self.innards(x)

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

def train_probe(probe, train_dataloader, val_dataloader, coords, layer): 
    lossfn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(probe.parameters(), lr=5e-6)

    epochs = 100

    print('start training')

    best_loss = 30
    best_loss_epoch = 1

    best_error = 0
    best_error_epoch = 0

    for epoch in range(epochs):
        for idx, (train_in, train_tar) in enumerate(train_dataloader):
            opt.zero_grad()
            train_pred = probe(train_in)
            train_loss = lossfn(train_pred, train_tar)
            if idx%1000==0:
                with torch.no_grad():
                    print(train_loss.item())
                    error_rate = errorfn(probe, val_dataloader)
                    if error_rate > best_error:
                        print('NEW BEST')
                        torch.save(probe, f'probes/best_prob_{coords[0]}_{coords[1]}_lay{lay}.pt')
                        best_error = error_rate
                        best_error_epoch = epoch + 1
            train_loss.backward()
            opt.step()
        print(f'End of epoch {epoch + 1}')

    print('end of training')
    print(f'Best error: {best_error} during epoch {best_error_epoch}')
    best_probe = torch.load(f'probes/best_prob_{coords[0]}_{coords[1]}_lay{lay}.pt')
    print(errorfn(best_probe, val_dataloader))
    torch.save(best_probe, f'probes/lay{lay}/{int(best_error)}_good_prob_{coords[0]}_{coords[1]}_lay{lay}.pt')

def train_probe_for_coord(coord):

    train_dataset = act_to_cell_state(coord[0], coord[1], train_activation_data, train_boards_me_ten)
    vali_dataset = act_to_cell_state(coord[0], coord[1], val_activation_data, val_boards_me_ten)
    train_dl = DataLoader(train_dataset, batch_size=100, shuffle=True)
    vali_dl = DataLoader(vali_dataset, batch_size=100, shuffle=True)

    probe = lin_prob(1280)
    if torch.cuda.is_available():
        probe.to('cuda')

    train_probe(probe, train_dl, vali_dl, coord, lay)

    return probe

def train_probes_for_layer(layer, board_d):
    probes_for_cells = []
    for i in range(board_d):
        probes_for_row = []
        for j in range(board_d):
            probes_for_row.append(train_probe_for_coord([i, j]))
        probes_for_cells.append(deepcopy(probes_for_row))
    return deepcopy(probes_for_cells)

#def train_lin_probes_main():
if __name__ == '__main__':
    enc_dict, dec_dict, encode, decode = read_enc_dec_dicts(6)

    print(decode(encode(['A0', 'B0', 'p'])))

    with open('data/data_boards_me.txt', 'r') as f:
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

    just_games = [encode(['s'] + game_board_pair[0][:-1]) for game_board_pair in boards_me]
    just_games_ten = torch.tensor(just_games)

    oth_mod = torch.load('nets/99_good.pt')

    #lay = 1
    layers = [0, 2, 3, 4, 5, 6]
    #num_of_lay = 8
    board_dim = 6

    for lay in layers:
        activation_data = create_activation_data(oth_mod, lay, just_games_ten)

        cutoff = int(0.9*len(boards_me_ten))
        train_activation_data = activation_data[:cutoff]
        train_boards_me_ten = boards_me_ten[:cutoff]
        val_activation_data = activation_data[cutoff:]
        val_boards_me_ten = boards_me_ten[cutoff:]

        probe = train_probe_for_coord([1, 2])
        #probes = train_probes_for_layer(lay, board_dim)

        #for lay in range(num_of_lay):

#if __name__ == '__main__':
#    train_lin_probes_main()


