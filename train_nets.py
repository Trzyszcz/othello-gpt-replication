import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
#import matplotlib.pyplot as plt
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig
import ast
import gen_oth
from gen_oth import how_many_correct_moves
from datetime import datetime

def create_enc_dec_dicts(vocab, board_size):
    enc_dict = {tok:num for num, tok in enumerate(vocab)}
    dec_dict = {num:tok for num, tok in enumerate(vocab)}

    with open(f'enc_dec_dicts/enc_dict{board_size}.txt', 'x') as f:
        f.write(enc_dict.__str__())

    with open(f'enc_dec_dicts/dec_dict{board_size}.txt', 'x') as f:
        f.write(dec_dict.__str__())

    encode = lambda tok_lst: [enc_dict[tok] for tok in tok_lst]
    decode = lambda num_lst: [dec_dict[num] for num in num_lst]

    return enc_dict, dec_dict, encode, decode

def read_enc_dec_dicts(board_size):
    with open(f'enc_dec_dicts/enc_dict{board_size}.txt', 'r') as f:
        enc_dict_data = f.read()

    with open(f'enc_dec_dicts/dec_dict{board_size}.txt', 'r') as f:
        dec_dict_data = f.read()

    enc_dict = ast.literal_eval(enc_dict_data)
    dec_dict = ast.literal_eval(dec_dict_data)

    encode = lambda tok_lst: [enc_dict[tok] for tok in tok_lst]
    decode = lambda num_lst: [dec_dict[num] for num in num_lst]

    return enc_dict, dec_dict, encode, decode

def show_moves_from_tensor(tens):
    _, _, _, decode = read_enc_dec_dicts(6)

    for i in range(len(tens)):
        if tens[i].item() >= 0.0001:
            print('{} : {:.3f}'.format(decode([i])[0], tens[i].item()))

if __name__ == '__main__':
    data_file = open('data6.txt', 'r')
    lines = []
    for line in data_file:
        lines.append(['s'] + line.split(', ')[:-1])
    print(lines[0:3])
    data_file.close()

    vocab = set()
    for line in lines:
        for token in line:
            vocab.add(token)

    data_size = len(lines)
    vocab_size = len(vocab)

    print(f'Data size: {data_size}')
    print(f'Vocab size: {vocab_size}')

    enc_dict, dec_dict, encode, decode = read_enc_dec_dicts(6)

    print(decode(encode(['A0', 'B0', 'p'])))

    lines = [encode(line) for line in lines]
    split_point = int(0.9*data_size)
    train_data = lines[:split_point]
    val_data = lines[split_point:]
    train_loss_data = []
    val_loss_data = []

    class Oth_dataset(Dataset):
        def __init__(self, in_data):
            temp_tens = torch.tensor(in_data)
            if torch.cuda.is_available():
                self.data = temp_tens.to('cuda')
            else:
                self.data = temp_tens
            self.size = len(in_data)
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            X = self.data[idx][:-1]
            y = self.data[idx][1:]
            return X, y

    train_dl = DataLoader(Oth_dataset(train_data), batch_size=200, shuffle=True)
    val_dl = DataLoader(Oth_dataset(val_data), batch_size=100, shuffle=True)
    val_iter = iter(val_dl)

    print('data constructed')

    def generate_game(trans_model):
        input_tok = torch.tensor([encode(['s' for _ in range(32)])])
        gen_game = []
    
        for i in range(31):
            logits = trans_model.forward(input_tok)
            prob = nn.functional.softmax(logits, dim=-1)
            token = torch.multinomial(prob[0][i], 1)
            gen_game.append(token.item())
            input_tok[0][i+1] = token.item()
        return gen_game

    def check_model(trans_model):
        games = []
        cor_num = 0
        for i in range(50):
            games.append(decode(generate_game(oth_net)))
            if gen_oth.check_game(6, games[i]):
                cor_num += 1
            #print(gen_oth.check_game(4, games[i]))
        print(f'{cor_num*2}/100')
        return cor_num

    def cor_num_of_moves(trans_model, val_iterator, val_dataloader, uniform=True):
    
        try:
            val_in, val_tar = next(val_iterator)
        except StopIteration:
            val_iterator = iter(val_dataloader)
            val_in, val_tar = next(val_iterator)

        #print(val_in.shape)
        #print(decode(val_in[0].tolist()))
        logits = oth_net(val_in, return_type='logits')
        probs = nn.functional.softmax(logits, dim=-1)
        #for i in range(2):
        #    show_moves_from_tensor(probs[0][i])

        number_of_correct_moves = 0
        number_of_moves = 0

        for i in range(len(probs)):
            #print(logits[i])
            if uniform:
                moves_encoded = torch.multinomial(probs[i], 1)
            else:
                moves_encoded = torch.argmax(probs[i], dim=-1)
            game = torch.squeeze(moves_encoded)
            #print(game.shape)
            moves_list = game.tolist()
            number_of_moves += len(moves_list)
            moves_list_decoded = decode(moves_list)
            actual_moves = decode(val_in[i].tolist()[1:] + [val_tar[i].tolist()[-1]])
            #print(actual_moves)
            #print(moves_list_decoded)
            number_of_correct_moves += how_many_correct_moves(6, actual_moves, moves_list_decoded)

        percent_of_cor = (number_of_correct_moves/number_of_moves) * 100
    
        return percent_of_cor

    #cfg = HookedTransformerConfig(n_layers=12, d_model=90, n_ctx=32, d_head=6, d_vocab=vocab_size, act_fn='gelu')
    cfg = HookedTransformerConfig(n_layers=8, d_model=480, n_ctx=32, d_head=60, d_vocab=vocab_size, act_fn='gelu')

    oth_net = HookedTransformer(cfg)

    #oth_net = torch.load('best_oth_mod.pt')

    if torch.cuda.is_available():
        oth_net.to('cuda')

    lossfn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(oth_net.parameters(), lr=1e-4)

    epochs = 20

    now = datetime.now()
    current_time_start = now.strftime('%H:%M:%S')
    print(f'Start training {current_time_start}')

    best_result = 0
    best_result_epoch = 0

    for epoch in range(epochs):
        for idx, (train_in, train_tar) in enumerate(train_dl):
            opt.zero_grad()
            train_loss = oth_net(train_in, return_type='loss')
            #print(train_loss.shape)
            if idx%100==0:
                print(f'Train loss:{train_loss.item()}')
        
            if idx%200==0:
                with torch.no_grad():
                    #current_result = check_model(oth_net)
                    current_result = cor_num_of_moves(oth_net, val_iter, val_dl, uniform=False)
                    if current_result > best_result:
                        print(f'NEW BEST epoch {epoch + 1} correct {current_result}')
                        torch.save(oth_net, 'best_oth_mod.pt')
                        #print(f'change best result to {current_result}')
                        best_result = current_result
                        best_result_epoch = epoch + 1
        
            train_loss.backward()
            opt.step()
        print(f'end of epoch {epoch+1}')

    print(f'best result: {best_result} in epoch {best_result_epoch}')

    now = datetime.now()
    current_time = now.strftime('%H:%M:%S')
    print(f'Start training {current_time_start}')
    print(f'End training {current_time}')

    torch.save(oth_net, 'oth_mod.pt')

    check_model(oth_net)
    print(cor_num_of_moves(oth_net, val_iter, val_dl, uniform=False))
