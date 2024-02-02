import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
#import matplotlib.pyplot as plt
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig
import ast
import gen_oth

data_file = open('data.txt', 'r')
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
"""
enc_dict = {tok:num for num, tok in enumerate(vocab)}
dec_dict = {num:tok for num, tok in enumerate(vocab)}

with open('enc_dict.txt', 'x') as f:
    f.write(enc_dict.__str__())

with open('dec_dict.txt', 'x') as f:
    f.write(dec_dict.__str__())
"""

with open('enc_dict.txt', 'r') as f:
    enc_dict_data = f.read()

with open('dec_dict.txt', 'r') as f:
    dec_dict_data = f.read()

enc_dict = ast.literal_eval(enc_dict_data)
dec_dict = ast.literal_eval(dec_dict_data)

encode = lambda tok_lst: [enc_dict[tok] for tok in tok_lst]
decode = lambda num_lst: [dec_dict[num] for num in num_lst]

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
val_dl = DataLoader(Oth_dataset(val_data), batch_size=200, shuffle=True)

print('data constructed')

def show_moves_from_tensor(tens):
    for i in range(len(tens)):
        print('{} : {:.3f}'.format(decode([i])[0], tens[i].item()))

def generate_game(trans_model):
    input_tok = torch.tensor([encode(['s' for _ in range(12)])])
    gen_game = []

    for i in range(11):
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
        if gen_oth.check_game(4, games[i]):
            cor_num += 1
        #print(gen_oth.check_game(4, games[i]))
    print(f'{cor_num*2}/100')
    return cor_num



cfg = HookedTransformerConfig(n_layers=14, d_model=36, n_ctx=12, d_head=6, d_vocab=vocab_size, act_fn='relu')
oth_net = HookedTransformer(cfg)

if torch.cuda.is_available():
    oth_net.to('cuda')

lossfn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(oth_net.parameters(), lr=1e-2)

epochs = 14

print('start training')

best_result = 0
best_result_epoch = 0

for epoch in range(epochs):
    for idx, (train_in, train_tar) in enumerate(train_dl):
        '''
        print(train_tar.shape)
        print(train_tar[0])
        opt.zero_grad()
        train_pred = oth_net(train_in)
        print('\n \npred \n \n')
        print(train_pred.shape)
        print(train_pred[0])
        train_loss = lossfn(train_pred, train_tar)
        '''
        opt.zero_grad()
        train_loss = oth_net(train_in, return_type='loss')
        #print(train_loss.shape)
        if idx%10==0:
            print(train_loss.item())
        if idx%100==0:
            current_result = check_model(oth_net)
            print(f'epoch {epoch + 1}')
            if current_result > best_result:
                torch.save(oth_net, 'best_oth_mod.mod')
                #print(f'change best result to {current_result}')
                best_result = current_result * 2
                best_result_epoch = epoch + 1
        train_loss.backward()
        opt.step()

print(f'best result: {best_result} in epoch {best_result_epoch}')

torch.save(oth_net, 'oth_mod.mod')



#oth_net = torch.load('oth_mod2.mod')

#print(oth_net.generate(input=torch.tensor([encode(['s'])]), max_new_tokens=13, stop_at_eos=False, return_type='tensor', padding_side='right'))
#print(encode(['s']))
#print(oth_net.generate(input=torch.tensor([encode(['s'])]), stop_at_eos=False))

#print(games)

"""
input_tok = torch.tensor([encode(['s' for _ in range(12)])])
gen_game = []

for i in range(11):
    logits = oth_net.forward(input_tok)
    #print(logits[0][0])
    prob = nn.functional.softmax(logits, dim=-1)
    #print(prob[0][0])
    show_moves_from_tensor(prob[0][i])
    token = torch.multinomial(prob[0][i], 1)
    #print(output_tok.shape)
    print(decode(token.tolist()))
    gen_game.append(token.item())
    input_tok[0][i+1] = token.item()

print(decode(gen_game))
"""
'''
logits = oth_net.forward(input_tok)
print(logits[0][1])
prob = nn.functional.softmax(logits, dim=-1)
print(prob[0][1])
show_moves_from_tensor(prob[0][1])
token = torch.multinomial(prob[0][1], 1)
#print(output_tok.shape)
print(decode(token.tolist()))
'''

