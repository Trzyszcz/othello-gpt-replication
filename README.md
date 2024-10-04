# Description

This code replicates intervention procedure from "Emergent Linear Representations in World Models of Self-Supervised Sequence Models" by Nanda et al. (https://arxiv.org/abs/2309.00941), and provides text UI interface for it.
Model based on transformer architecture is train to predict legal next moves for game Othello (https://en.wikipedia.org/wiki/Reversi#Othello). It doesn't guess based on given board state, but by predicting next token of game transcription.
Then linear probes are trained to guess state of individual tiles from state of residual stream, based on the assumption that network calculates representation of the boards state, with different directions in activation space encoding state
of different tiles. By moving activation vector at specified layers along those directions we intervene in the forward pass, to flip the state of specific tiles. If model uses this representation of the board state to predict next legal moves,
such intervention should affect this prediction by making certain moves legal or illegal.

I trained network on smaller version of Othello, played on a 6x6 board instead of 8x8.

# Instalation
Pytorch and transformer_lens libraries are needed.

# How it works

To intervene on a custom game, write your game to `games/game_to_inter` file, with moves separated by ", ". It uses notation in which columns are numbered from 0 to 5.
Then open `intervention.py` using python3
![image](https://github.com/user-attachments/assets/e667c57d-e2af-487d-9fde-21102d475d68)

Now you can chose move on which you want to intervene, by navigating forward, backward or jumping to specific one. To chose current move enter "I"

![image](https://github.com/user-attachments/assets/cbc916c0-09ab-4c08-b96c-45f003269ae0)

Let say we want to change piece on F3 from white to black. As white is to move now, we want to chose option "mine to enemy" after providing coordinates. This particular intervention works well on layer 2 with scaling parameter 20.

![image](https://github.com/user-attachments/assets/73c9e977-6551-4a12-891c-2e1009f00345)

Then program will show us "Imaginary board before intervention". It is the state of the board based on information probes gather form residual stream after chosen layer. The comparison between this and "Imaginary board after intervention"
is very usefull for chosing scaling parameter - if it is to small, both boards will be identical, if it is too big, the second board will get quite messy. After that we get moves which are legal according to the model before intervention.

![image](https://github.com/user-attachments/assets/9247da2e-a27a-4384-bda6-b42800854930)

Finally, additional legal moves according to model after intervention are printed in green. If model guesses that some moves are no longer legal, program will print them in red. For example:

![image](https://github.com/user-attachments/assets/b42213ef-a1f7-4cdb-8836-8d5598a22959)

As you can see, "imaginary board after intervention" isn't very sensible, although this is the state after layer 2, there are still six more layers to correct it (or the probes don't read the internal representation of the board that well,
or model's internal representation isn't that precice).

`gen_oth.py` can be used to generate data for training model to predict moves for a board of a arbitraty even size bigger than 2.
`train_nets.py` can be used to train model.
`train_linear_probes.py` can be used to train probes.
`visualise_board_from_activations.py` can be used to print "imaginary board state" after every move in the game.
