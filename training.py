import torch
from pre_process import PreProcessData
from pre_process import Lang
from models import EncoderRNN
from models import AttnDecoderRNN
from torch import optim
import random
import time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pickle_funcs import create_pickle
from pickle_funcs import load_pickle
from global_vars import _dictionary_name
from global_vars import _device
from global_vars import _hidden_size
from global_vars import _n_iters
from global_vars import _learning_rate
from global_vars import _plot_every
from global_vars import _print_every
from global_vars import _MAX_LENGTH
from global_vars import _teacher_forcing_ratio
from global_vars import _SOS_token
from global_vars import _EOS_token
from global_vars import _epochs
from global_vars import _dropout_p


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(device, lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(_EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(device, input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(device, input_lang, pair[0])
    target_tensor = tensor_from_sentence(device, output_lang, pair[1])
    return input_tensor, target_tensor

def train(device, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=_MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[_SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < _teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == _EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def eval_test(device, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=_MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[_SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < _teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == _EOS_token:
                    break

    encoder.train()
    decoder.train()
    return loss.item() / target_length


def train_iters(device, input_lang, output_lang, pairs, encoder, decoder, n_iters, print_every, plot_every,
                learning_rate, eval=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    encoder_scheduler = MultiStepLR(optimizer=encoder_optimizer, milestones=[110143,220286], gamma=0.1)
    decoder_scheduler = MultiStepLR(optimizer=decoder_optimizer, milestones=[130000,220286], gamma=0.1)
    training_pairs = [tensors_from_pair(device, input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        if eval:
            loss = eval_test(device, input_tensor, target_tensor, encoder,
                              decoder, encoder_optimizer, decoder_optimizer, criterion)
        else:
            loss = train(device, input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

            encoder_scheduler.step()
            decoder_scheduler.step()

        print_loss_total += loss
        plot_loss_total += loss

        if iter % _print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0



    show_plot(plot_losses)
    return encoder, decoder

import numpy as np
class TrainingObject:
    def __init__(self, dictionary, device, input_lang, output_lang, pairs, n_iters, print_every, plot_every,
                 learning_rate):

        encoder1 = EncoderRNN(_device, dictionary.input_lang.n_words, _hidden_size).to(_device)
        attn_decoder1 = AttnDecoderRNN(_device, _hidden_size, dictionary.output_lang.n_words, _dropout_p).to(_device)
        random.shuffle(pairs)
        train_pairs, val_pairs = pairs[:-20000],  pairs[-20000:]
        for i in range(_epochs - 1):
            print("-------------------",i+1,"----------",learning_rate,"-----------------")
            encoder1, attn_decoder1 = train_iters(device, input_lang, output_lang, train_pairs, encoder1, attn_decoder1,
                                                  n_iters, print_every, plot_every, learning_rate)

            if i%2==0:
                print("-------------------------------------------------------------------------")
                print("evuation on validation set:")
                # val_pairs
                _,_ = train_iters(device, input_lang, output_lang, val_pairs, encoder1, attn_decoder1,
                                                 n_iters, print_every, plot_every, learning_rate, eval=True)
                print("-------------------------------------------------------------------------")


        self.encoder, self.decoder = train_iters(device, input_lang, output_lang, pairs, encoder1, attn_decoder1,
                                                 n_iters, print_every, plot_every, learning_rate)

        name = input_lang.name + "-" + output_lang.name + "-model.pickle"

        create_pickle(name, self)


def main():
    _dictionary = load_pickle(_dictionary_name)

    TrainingObject(dictionary=_dictionary, device=_device, input_lang=_dictionary.input_lang,
                   output_lang=_dictionary.output_lang, pairs=_dictionary.pairs, n_iters=_n_iters,
                   print_every=_print_every, plot_every=_plot_every, learning_rate=_learning_rate)


if __name__ == '__main__':
    main()
