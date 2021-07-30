import torch
import random
import matplotlib.ticker as ticker
from pre_process import PreProcessData
from pre_process import Lang
from training import TrainingObject
from pickle_funcs import load_pickle
from global_vars import _model_name
from global_vars import _device
from global_vars import _MAX_LENGTH
from global_vars import _SOS_token
from global_vars import _EOS_token
from global_vars import _dictionary_name
import matplotlib.pyplot as plt

class InferenceObject:
    def __init__(self):

        self.training_data = load_pickle(_model_name)
		

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


def evaluate(dictionary, encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(_device, dictionary.input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(_MAX_LENGTH, encoder.hidden_size, device=_device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[_SOS_token]], device=_device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(_MAX_LENGTH, _MAX_LENGTH)

        for di in range(_MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == _EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(dictionary.output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
		
		
def evaluateRandomly(dictionary, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(dictionary.pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(dictionary, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
		
		

	
	
def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(_dictionary, _training, input_sentence):
    output_words, attentions = evaluate(
       _dictionary ,_training.encoder, _training.decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)





def main():

    _dictionary = load_pickle(_dictionary_name)
    _training = load_pickle(_model_name)
    evaluateRandomly(dictionary=_dictionary, encoder=_training.encoder, decoder=_training.decoder)

    evaluateAndShowAttention(_dictionary, _training, "elle a cinq ans de moins que moi .")

    evaluateAndShowAttention(_dictionary, _training, "elle est trop petit .")

    evaluateAndShowAttention(_dictionary, _training, "je ne crains pas de mourir .")

    evaluateAndShowAttention(_dictionary, _training, "c est un jeune directeur plein de talent .")

if __name__ == '__main__':
    main()

