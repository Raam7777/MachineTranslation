import torch

# Shared vars

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_hidden_size = 256

_MAX_LENGTH = 15

_SOS_token = 0

_EOS_token = 1

# Pre Process vars

_lang1 = 'eng'

_lang2 = 'fra'

_reverse = False

# Training vars

_dictionary_name = "dictionaries/eng-fra-dictionary.pickle"

_dropout_p=0.1

_teacher_forcing_ratio = 1

_n_iters = 110143

_print_every=50

_plot_every=100

_learning_rate=0.01

_epochs = 10

# Inference vars

_model_name = "training_models/eng-fra-model.pickle"




