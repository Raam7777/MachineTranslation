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

_reverse = True

# Training vars

_dictionary_name = "dictionaries/fra-eng-dictionary.pickle"

_teacher_forcing_ratio = 0.5

_n_iters = 50

_print_every=5

_plot_every=100

_learning_rate=0.01

_epochs = 1

# Inference vars

_model_name = "training_models/fra-eng-model.pickle"




