import torch

# Shared vars

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_hidden_size = 256

_MAX_LENGTH = 15

_SOS_token = 0

_EOS_token = 1

# Pre Process vars

_lang1 = 'heb'

_lang2 = 'arm'

_reverse = True

# Training vars

_dictionary_name = "dictionaries/eng-fra-dictionary.pickle"

_teacher_forcing_ratio = 0.5

_n_iters = 50

_print_every=5

_plot_every=100

_learning_rate=0.01

_epochs = 5

# Inference vars

_model_name = "training_models/fra-eng-model.pickle"


