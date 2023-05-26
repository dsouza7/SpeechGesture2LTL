# Importing torch and torchvision
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
import torch
import torchvision
import pandas as pd
import tensorflow_hub as hub
import spacy
import time
import torchtext
from torchtext.data import get_tokenizer
import random
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_excel("C:\\Users\\CHRIS\\Desktop\\dataset.xlsx")
# Defining dataset
class LTL_Dataset(Dataset):
    def __init__(self, speech, ltls):
        self.speech = speech
        # self.gestures = gestures
        self.ltls = ltls
        # # Converting to PyTorch tensors
        # self.speech = torch.tensor(self.speech, dtype=torch.float32)
        # # self.gestures = torch.tensor(self.speech, dtype=torch.float32)
        # self.ltls = torch.tensor(self.ltls, dtype=torch.float32)

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, index):
        return self.speech[index], self.ltls[index]
def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
# Data preprocessing
def preprocess_data(speech, ltls):
    # Convert speech to integer sequences
    word2index = {}
    for sentence in speech:
        for word in sentence.split():
            if word not in word2index:
                word2index[word] = len(word2index)
    sequences = [[word2index[word] for word in sentence.split()] for sentence in speech]
    print("sequences:",len(sequences))
    # Pad the sequences
    max_length = 198
    print("length",max_length)
    padded_speech = torch.zeros(len(sequences), max_length, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded_speech[i, :len(seq)] = torch.tensor(seq)
    # Convert ltls to integer labels
    label_encoder = LabelEncoder()
    ltls_encoded = label_encoder.fit_transform(ltls)
    print("ltls",ltls_encoded)
    # Convert ltls_encoded to one-hot vectors
    # one_hot_ltls = torch.stack([F.one_hot(torch.tensor(ltls), num_classes=len(ltls_encoded)).squeeze(0) for ltls in ltls_encoded])
    one_hot_ltls = torch.stack([F.one_hot(torch.tensor(ltls, dtype=torch.long), num_classes=len(ltls_encoded)).squeeze(0) for ltls in ltls_encoded])
    print("ltls",len(one_hot_ltls))
    return padded_speech, one_hot_ltls, ltls_encoded
#just speech
def preprocess_speech(speech):
    # Convert speech to integer sequences
    word2index = {}
    for sentence in speech:
        for word in sentence.split():
            if word not in word2index:
                word2index[word] = len(word2index)
    sequences = [[word2index[word] for word in sentence.split()] for sentence in speech]
    print("words",len(sequences))
    # Pad the sequences
    max_length = max([len(seq) for seq in sequences])
    padded_speech = torch.zeros(len(sequences), max_length, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded_speech[i, :len(seq)] = torch.tensor(seq)
    # # Convert to PyTorch tensor
    # embeddings = torch.tensor(embeddings.numpy(), dtype=torch.float32)
    # # Pad the sequences
    # padded_speech = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)

    return padded_speech
#seq2seq model
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2Seq, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs, targets=None):
        if targets is not None:
            encoder_outputs, hidden = self.encoder(inputs)
            print("Encoder Outputs shape:", encoder_outputs.shape)
            print("Hidden state shape:", hidden[0].shape)
            print("Cell state shape:", hidden[1].shape)
            # Initialize the decoder hidden and cell state with the encoder's final hidden and cell state
            decoder_input = torch.zeros(targets.size(0), 1, targets.size(1), device=device)
            print(decoder_input[0].size())
            # decoder_hidden = hidden
            # decoder_cell = torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1])
            decoder_hidden = hidden[0].unsqueeze(0)  # shape: [1, 1, 128]
            decoder_cell = hidden[1].unsqueeze(0)
            print(decoder_hidden.size())
            print(decoder_cell.size())
            print(targets.size(0))
            decoder_hidden = decoder_hidden.expand(1, targets.size(0), -1)  # Shape: (1, 158, 128)
            decoder_cell = decoder_cell.expand(1, targets.size(0), -1)
            print(decoder_hidden.size())
            print(decoder_cell.size())
            decoder_outputs = []
            targets = targets.unsqueeze(1)
            print(targets.size())
            # Decode the output sequence
            for i in range(targets.size(1)):

                output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
                # output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden.squeeze(0), decoder_cell.squeeze(0)))
                output = self.fc(output)
                decoder_outputs.append(output)
                decoder_input = targets[:, i, :].unsqueeze(1)

            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            return decoder_outputs
        else:
            # Inference mode
            encoder_outputs, hidden = self.encoder(inputs)
            print("Encoder Outputs shape:", encoder_outputs.shape)
            print("Hidden state shape:", hidden[0].shape)
            print("Cell state shape:", hidden[1].shape)
            decoder_input = torch.zeros(inputs.size(0), 1, self.output_dim, device=inputs.device)
            decoder_hidden = hidden[0].unsqueeze(0)
            decoder_cell = hidden[1].unsqueeze(0)
            decoder_hidden = decoder_hidden.expand(1, inputs.size(0), -1)  # Shape: (1, 158, 128)
            decoder_cell = decoder_cell.expand(1, inputs.size(0), -1)
            print(decoder_input.size())
            print(decoder_hidden.size())
            print(decoder_cell.size())
            print(inputs.size(0))
            predicted_outputs = []
            # print(decoder_input.size())
            for i in range(100):  # length of seq
                print("1",decoder_input.size())
                decoder_input = decoder_input.float()
                output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
                print("2",decoder_input.size())
                output = self.fc(output)
                print("3",decoder_input.size())
                predicted_token = output.argmax(dim=2, keepdim=True)
                print("4",decoder_input.size())
                print("4",predicted_token.size())
                predicted_outputs.append(predicted_token)
                print("5",decoder_input.size())
                # decoder_input = predicted_token
                decoder_input = predicted_token.expand(-1, 1, -1)
                print("6",decoder_input.size())
                print("Predicted token",predicted_token)

            predicted_outputs = torch.cat(predicted_outputs, dim=1)
            return predicted_outputs
#Convert to ltl
# def convert_to_ltl(predicted_ltl):
#     # Define the token-to-symbol mapping
#     mapping = {0: 'symbol1', 1: 'symbol2', 2: 'symbol3', ...}
#
#     # Convert predicted LTL tokens to symbols
#     ltl_symbols = [mapping[token_idx] for token_idx in predicted_ltl]
#
#     # Concatenate the symbols into a string
#     ltl_string = ' '.join(ltl_symbols)
#
#     return ltl_string
# Loading the dataset
# spacy_en = spacy.load('en_core_web_sm')
df = pd.read_excel("C:\\Users\\CHRIS\\Desktop\\dataset.xlsx")
# dataset = np.load('NatSGD_v0.3.npz',allow_pickle=True)['NatComm'].item()
# dataset = np.load('NatSGD_v0.3.npz', allow_pickle=True)['NatComm'][()]
# # pid = dataset['PID'].to_numpy()
# # DBSN = dataset['DBSN'].to_numpy()
# for datauid in dataset['']:
#     if dataset['PID'][datauid][0] == DBSN:
#         print('This is your record')
#Check for missing values
# df=df.isnull()
#Removing null values in LTL
# df['LTL'] = df['LTL'].replace(r'^\s*$', np.nan, regex=True)
# df = df.dropna()
# #Removing Notes1 col
# df=df.drop(['Notes1'], axis=1)
#Getting speech and LTL data
speech = df['Notes'].astype(str).to_numpy()
ltls = df['LTL'].to_numpy()
print("LTL",ltls)
#pre-
# TRG = Field(tokenize = tokenize_en,init_token = '<sos>',eos_token = '<eos>',lower = True,batch_first = True)
# train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),fields = (SRC, TRG))
padded_speech, one_hot_ltls, ltls_encoded = preprocess_data(speech, ltls)
print(len(padded_speech))
print(len(ltls_encoded))
# tokens = tokenize_en(speech)
# Split the data into training and testing sets
speech_train, speech_test = train_test_split(padded_speech, test_size=0.2)
ltl_train, ltl_test = train_test_split(one_hot_ltls, test_size=0.2)
# Convert the training and testing data into PyTorch tensors
# speech_train = torch.tensor(speech_train, dtype=torch.float32)
# speech_test = torch.tensor(speech_test, dtype=torch.float32)
# ltl_train = torch.tensor(ltl_train, dtype=torch.float32)
# ltl_test = torch.tensor(ltl_test, dtype=torch.float32)
print(f"Number of training examples: {len(speech_train)}")
print(f"Number of testing examples: {len(speech_test)}")
# DataLoader for the preprocessed data
# Create PyTorch Dataset and DataLoader for training data
train_dataset = LTL_Dataset(speech_train, ltl_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# Create PyTorch Dataset and DataLoader for testing data
test_dataset = LTL_Dataset(speech_test, ltl_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(speech_train.shape, ltl_train.shape)
print(speech_test.shape, ltl_test.shape)

#CNN seq2seq
#Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        #embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell
#Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        #embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        #prediction = [batch size, output dim]

        return prediction, hidden, cell
#Seq2seq test
class Seq2Seq_test(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        print("hidden",hidden.size())

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, trg_len):

            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
#Training the model
#Instantanizing the model
# num_epochs = 10
# input_dim = 1
# print("inp",input_dim)
# hidden_dim = 128
# output_dim = len(ltls_encoded)
# print("out",output_dim)
# model = Seq2Seq(input_dim, hidden_dim, output_dim).to(device)
# Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# learning_rate = 0.01
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
# for epoch in range(num_epochs):
#     model.train()
#     print("Entering training loop")
#     for i, (inputs, labels) in enumerate(train_loader):
#         inputs = inputs.to(device)
#         print(inputs.size())
#         labels = labels.to(device)
#         print(labels.size())
#         # Zero out the gradients
#         optimizer.zero_grad()
#         # Forward pass
#         outputs = model(inputs, labels)
#         # Loss function
#         loss = criterion(outputs.view(-1, output_dim), labels.argmax(dim=1))
#         # Backward pass
#         loss.backward()
#         optimizer.step()
#         # Training loss and accuracy
#         if (i+1) % 5 == 0:
#             with torch.no_grad():
#                 model.eval()
#                 outputs = model(speech_train.to(device), ltl_train.to(device))
#                 train_loss = criterion(outputs.view(-1, output_dim), ltl_train.argmax(dim=1))
#                 # train_acc = (outputs.argmax(dim=2) == ltl_train.argmax(dim=2)).float().mean()
#                 train_acc = (outputs.squeeze(1).argmax(dim=1) == ltl_train.argmax(dim=1)).float().mean()
#
#                 outputs = model(speech_test.to(device), ltl_test.to(device))
#                 test_loss = criterion(outputs.view(-1, output_dim), ltl_test.argmax(dim=1))
#                 # test_acc = (outputs.argmax(dim=2) == ltl_test.argmax(dim=2)).float().mean()
#                 test_acc = (outputs.squeeze(1).argmax(dim=1) == ltl_test.argmax(dim=1)).float().mean()
#
#                 print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(speech_train)}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# # Evaluate the model on the test data after training
# test_losses=[]
# with torch.no_grad():
#     model.eval()
#     outputs = model(speech_test.to(device), ltl_test.to(device))
#     print("Outputs",outputs.shape)
#     print("LTL",ltl_test.shape)
#     # test_loss = criterion(outputs.view(-1, output_dim), ltl_test.view(-1).to(device))
#     test_loss = criterion(outputs.view(-1, output_dim), ltl_test.view(-1, 198).to(device))
#     test_losses.append(test_loss.item())
#     print(f'Test Loss: {test_loss.item():.6f}')

# # Preprocess the speech input
# preprocessed_input = preprocess_speech('turn on right burner')
# preprocessed_input = torch.tensor(preprocessed_input, dtype=torch.float32)
# # Pass the preprocessed input through the trained model
# model.eval()
# with torch.no_grad():
#     output = model(preprocessed_input)
#
# print("first try:",output)
# print(output.size())
# Process the model output
# predicted_ltl = process_model_output(output)
#
# # Convert the output into LTL format
# ltl_string = convert_to_ltl_string(predicted_ltl)

# Print the generated LTL
# print("Generated LTL:", ltl_string)
INPUT_DIM = len(padded_speech)
print("input dim",INPUT_DIM)
OUTPUT_DIM = len(ltls_encoded)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq_test(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
def train(model, train_dataset, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, (inputs, labels) in enumerate(train_dataset):

        inputs = inputs.to(device)
        print(inputs.size())
        labels = labels.to(device)
        print(labels.size())
        optimizer.zero_grad()

        output = model(inputs,labels)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        labels = labels[1:].view(-1)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_dataset)

def evaluate(model, test_dataset, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(test_dataset):

            inputs = inputs.to(device)
            print(inputs.size())
            labels = labels.to(device)
            print(labels.size())

            output = model(inputs, labels, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            labels = labels[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, labels)

            epoch_loss += loss.item()

    return epoch_loss / len(test_dataset)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, test_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut1-model.pt'))

test_loss = evaluate(model, test_loader, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

#translation
# def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    #
    # model.eval()
    #
    # if isinstance(sentence, str):
    #     nlp = spacy.load('de_core_news_sm')
    #     tokens = [token.text.lower() for token in nlp(sentence)]
    # else:
    #     tokens = [token.lower() for token in sentence]
    #
    # tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    #
    # src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    #
    # src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    #
    # with torch.no_grad():
    #     encoder_conved, encoder_combined = model.encoder(src_tensor)
    #
    # trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    #
    # for i in range(max_len):
    #
    #     trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
    #
    #     with torch.no_grad():
    #         output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)
    #
    #     pred_token = output.argmax(2)[:,-1].item()
    #
    #     trg_indexes.append(pred_token)
    #
    #     if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
    #         break
    #
    # trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    #
    # return trg_tokens[1:], attention

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):

    model.eval()

    if isinstance(sentence, str):
        # Tokenize the input sentence manually instead of using Spacy
        tokens = sentence.strip().lower().split()
    else:
        tokens = [token.lower() for token in sentence]

    if isinstance(src_field, np.ndarray):
        src_init_token = "<sos>"
        src_eos_token = "<eos>"
        src_vocab = None
    else:
        src_init_token = src_field.init_token
        src_eos_token = src_field.eos_token
        src_vocab = src_field.vocab
    if isinstance(trg_field, np.ndarray):
        trg_init_index = 0  # Use the index of the first element as the initial index
        trg_vocab = None  # No vocabulary object for NumPy arrays
    else:
        trg_init_index = trg_field.vocab.stoi[trg_field.init_token]
        trg_vocab = trg_field.vocab

    # Convert tokens to indexes using the source field's vocabulary
    # src_indexes = [src_vocab[token] if src_vocab else token for token in tokens]
    src_indexes = [np.where(src_field == token)[0][0] if token in src_field else len(src_field) - 1 for token in tokens]
    src_indexes = np.array(src_indexes, dtype=np.int64)
    # Create a tensor from the source indexes and move it to the appropriate device
    src_tensor = torch.from_numpy(src_indexes).unsqueeze(0).to(device)

    with torch.no_grad():
        # Pass the source tensor through the encoder
        encoder_hidden, encoder_cell = model.encoder(src_tensor)

    # Initialize the target indexes with the start-of-sequence token
    # trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    trg_indexes = [trg_init_index]
    hidden = encoder_hidden[:, :1, :]
    cell = encoder_cell[:, :1, :]
    for _ in range(max_len):
        # Create a tensor from the target indexes and move it to the appropriate device
        # trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_tensor = torch.LongTensor(trg_indexes).to(device)

        with torch.no_grad():
            # Pass the target tensor and encoder hidden/cell states through the decoder
            output, hidden, cell = model.decoder(trg_tensor, encoder_hidden, encoder_cell)

        # Get the predicted token by selecting the one with the highest probability
        pred_token = output.argmax(2)[:, -1].item()

        # Append the predicted token to the target indexes
        trg_indexes.append(pred_token)
        if trg_vocab is None:
            # For NumPy arrays, break if the predicted token matches the last element
            if pred_token == trg_field[-1]:
                break
        else:
            # For TorchText Field, break if the predicted token matches the <eos> token
            if pred_token == trg_vocab.stoi[trg_field.eos_token]:
                break
    # Convert the target indexes to tokens using the target field's vocabulary
    # trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    trg_tokens = [trg_field.vocab.itos[i] if trg_vocab else trg_fdield[i] for i in trg_indexes]

    return trg_tokens[1:], None
#heat map
def display_attention(sentence, translation, attention):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(0).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'],
                       rotation=45)
    ax.set_yticklabels(['']+translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

#testing translation
example_idx = 2

src = ['turn','on','burner']
trg = ['X', '(', 'G', '(','C','_','StovePot_Knob', 'U', 'StovePot_Knob',')', '&', 'G', '(','StovePot_Knob', 'U', 'Stove_On',')', ')']

print(f'src = {src}')
print(f'trg = {trg}')

translation, attention = translate_sentence(src, speech, ltls_encoded, model, device)

print(f'predicted trg = {translation}')