from multiprocessing.connection import Listener
from multiprocessing.context import AuthenticationError
import numpy as np
import torch
from torch import nn
from transformers import *

# Gets position embedding matrix
def position_embeddings(max_pos, size):
    embeddings = np.zeros((max_pos, size))
    w = 1 / (10000 ** (2*np.arange(size // 2 )/size))
    for pos in range(max_pos):
        embeddings[pos,0::2] = np.sin(w*pos)
        embeddings[pos,1::2] = np.cos(w*pos)
    return torch.Tensor(embeddings)

# Split a tokenized string into 512-token chunks
# Chunks overlap so first starts at token 0, second at 256, etc
def split(tokenized):
    sections = []
    positions = []
    start = 0
    while start + 512 < len(tokenized):
        sections.append(np.array(tokenized[start:start+512]))
        positions.append(list(range(start, start+512)))
        start += 256

    # Pad the last section with zeros
    last_section = tokenized[start:]
    sections.append(last_section + [0]*(512-len(last_section)))
    positions.append(list(range(start, start+512)))

  
    return np.array(sections), positions

# The classification model that runs on top of BERT
class Classifier(nn.Module):
    def __init__(self, bert_size, position_size):
        super().__init__()

        # Gets attention value
        self.attention = nn.Linear(bert_size + position_size, 1)
        self.softmax = nn.Softmax(1)

        # Predicts labels
        self.prediction = nn.Sequential(
            nn.Linear(bert_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 6),
            nn.Sigmoid()
        )

    '''
    embeddings: shape (segment count, 512, bert_hidden_size)
    position_encodings:  shape (segment count, 512, position_encoding_size)
    comment_bounds: Array of tuples of the form [(start, end)]. comment_bounds[i] = (a, b) indicates that comment i's embeddings can be extracted as embeddings[a:b]
    '''
    def forward(self, embeddings, position_encodings, comment_bounds = None):
        attention_input = torch.cat([embeddings, position_encodings], dim=2) # (batch, 512, position_size + bert_hidden_size)

        # (batch, 512, 1)
        attentions = self.attention(attention_input)
        if comment_bounds is None:
            attentions = self.softmax(attentions) # (batch, 512, 1)
            vecs = torch.sum(attentions * embeddings, dim=1) # (batch, bert_hidden_size)
            return self.prediction(vecs) # (batch, 1)

        vecs = []
        for (a,b) in comment_bounds:
            comment_embeddings = embeddings[a:b] # (segment_count, 512, bert_hidden_size)
            comment_attentions = attentions[a:b] # (segment_count, 512, 1)
            attention_weights = self.softmax(comment_attentions) # (segment_count, 512, 1)
            weighted_embeddings = attention_weights * embeddings[a:b] # (segment_count, 512, bert_hidden_size)
            vec = torch.sum(weighted_embeddings.view(-1, weighted_embeddings.shape[-1]), dim=0, keepdim=True) # (segment_count, bert_hidden_size)
            vecs.append(vec)
        return self.prediction(torch.cat(vecs))

def predict(comment, tokenizer, pos_encodings, bert, model, device):
    encoded = tokenizer.encode(comment, add_special_tokens=True, max_length=10000)

    split_comment, position_arr = split(encoded)
    input_ids = torch.LongTensor(split_comment)
    flat_positions = torch.LongTensor(position_arr)

    comment_id = torch.LongTensor([0]*split_comment.shape[0])

    encoded_positions = pos_encodings[flat_positions]

    comment_bounds = [(0, len(split_comment))]

    with torch.no_grad():
        encoded_comments = bert(input_ids.to(device))[0]
        output = model(encoded_comments, encoded_positions.to(device), comment_bounds)
    
    return output[0].numpy()

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    
    position_encoding_size = 256
    pos_embed = position_embeddings(10000, position_encoding_size)

    bert_hidden_size = 768
    bert = BertModel.from_pretrained('bert-base-cased')

    model_path = "./model/model_epoch_{:04d}_batch_{:04d}_bce_{:.04f}.pt".format(29, 100, 0.0629)
    model = Classifier(bert_hidden_size, position_encoding_size)
    model.load_state_dict(torch.load(model_path))

    bert.eval()
    model.eval()
    device = torch.device("cpu")
    bert = bert.to(device)
    model = model.to(device)
    
    address = ("localhost", 6000)
    listener = Listener(address, authkey=b"274p_server")
    while True:
        try:
            with listener.accept() as conn:
                print("Connection Accepted From {}".format(listener.last_accepted))
                msg = conn.recv()
                prediction = predict(msg, tokenizer, pos_embed, bert, model, device)
                conn.send(prediction)
        except AuthenticationError:
            print("Connection Rejected")
 
