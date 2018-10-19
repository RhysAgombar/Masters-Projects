import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import numpy as np
import pickle
from io import open

def procData(path):
    file = open(path)

    data = ""
    count = 1
    for line in file.readlines():
        data = data + (line).lower()
        count = count + 1
        if count % 200000 == 0:
            break

    data = re.sub("\\t","", data)
    data = re.sub("\\n","", data)
    data = re.sub("  ", " ", data)
    data = data.split(" . ")

    holder = []
    for d in data:
        if d != "":
            holder.append(d)
    data = holder

    for i in range(len(data)):
        data[i] = data[i] + " . "

    o1 = set(''.join(data).split(" "))

    for j in range(len(data)):
        data[j] = data[j].split(" ")

        holder = []
        for i in range(len(data[j])):
            if data[j][i] != "":
                holder.append(data[j][i])
        data[j] = holder

    return data, o1

path = "recipe_tokenized_instructions_.txt"
data, vocab = procData(path)

quatragrams = []
for j in range(len(data)):
    padded = data[j][:]
    for k in range(3):
        padded.insert(0,"")
    for i in range(len(padded)-3):
        quatragrams.append([[padded[i],padded[i+1],padded[i+2]], padded[i+3]])

CONTEXT_SIZE = 3
EMBEDDING_DIM = 50
word_indices = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def getEmb(self):
        return self.embeddings

    def getProb(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(1000):
    total_loss = 0
    for context, target in quatragrams:

        context_idxs = torch.tensor([word_indices[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([word_indices[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(round(total_loss, 3))

emb = [model.embeddings(torch.tensor(word_indices[w])) for w in vocab]

count = 0.0
for i in word_indices:
    em = model.embeddings(torch.tensor([word_indices[i]]))
    t = model.getEmb()
    normalized_embedding = t.weight/((t.weight**2).sum(0)**0.5).expand_as(t.weight)
    dist, ind = torch.topk(torch.mv(normalized_embedding,em[0]),5)

    estimation = int(ind[0])
    target = word_indices[i]

    if (estimation == target):
        count += 1.0


emFile = open("Em.pickle","wb")
pickle.dump(emb, emFile)
emFile.close()

wiFile = open("WI.pickle","wb")
pickle.dump(word_indices, wiFile)
wiFile.close()

gramFile = open("QG.pickle","wb")
pickle.dump(quatragrams, gramFile)
gramFile.close()

vocabFile = open("Vocab.pickle","wb")
pickle.dump(vocab, vocabFile)
vocabFile.close()
