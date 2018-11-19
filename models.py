import torch
from torch import nn
from torch.nn import Sequential
from torchvision import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchsummary import summary

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        pretrained_model = models.resnet34(pretrained=True)
        self.resnet = Sequential(*list(pretrained_model.children())[:-2])
        self.batchnorm = nn.BatchNorm2d(num_features=512,momentum=0.01)

    def forward(self, x):
        x = self.resnet(x)
        x = self.batchnorm(x)
        x = torch.reshape(x,(-1,49,512))
        return x

class AttentionModel(nn.Module):
    def __init__(self,hidden_dim):
        super(AttentionModel, self).__init__()
        self.hidden_to_input = nn.Linear(hidden_dim,512)
        self.inputs_to_e = nn.Linear(512,1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,encoded_image,hidden_input):
        hidden_inp = self.hidden_to_input(hidden_input).unsqueeze(1)
        sum_of_inp = self.relu(encoded_image + hidden_inp)
        e = self.inputs_to_e(sum_of_inp).squeeze(2)
        alpha = self.softmax(e)

        return alpha

class DecoderRNN(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocabulary_size):
        super(DecoderRNN,self).__init__()
        self.embedding_matrix = nn.Embedding(vocabulary_size,embedding_dim)
        self.lstm_cell = nn.LSTMCell(embedding_dim+512,hidden_dim,bias=True)
        self.L_o = nn.Linear(embedding_dim,vocabulary_size)
        self.L_h = nn.Linear(hidden_dim,embedding_dim)
        self.L_z = nn.Linear(512,embedding_dim)
        self.init_c = nn.Linear(512,hidden_dim)
        self.init_h = nn.Linear(512,hidden_dim)
        self.beta = nn.Linear(hidden_dim,512)
        self.sigmoid = nn.Sigmoid()

        self.attention = AttentionModel(hidden_dim)

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size
        self.init_weights()

    def init_weights(self):
        self.embedding_matrix.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_variable(self,encoded_image):
        mean_encoded_image = encoded_image.mean(1)
        h = self.init_h(mean_encoded_image)
        c = self.init_c(mean_encoded_image)
        return h,c

    def forward(self,features, captions, lengths):
        h, c = self.init_hidden_variable(features)
        embeddings = self.embedding_matrix(captions)
        outputs = torch.zeros(features.size(0), max(lengths), self.vocabulary_size)
        alphas = torch.zeros(features.size(0), max(lengths), 49)


        zeros = torch.zeros(features.size(0),dtype=torch.long,device='cuda:0')
        alpha = self.attention(features,h)
        encoding_with_attention = (features * alpha.unsqueeze(2)).sum(dim=1)
        gating_scalar = self.sigmoid(self.beta(h))
        encoding_with_attention = encoding_with_attention*gating_scalar
        h,c = self.lstm_cell(torch.cat([self.embedding_matrix(zeros),encoding_with_attention],dim=1),(h,c))
        output = self.L_o(self.embedding_matrix(zeros)+self.L_h(h)+self.L_z(encoding_with_attention))
        outputs[:, 0, :] = output
        alphas[:, 0, :] = alpha

        for t in range(1,max(lengths)):
            batch_size = sum([l > t for l in lengths])
            alpha = self.attention(features[:batch_size],h[:batch_size])
            encoding_with_attention = (features[:batch_size] * alpha.unsqueeze(2)).sum(dim=1)
            gating_scalar = self.sigmoid(self.beta(h[:batch_size]))
            encoding_with_attention = encoding_with_attention*gating_scalar
            h,c = self.lstm_cell(torch.cat([embeddings[:batch_size,t-1,:],encoding_with_attention],dim=1),(h[:batch_size],c[:batch_size]))
            output = self.L_o(embeddings[:batch_size,t-1,:]+self.L_h(h[:batch_size])+self.L_z(encoding_with_attention))
            outputs[:batch_size, t, :] = output
            alphas[:batch_size, t, :] = alpha

        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        return outputs,alphas

    def sample(self,features,max_length=25):
        h, c = self.init_hidden_variable(features)
        outputs = torch.zeros(features.size(0), max_length)
        output = torch.zeros(features.size(0),dtype=torch.long,device='cuda:0')
        embeddings = self.embedding_matrix

        for t in range(max_length):
            alpha = self.attention(features,h)
            encoding_with_attention = (features * alpha.unsqueeze(2)).sum(dim=1)
            gating_scalar = self.sigmoid(self.beta(h))
            encoding_with_attention = encoding_with_attention*gating_scalar
            h,c = self.lstm_cell(torch.cat([embeddings(output),encoding_with_attention],dim=1),(h,c))
            output = self.L_o(embeddings(output)+self.L_h(h)+self.L_z(encoding_with_attention))
            output = output.argmax(dim=1)
            outputs[:,t] = output

        return outputs
