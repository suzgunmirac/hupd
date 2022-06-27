# Importing relevant libraries and dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression (nn.Module):
    """ Simple logistic regression model """

    def __init__ (self, vocab_size, embed_dim, n_classes, pad_idx):
        super (LogisticRegression, self).__init__ ()
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim, 
            padding_idx = pad_idx)
        # Linear layer
        self.fc = nn.Linear (embed_dim, n_classes)
        
    def forward (self, input_ids):
        # Apply the embedding layer
        embed = self.embedding(input_ids)
        # Apply the linear layer
        output = self.fc (embed)
        # Take the sum of the overeall word embeddings for each sentence
        output = output.sum (dim=1)
        return output


class BasicCNNModel (nn.Module):
    """ Simple 2D-CNN model """
    def __init__(self, vocab_size, embed_dim, n_classes, n_filters, filter_sizes, dropout, pad_idx):
        super(BasicCNNModel, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim, 
            padding_idx = pad_idx)
        # Conv layer
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1, 
                out_channels = n_filters, 
                kernel_size = (fs, embed_dim)) 
             for fs in filter_sizes])
        # Linear layer
        self.fc = nn.Linear(
            len(filter_sizes) * n_filters, 
            n_classes)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        # embed = [batch size, sent len, emb dim]
        embed = embed.unsqueeze(1)
        # embed = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embed)).squeeze(3) for conv in self.convs]    
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        output = self.fc(cat) #.sigmoid ().squeeze()
        return output


class BigCNNModel (nn.Module):
    """ Slightly more sophisticated 2D-CNN model """
    def __init__(self, vocab_size, embed_dim, pad_idx, n_classes, n_filters=25, filter_sizes=[3,4,5], dropout=0.25):
        super(BigCNNModel, self).__init__()
        print(f'filter_sizes: {filter_sizes}')
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim, 
            padding_idx = pad_idx)
        # Conv layers
        self.convs_v1 = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1, 
                out_channels = n_filters, 
                kernel_size = (fs, embed_dim)) 
             for fs in filter_sizes[0]])
        self.convs_v2 = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1, 
                out_channels = n_filters, 
                kernel_size = (fs, embed_dim)) 
             for fs in filter_sizes[1]])
        self.convs_v3 = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1, 
                out_channels = n_filters, 
                kernel_size = (fs, embed_dim)) 
             for fs in filter_sizes[2]])
        # Linear layer
        self.fc = nn.Linear(
            len(filter_sizes) * n_filters * 3, 
            n_classes)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        # embed = [batch size, sent len, emb dim]
        embed = embed.unsqueeze(1)
        # embed = [batch size, 1, sent len, emb dim]
        conved_v1 = [F.relu(conv(embed)).squeeze(3) for conv in self.convs_v1]
        conved_v2 = [F.relu(conv(embed)).squeeze(3) for conv in self.convs_v2]
        conved_v3 = [F.relu(conv(embed)).squeeze(3) for conv in self.convs_v3]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled_v1 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_v1]
        pooled_v2 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_v2]
        pooled_v3 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_v3]
        # pooled_n = [batch size, n_filters]
        cat_v1 = self.dropout(torch.cat(pooled_v1, dim = 1))
        cat_v2 = self.dropout(torch.cat(pooled_v2, dim = 1))
        cat_v3 = self.dropout(torch.cat(pooled_v3, dim = 1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        cat = torch.cat([cat_v1, cat_v2, cat_v3], dim=1)
        output = self.fc(cat) #.sigmoid ().squeeze()
        return output