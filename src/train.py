import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import torch

import utils 
from sequence_modeling.model import RNNModel
from sequence_modeling.learner import Learner
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sequence_modeling.data_preparation import MovieEncodingEmbedding

og_glove_file = r"/Users/anshuman/UMass/CS 682/Project/glove.6B/glove.6b.300d.txt" ## glove text file that you download from internet, 6b words 300d,
new_glove_file =r"glove6b300d.bin"
# glove_file = datapath(og_glove_file)
# tmp_file = get_tmpfile("glove.txt")
# _ = glove2word2vec(glove_file, tmp_file)
# model = KeyedVectors.load_word2vec_format(tmp_file)
# model.save_word2vec_format(new_glove_file,binary=True)

glove_model = gensim.models.KeyedVectors.load_word2vec_format(new_glove_file,binary=True)
##get vocabulary
vocab = list(glove_model.wv.vocab.keys())

movieData = pd.read_csv('data/ml-1m/movies.dat',sep="::",names=["MovieID","Name","Genres"],engine='python')
print(movieData.shape)
numRows = movieData.shape[0]
# print(movieData, movieData.loc[0][0], movieData.loc[0][1], movieData.loc[0][2])
movie_embeddings = np.zeros((numRows, 319))

genreList = ["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir",
              "Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

# Prepare movie embeddings -> A Mapping from MovieID to its embedding. This can then be used during model training
for rowNum in range(numRows):
    movieID = movieData.loc[rowNum]['MovieID']
#     print(rowNum, movieID)
    movie_embeddings[rowNum] = MovieEncodingEmbedding(movieID, movieData, genreList, glove_model, vocab)

print("Movie Embeddings", movie_embeddings.shape)


gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch':70,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': False,
              'device_id': 7,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 20,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              # 'layers': [326, 64, 32, 8],
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': False,
              'device_id': 7,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict_Epoch69_HR0.5460_NDCG0.3681.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
              'item_emb':movie_embeddings}

neumf_config = {'alias': 'baseline_neumf_alldata',
                'num_epoch': 100,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                # 'layers': [327, 64, 32],
                'l2_regularization': 0.0000001,
                'use_cuda': False,
                'device_id': 7,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict_Epoch69_HR0.5460_NDCG0.3681.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_emb_pretrain_reg_0.0000001_64_32_Epoch19_HR0.5510_NDCG0.3892.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
                'item_emb':movie_embeddings
                }

# Load Data
ml1m_dir = 'data/ml-1m/ratings.dat' #ratings.dat
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

# Reindex
ml1m_uwf = 'data/ml-1m/unwatchedFilms.dat'
uwflist = []
with open(ml1m_uwf) as f: uwflist =[int(i[:-1]) for i in f.readlines()]        
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
# DataLoader for training
#learner = Learner()
#
#
sample_generator = SampleGenerator(ratings=ml1m_rating,uwf_list=uwflist)
evaluate_data = sample_generator.evaluate_data

# Specify the exact model
# config = gmf_config
# engine = GMFEngine(config)
# config = mlp_config
# engine = MLPEngine(config)

#state_dict = torch.load( 'checkpoints/gmf_factor8neg4-implict_Epoch199_HR0.6363_NDCG0.3678.model')
#engine.model.load_state_dict(state_dict)
#utils.resume_checkpoint(engine.model,, config['device_id'])
#engine.model = torch.load('checkpoints/gmf_factor8neg4-implict_Epoch199_HR0.6363_NDCG0.3678.model')
#hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
#engine.save(config['alias'], 0, hit_ratio, ndcg)


# config = neumf_config
# engine = NeuMFEngine(config)
loss = None

config = neumf_config
engine = NeuMFEngine(config)

for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    c_loss = engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio, ndcg)
    # if not loss or val_loss < loss:
    #     utils.save_checkpoint(engine.model,"checkpoints/best_model_collab_filtering_"+emodel+"_"+str(epoch)+".pt")
    #     best_val_loss = c_loss