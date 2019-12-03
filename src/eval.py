import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import torch
import utils 
from model import RNNModel
from learner import Learner
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
              'num_users': 5421,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': False,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 30,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': False,
              'device_id': 7,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch': 200,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'use_cuda': True,
                'device_id': 7,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

# Load Data
ml1m_dir = 'data/ml-1m/ratingTraining.dat' #ratings.dat
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
config = gmf_config
engine = GMFEngine(config)
e_model = "gmf"
#config = mlp_config
#engine = MLPEngine(config)
#state_dict = torch.load( 'checkpoints/gmf_factor8neg4-implict_Epoch199_HR0.6363_NDCG0.3678.model')
#engine.model.load_state_dict(state_dict)
#utils.resume_checkpoint(engine.model,, config['device_id'])
#engine.model = torch.load('checkpoints/gmf_factor8neg4-implict_Epoch199_HR0.6363_NDCG0.3678.model')
#hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
#engine.save(config['alias'], 0, hit_ratio, ndcg)

# config = neumf_config
# engine = NeuMFEngine(config)
ntokens = 3953
nlayers = 2
hidden_size = 500
dropout = 0.5
train_batch_size = 200    
print("Data Generated")
#model = RNNModel('gru', ntokens, input_size, hidden_size, nlayers, dropout)
#model.load_state_dict(torch.load("sequence_modelling/model.pt"))
#model.eval()
#criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=5)
#optimizer = optim.SGD(model.parameters(), lr=20)
#generator = Learner(model, criterion, optimizer)        
#df = sample_generator.train_ratings
#sequence_distribution = {}
#for i,d in df.groupby('user'):
#    sequence_distribution[d.iloc[0]['user']] = learner.generate_dist_from_subsequence(d['item'].to_list(),movie_embeddings)        
state_dict = torch.load("checkpoints/gmf_factor8neg4-implict_Epoch69_HR0.5412_NDCG0.3654.model")
engine.model.load_state_dict(state_dict)
print("Model Loaded")
sequence_distribution = {}
for i in range(6040):
    x  = {}
    y = np.random.uniform(1,0,3707)
    for j in range(y.size):
        x[j] = y[j]
    sequence_distribution[i] = x   
print("Dummy Data created")
#sequence_distribution = None
hit,ndcg,gbhit = engine.combine_evaluate(evaluate_data,sequence_distribution)
