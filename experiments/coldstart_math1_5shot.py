# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch.utils.data as Data
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
import progressbar
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import math
import scipy
import matplotlib.pyplot as plt
import collections
# %matplotlib inline
import logging

USE_CUDA = torch.cuda.is_available()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if USE_CUDA:
    torch.cuda.manual_seed(1)


seed = 1
threshold = 0.5
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
gpu_available = torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#hyper parameters
NUM_EPOCHS = 100
MAX_RECORD_SIZE = 1e6
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 32 #knowledge_embedding_size, dimention of knowledge space

logging.info(gpu_available)

# ! /opt/bin/nvidia-smi

def load_data(path) -> dict:
    full_data = pd.read_csv(path + 'data.txt', header=None, sep='\t').values.astype(np.int64) # CrossEntropy这里要改城int64
    knowledge_matrix = pd.read_csv(path + 'q.txt', header=None, sep='\t').values.astype(np.float32)
    students_num, items_num, skills_num = full_data.shape[0], full_data.shape[1], knowledge_matrix.shape[1]
    full_data = np.array([{'stu_id': stu_id, 'item_id': item_id, 'score': full_data[stu_id, item_id], 'knowledge': knowledge_matrix[item_id]}
          for stu_id in range(students_num) for item_id in range(items_num)])
    
    np.random.shuffle(full_data)
    
    return {'full_data': full_data, 'students_num': students_num, 'items_num': items_num, 'skills_num':  skills_num}

def load_old_new_data(path, ratio) -> dict:
    full_data = pd.read_csv(path + 'data.txt', header=None, sep='\t').values.astype(np.int64)
    knowledge_matrix = pd.read_csv(path + 'q.txt', header=None, sep='\t').values.astype(np.float32)
    students_num, items_num, skills_num = full_data.shape[0], full_data.shape[1], knowledge_matrix.shape[1]
    data = np.array([{'stu_id': stu_id, 'item_id': item_id, 'score': full_data[stu_id, item_id], 'knowledge': knowledge_matrix[item_id]}
          for stu_id in range(students_num) for item_id in range(items_num)])
    
    np.random.shuffle(data)
    old_data = data[: int(len(data) * ratio)]
    new_data = data[int(len(data) * ratio): ]
    return {'old_data': old_data, 'new_data': new_data}

def split_data(data, ratio):
    train = data[: int(len(data) * ratio)]
    valid = data[int(len(data) * ratio): ]
    return train, valid

def split_new_data(data, ratio):
    mini_batch1 = data[: int(len(data) * ratio)]
    mini_batch2 = data[int(len(data) * ratio): int(len(data) * ratio) * 2]
    mini_batch3 = data[int(len(data) * ratio) * 2 : int(len(data) * ratio) * 3]
    test = data[int(len(data) * 3 * ratio): ]
    return mini_batch1, mini_batch2, mini_batch3, test

'''
ratio1: [old:new]
ratio2: [train:val]
ratio3: [3*mini_batch:test] < 0.3
'''
def preprocess_data(path, ratio1, ratio2, ratio3):
    data = load_old_new_data(path=path, ratio=ratio1)
    old_data, new_data = data['old_data'], data['new_data']
    # old -> train, valid
    old_train, old_valid = split_data(old_data, ratio2)
    
    # new -> 3*mini-batch, test
    mini_batch1, mini_batch2, mini_batch3, test = split_new_data(new_data, ratio3)
    
    
    return {'old_train': old_train, 'old_valid': new_data, 'mini_batch1': mini_batch1, \
            'mini_batch2': mini_batch2, 'mini_batch3': mini_batch3, 'test': test}

path = './Math1/'
ratio1, ratio2, ratio3 = 0.3, 0.6, 0.25
data = preprocess_data(path, ratio1, ratio2, ratio3)

logging.info(data['old_train'].shape, data['old_valid'].shape, data['mini_batch1'].shape, data['mini_batch2'].shape, data['mini_batch3'].shape, data['test'].shape)

full_data = load_data(path)
student_n, item_n, knowledge_n, knowledge_embed_size = \
full_data['students_num'], full_data['items_num'], full_data['skills_num'], EMBEDDING_SIZE

class MyDataset(Data.Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__() 
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['stu_id'], self.data[idx]['item_id'], self.data[idx]['knowledge'], self.data[idx]['score']

train_dataset = MyDataset(data['old_train'])
dataloader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

for batch_stu_id, batch_item_id, batch_knowledge_id, batch_label in dataloader:
    logging.info(batch_stu_id.dtype, batch_item_id.dtype, batch_label.dtype, batch_label.dtype)
    break

class AttentionLayer(nn.Module):
    
    def __init__(self, student_n, item_n, knowledge_n, knowledge_embed_size, n_heads=8):
        
        super(AttentionLayer, self).__init__()
        
        self.student_n = student_n
        self.item_n = item_n
        self.knowledge_n = knowledge_n
        self.knowledge_embed_size = knowledge_embed_size
        self.n_heads = n_heads
        self.d_model = self.knowledge_embed_size
        
        self.emb_stu = nn.Embedding(student_n, knowledge_embed_size) # Q
        self.emb_item = nn.Embedding(item_n, knowledge_embed_size) # K
        self.emb_knowledge = nn.Linear(knowledge_n, knowledge_embed_size) # V
        
        self.W_stu_knowledge = nn.Linear(self.d_model, knowledge_embed_size * self.n_heads, bias=False)
        
        self.W_item_knowledge = nn.Linear(self.d_model, knowledge_embed_size * self.n_heads, bias=False)
        
        self.W_skill_knowledge = nn.Linear(self.d_model, knowledge_embed_size * self.n_heads, bias=False)
        
#         self.similar = nn.CosineSimilarity(dim=0, eps=1e-6)
                
        self.softmax = nn.Softmax(dim=0)
        
        self.drop = nn.Dropout(p=0.5)
        
                # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        
    def forward(self, batch_stu_id, batch_item_id, batch_knowledge_id):
        
        # three embedding representation in paper: [batch_size, knowledge_embed_size * n_heads]
        embed_stu = torch.sigmoid(self.emb_stu(batch_stu_id))   
        embed_item = torch.sigmoid(self.emb_item(batch_item_id))     
        embed_knowledge = torch.sigmoid(self.emb_knowledge(batch_knowledge_id)) 
        
        # three relation attention in paper: [batch_size, knowledge_embed_size * n_heads]
        stu_knowledge_attention = self.W_stu_knowledge(embed_stu)
        item_knowledge_attention = self.W_item_knowledge(embed_item)
        skill_knowledge_attention = self.W_skill_knowledge(embed_knowledge)
        
        
        attention_score = (stu_knowledge_attention * item_knowledge_attention) / np.sqrt(self.knowledge_embed_size)\
                          * skill_knowledge_attention
        
        return attention_score


class ACDM(nn.Module):
    
    def __init__(self, student_n, item_n, knowledge_n, knowledge_embed_size, n_heads=8):
        
        super(ACDM, self).__init__()
        
        self.student_n = student_n
        self.item_n = item_n
        self.knowledge_n = knowledge_n
        self.knowledge_embed_size = knowledge_embed_size
        self.n_heads = n_heads
        
        self.muti_attention = AttentionLayer(student_n, item_n, knowledge_n, knowledge_embed_size)
        
        self.similar = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.softmax = nn.Softmax(dim=0)
        
        self.linear1 = nn.Linear(self.knowledge_embed_size * self.n_heads, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)
        
        self.drop = nn.Dropout(p=0.5)
        
                # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        
    def forward(self, batch_stu_id, batch_item_id, batch_knowledge_id, knowledge_n):
        
        attention_score = self.muti_attention(batch_stu_id, batch_item_id, batch_knowledge_id)
        # [batch_size, ]
        hidden1 = self.drop(torch.sigmoid(self.linear1(attention_score))) 
        hidden2 = self.drop(torch.sigmoid(self.linear2(hidden1))) 
        out = torch.sigmoid(self.linear3(hidden2))
        out = out
        
        return out
    
class GateLayer(nn.Module):
    def __init__(self, feature_size, num_layers, f=torch.relu):

        super(GateLayer, self).__init__()

        self.num_layers = num_layers

        self.guess = nn.ModuleList([nn.Linear(feature_size, feature_size) for _ in range(num_layers)])

        self.slip = nn.ModuleList([nn.Linear(feature_size, feature_size) for _ in range(num_layers)])

        self.pass_func = nn.ModuleList([nn.Linear(feature_size, feature_size) for _ in range(num_layers)])

        self.nopass_func = nn.ModuleList([nn.Linear(feature_size, feature_size) for _ in range(num_layers)])

        self.f = f
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            guess_prob = torch.sigmoid(self.guess[layer](x))
            slip_prob = torch.sigmoid(self.slip[layer](x))
            gate = guess_prob + slip_prob

            pass_results = self.f(self.pass_func[layer](x)) # f only functinoal on the pass
            no_pass_results = self.nopass_func[layer](x)

            x = pass_results + gate * no_pass_results

        return x

class AGCDM(nn.Module):
    def __init__(self, student_n, item_n, knowledge_n, knowledge_embed_size, n_heads=8):
        super(AGCDM, self).__init__()
        
        self.n_heads = n_heads
        self.attention = AttentionLayer(student_n, item_n, knowledge_n, knowledge_embed_size)
        self.gate = GateLayer(knowledge_embed_size * self.n_heads, 1, torch.sigmoid)
        
        self.linear = nn.Linear(knowledge_embed_size * self.n_heads, 1)
        
    def forward(self, batch_stu_id, batch_item_id, batch_knowledge_id, knowledge_n):
        
        attention_score = self.attention(batch_stu_id, batch_item_id, batch_knowledge_id)
        gate_score = self.gate(attention_score)
        score = self.linear(gate_score)
        return score
    

class MetaLearner(object):
    
    def __init__(self, model_type, data, \
                 student_n, item_n, knowledge_n, loss_func, \
                 knowledge_embed_size=EMBEDDING_SIZE, epoch_size=NUM_EPOCHS, \
                 batch_size=BATCH_SIZE, learning_rate = LEARNING_RATE, gpu_available = True):
        
        super(MetaLearner, self).__init__()
        
        self.data = data
        self.student_n = student_n
        self.item_n = item_n
        self.knowledge_n = knowledge_n
        self.knowledge_embed_size = knowledge_embed_size
        
        self.train_epochs = epoch_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.gpu_available = gpu_available

        self.model = self.new_model()


        # gpu
        # if self.gpu_available:
        #     self.model = self.model.to(device)
            
        self.loss_func = loss_func
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # meta-leaner hyperparameters
        self.meta = False
        self.num_tasks = 20
        self.num_shot = 5
        self.task_epochs = 10
        self.alpha = 1e-3
        self.beta = 1e-3
        self.lam = 1e-3

        
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.results = self.new_results()

    def new_model(self):
        if self.model_type == 'AGCDM':
            model = AGCDM(student_n, item_n, knowledge_n, knowledge_embed_size)
        elif self.model_type == 'ACDM':
            model = ACDM(student_n, item_n, knowledge_n, knowledge_embed_size)
        else:
            raise ValueError('No models')

        if self.gpu_available:
            model = model.to(device)
        return model

    def new_results(self):
        results = {}
        results['batch1'], results['batch2'], results['batch3'] = {}, {}, {}
        for key in results.keys():
            results[key]['rmse'], results[key]['acc'], results[key]['recall'], results[key]['f1'], results[key]['auc'] \
            = [], [], [], [], []
        return results
    
    def update_results(self, rmse, acc, recall, f1, auc, batch_name):
        self.results[batch_name]['rmse'].append(rmse)
        self.results[batch_name]['acc'].append(acc)
        self.results[batch_name]['recall'].append(recall)
        self.results[batch_name]['f1'].append(f1)
        self.results[batch_name]['auc'].append(auc)
        
    def sample_task_data(self, data):#mate过程中用到的随机抽样，抽出一个小task去fit数据集
        dataloader = Data.DataLoader(MyDataset(data), batch_size=self.num_shot, shuffle=True, num_workers=0) # dataloader本身有shuffle后的sample功能
        task_data = next(iter(dataloader))
        return task_data
    
    def show_params_grad(self):
        for params in self.model.parameters():
            logging.info(params.grad)
            break
        
        
    def train_task(self, task_data):
        stu, item, knowledge, label = task_data[0], task_data[1], task_data[2], task_data[3]
        if self.gpu_available:
            stu, item, knowledge, label = \
            stu.to(device), item.to(device), knowledge.to(device), label.to(device)
        self.optimizer.zero_grad()
        output_1 = self.model(stu, item, knowledge, knowledge_n)
        output_0 = torch.ones(output_1.size()).to(device) - output_1
        #logging.info(output_1.shape, output_0.shape)
        out = torch.cat((output_0, output_1), 1)
        loss_task = self.loss_func(out, label)
        loss_task.backward() 
        self.optimizer.step() 
        
    def reset_model(self):
        self.model = self.new_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.meta = False
        del self.train_losses[:]
        del self.val_losses[:]
        del self.test_losses[:]
        self.results = self.new_results()
        
    def learn_algorithm(self):
        
        logging.info("Learning an algorithm for warm up cold-start....")
        self.meta = True
        
        for e in range(self.task_epochs):        
            
            self.opti_params_ = []

            #1. for train task i in batch of tasks
            for i in range(self.num_tasks):
                
                task_data = self.sample_task_data(self.data['old_train'])
    
                self.train_task(task_data)
                
                opti_params = deepcopy(self.model.state_dict())
                
                self.opti_params_.append(opti_params)
            
                
            meta_grad_dict = deepcopy(self.model.state_dict())
            meta_grad_dict = {name: nn.init.constant_(meta_grad_dict[name], 0.) for name in meta_grad_dict} 
            
            
            #2. Add each tasks loss, backprogate to get a "fitness" parameters
            for i in range(self.num_tasks):
                
                task_data = self.sample_task_data(self.data['old_train'])
                stu, item, knowledge, label = task_data[0], task_data[1], task_data[2], task_data[3]

                if self.gpu_available:
                    stu, item, knowledge, label = \
                    stu.to(device), item.to(device), knowledge.to(device), label.to(device)
                
                net_optim = self.new_model()
                
                net_optim.load_state_dict(self.opti_params_[i])
                
                output_1 = net_optim(stu, item, knowledge, knowledge_n)
                output_0 = torch.ones(output_1.size()).to(device) - output_1
                out = torch.cat((output_0, output_1), 1)
                
                loss = self.loss_func(out, label)
                
                loss.backward()
                
                #update meta gradient bt net_optim_params's grad
                net_optim_params_grad = {}
                for name, params in zip(net_optim.state_dict(), net_optim.parameters()):
                    net_optim_params_grad[name] = params.grad.data
                #logging.info(net_optim_params_grad)
                meta_grad_dict = {name: meta_grad_dict[name] + net_optim_params_grad[name] / self.num_shot for name in meta_grad_dict} 
                #meta_grad_dict = {name: meta_grad_dict[name] + net_optim_params[name].grad.data / self.num_samples for name in meta_grad_dict} 
            
            
            #update net params by meta gradient
            net_params = self.model.state_dict()
            net_params_new = {name: net_params[name] + self.beta * meta_grad_dict[name] / self.num_shot for name in net_params} 
            self.model.load_state_dict(net_params_new)
    
    
    def evaluate(self, data):
        self.model.eval() # 抽离
        error = 0.
        with torch.no_grad():
            dataset = MyDataset(data)
            dataloader = Data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            for batch_stu_id, batch_item_id, batch_knowledge_id, batch_label in dataloader:
                # gpu
                if self.gpu_available:
                    batch_stu_id, batch_item_id, batch_knowledge_id, batch_label = \
                    batch_stu_id.to(device), batch_item_id.to(device), batch_knowledge_id.to(device), batch_label.to(device)
                
                #predict = self.model(batch_stu_id, batch_item_id, batch_knowledge_id, knowledge_n)
                output_1 = self.model(batch_stu_id, batch_item_id, batch_knowledge_id, knowledge_n)
                output_0 = torch.ones(output_1.size()).to(device) - output_1
                #logging.info(output_1.shape, output_0.shape)
                batch_out = torch.cat((output_0, output_1), 1)
                batch_error = self.loss_func(batch_out, batch_label)
                error += batch_error #/ len(data)
        self.model.train()
        return error.item()
    
        
    def train_test_mini_batch(self, mini_batch):
        '''
        Input: mini_batch
        Return: test scores
        '''
        
        train, test = split_data(mini_batch, ratio2)
        train_dataset = MyDataset(train)
        dataloader = Data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.5)
        
        
        for epoch in range(self.train_epochs):
            loss_epoch = 0.
            for batch_stu_id, batch_item_id, batch_knowledge_id, batch_label in dataloader:
                self.optimizer.zero_grad()
                #batch_out = self.model(batch_stu_id, batch_item_id, batch_knowledge_id, knowledge_n)
                                # gpu
                if self.gpu_available:
                    batch_stu_id, batch_item_id, batch_knowledge_id, batch_label = \
                    batch_stu_id.to(device), batch_item_id.to(device), batch_knowledge_id.to(device), batch_label.to(device)
                
                output_1 = self.model(batch_stu_id, batch_item_id, batch_knowledge_id, knowledge_n)
                output_0 = torch.ones(output_1.size()).to(device) - output_1

                batch_out = torch.cat((output_0, output_1), 1)
                loss_batch = self.loss_func(batch_out, batch_label)
                loss_batch.backward()
                loss_epoch += loss_batch
                self.optimizer.step()
            #loss_epoch = loss_epoch / len(self.train_data)
            self.train_losses.append(loss_epoch.item())    

            # test on validation data
            val_loss = self.evaluate(test)
            self.val_losses.append(val_loss)
            
            if self.meta == True and (val_loss - min(self.val_losses)) > 1e-1:
                break
            
            MODEL_PATH = './results/models/Experiment1/FrcSub_'

            if len(self.val_losses) == 0 or val_loss < min(self.val_losses):
                if self.meta == False:
                    torch.save(self.model.state_dict(), './results/models/Experiment2/Math1/'+self.model_type+'.pt')
                else:
                    torch.save(self.model.state_dict(), './results/models/Experiment2/Math1Meta_'+self.model_type+'.pt')
            else:
                scheduler.step()
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
             
            #logging.info("epoch: ", epoch+1, "| loss: ", loss_epoch.data.item())
        rmse = self.evaluate(self.data['test'])
        accuracy, recall, f1, roc_auc = self.get_test_score(self.data['test'])
        
        del self.train_losses[:]
        del self.val_losses[:]
        del self.test_losses[:]
        
        return rmse, accuracy, recall, f1, roc_auc
    
    def train(self):

        self.results = self.new_results()
        
        rmse, acc, recall, f1, auc = self.train_test_mini_batch(self.data['mini_batch1'])
        self.update_results(rmse, acc, recall, f1, auc, 'batch1')
        
        rmse, acc, recall, f1, auc = self.train_test_mini_batch(self.data['mini_batch2'])
        self.update_results(rmse, acc, recall, f1, auc, 'batch2')
        
        rmse, acc, recall, f1, auc = self.train_test_mini_batch(self.data['mini_batch3'])
        self.update_results(rmse, acc, recall, f1, auc, 'batch3')
        
        return self.results


    def binary_classify(self, data):
        data[data <= threshold] = 0
        data[data > threshold] = 1
        return data.astype(np.int64)
    
    def get_scores(self, true_scores, pred_scores):

#         fpr, tpr, thresholds = metrics.roc_curve(true_scores, pred_scores)
        true_scores = self.binary_classify(true_scores)
        pred_scores = self.binary_classify(pred_scores)
    
#         loss_func = nn.MSELoss()
#         rmse = np.sqrt(((true_scores - pred_scores) ** 2).mean())
        accuracy = accuracy_score(true_scores, pred_scores)
        recall = recall_score(true_scores, pred_scores)
        f1 = f1_score(true_scores, pred_scores)
        roc_auc = roc_auc_score(true_scores, pred_scores)

        return accuracy, recall, f1, roc_auc
    
    def get_test_score(self, data):
        self.model.eval() 
        error = 0.
        with torch.no_grad():
            dataset = MyDataset(data)
            dataloader = iter(Data.DataLoader(dataset, batch_size=len(data), shuffle=False, num_workers=0))
            stu_id, item_id, knowledge_id, true_scores = next(dataloader)
            #gpu
            if self.gpu_available:
                stu_id, item_id, knowledge_id, true_scores = \
                stu_id.to(device), item_id.to(device), knowledge_id.to(device), true_scores.to(device)

            true_scores = true_scores.view(-1).cpu().detach().numpy()
            
            # pred_scores = self.model(stu_id, item_id, knowledge_id, knowledge_n).view(-1).cpu().detach().numpy()
            output_1 = self.model(stu_id, item_id, knowledge_id, knowledge_n).cpu()
            output_0 = torch.ones(output_1.size()) - output_1
            batch_out = torch.cat((output_0, output_1), 1)
            pred = torch.nn.Softmax(dim=1)(batch_out)
            pred_scores = torch.argmax(pred, dim=1).detach().numpy()

            #logging.info(true_scores.shape, pred_scores.shape) same
            # output_1 = self.model(batch_stu_id, batch_item_id, batch_knowledge_id, knowledge_n)
            # output_0 = torch.ones(output_1.size()).to(device) - output_1
            # batch_out = torch.cat((output_0, output_1), 1)
            #logging.info(true_scores.shape, pred_scores.shape)
            accuracy, recall, f1, roc_auc = self.get_scores(true_scores, pred_scores)
        self.model.train()
        return accuracy, recall, f1, roc_auc
    
    # def show_train_val(self, dataname='FrcSub'):
    #     fig, (ax1, ax2) = plt.subplots(2, 1)

    #     x_loss = range(len(self.train_losses))
    #     ax1.plot(x_loss, self.train_losses, label='train loss', color = 'g', linewidth=2)
    #     ax1.set_xlabel('epoch')
    #     ax1.set_ylabel('loss')
    #     #ax1.set_facecolor('lightsteelblue')
    #     ax1.grid(b=True, color='gray', linestyle='--', linewidth=1, alpha=0.8)
    #     ax1.legend()

    #     x_rmse = range(len(self.val_losses))
    #     ax2.plot(x_rmse, self.val_losses, label='val loss', color = 'r', linewidth=2)
    #     ax2.set_xlabel('epoch')
    #     ax2.set_ylabel('error')
    #     ax2.grid(b=True, color='gray', linestyle='--', linewidth=1, alpha=0.8)
    #     ax2.legend()

#hyper parameters
NUM_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 8 #knowledge_embedding_size, dimention of knowledge space
TRAIN_NUM = 100

# Normal over 3-mini-batch
# AGCDM Train without meta
loss_func = nn.CrossEntropyLoss()
meta_learner = MetaLearner('AGCDM', data, \
                 student_n, item_n, knowledge_n, loss_func, \
                 knowledge_embed_size=EMBEDDING_SIZE, epoch_size=NUM_EPOCHS, \
                 batch_size=BATCH_SIZE, learning_rate = LEARNING_RATE)


# meta_learner.learn_algorithm()
meta_learner.reset_model()
rmse = meta_learner.evaluate(data['test'])
acc, recall, f1, auc = meta_learner.get_test_score(data['test'])                  

logging.info('cold-start >')
logging.info("AGCDM | Rmse: {:4.6f} | Accuracy: {:4.6f} | F1: {:4.6f} | AUC: {:4.6f}"\
      .format(rmse, acc, f1, auc))

results = meta_learner.train()

logging.info('warm-up a >')
logging.info("AGCDM | Rmse: {:4.6f} | Accuracy: {:4.6f} | F1: {:4.6f} | AUC: {:4.6f}"\
      .format(results['batch1']['rmse'][0], results['batch1']['acc'][0], \
              results['batch1']['f1'][0], results['batch1']['auc'][0]))
logging.info('warm-up b >')
logging.info("AGCDM | Rmse: {:4.6f} | Accuracy: {:4.6f} | F1: {:4.6f} | AUC: {:4.6f}"\
      .format(results['batch2']['rmse'][0], results['batch2']['acc'][0], \
              results['batch2']['f1'][0], results['batch2']['auc'][0]))
logging.info('warm-up c >')
logging.info("AGCDM | Rmse: {:4.6f} | Accuracy: {:4.6f} | F1: {:4.6f} | AUC: {:4.6f}"\
      .format(results['batch3']['rmse'][0], results['batch3']['acc'][0], \
              results['batch3']['f1'][0], results['batch3']['auc'][0]))

#1-shot meta over 3-mini-batch
# AGCDM Train with meta
loss_func = nn.CrossEntropyLoss()
meta_learner = MetaLearner('AGCDM', data, \
                 student_n, item_n, knowledge_n, loss_func, \
                 knowledge_embed_size=EMBEDDING_SIZE, epoch_size=NUM_EPOCHS, \
                 batch_size=BATCH_SIZE, learning_rate = LEARNING_RATE)

meta_learner.reset_model()
rmse = meta_learner.evaluate(data['test'])
acc, recall, f1, auc = meta_learner.get_test_score(data['test'])                  
meta_learner.learn_algorithm()

logging.info('cold-start >')
logging.info("AGCDM | Rmse: {:4.6f} | Accuracy: {:4.6f} | F1: {:4.6f} | AUC: {:4.6f}"\
      .format(rmse, acc, f1, auc))

results = meta_learner.train()

logging.info('warm-up a >')
logging.info("AGCDM | Rmse: {:4.6f} | Accuracy: {:4.6f} | F1: {:4.6f} | AUC: {:4.6f}"\
      .format(results['batch1']['rmse'][0], results['batch1']['acc'][0], \
              results['batch1']['f1'][0], results['batch1']['auc'][0]))
logging.info('warm-up b >')
logging.info("AGCDM | Rmse: {:4.6f} | Accuracy: {:4.6f} | F1: {:4.6f} | AUC: {:4.6f}"\
      .format(results['batch2']['rmse'][0], results['batch2']['acc'][0], \
              results['batch2']['f1'][0], results['batch2']['auc'][0]))
logging.info('warm-up c >')
logging.info("AGCDM | Rmse: {:4.6f} | Accuracy: {:4.6f} | F1: {:4.6f} | AUC: {:4.6f}"\
      .format(results['batch3']['rmse'][0], results['batch3']['acc'][0], \
              results['batch3']['f1'][0], results['batch3']['auc'][0]))

