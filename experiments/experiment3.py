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
import argparse
import logging
import os
import sys
# sys.path.insert('..')

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import warnings


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def set_seed(seed=123456):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)

def load_data(path, val_ratio, test_ratio):
    full_data = pd.read_csv(os.path.join(args.data_file, 'data.txt'), header=None, sep='\t').values.astype(np.float32)
    knowledge_matrix = pd.read_csv(os.path.join(args.data_file, 'q.txt'), header=None, sep='\t').values.astype(np.float32)
    students_num, items_num, skills_num = full_data.shape[0], full_data.shape[1], knowledge_matrix.shape[1]
    data = np.array([{'stu_id': stu_id, 'item_id': item_id, 'score': full_data[stu_id, item_id], 'knowledge': knowledge_matrix[item_id]}
          for stu_id in range(students_num) for item_id in range(items_num)])

    np.random.shuffle(data)
    
    train_val_data = data[ : int(len(data) * test_ratio)]
    test_data = data[int(len(data) * test_ratio) : ]
    
    train_data = train_val_data[ : int(len(train_val_data) * val_ratio)]
    val_data = train_val_data[int(len(train_val_data) * val_ratio) : ]
    
    return {'train_data': train_data, 'val_data': val_data, 'test_data': test_data, 'students_num': students_num, 'items_num': items_num, 'skills_num':  skills_num}

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
            slip and gate is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            guess_prob = torch.sigmoid(self.guess[layer](x)) # distribution of guess
            slip_prob = torch.sigmoid(self.slip[layer](x)) # distribution of slip
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
    
class MyDataset(Data.Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__() 
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['stu_id'], \
        self.data[idx]['item_id'], \
        self.data[idx]['knowledge'], \
        self.data[idx]['score']

class Learner(object):
    
    def __init__(self, model_type, train_data, val_data, test_data, \
                 student_n, item_n, knowledge_n, loss_func, \
                 knowledge_embed_size, epoch_size, \
                 batch_size, lr, gpu_available, args):
        
        super(Learner, self).__init__()
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.student_n = student_n
        self.item_n = item_n
        self.knowledge_n = knowledge_n
        self.knowledge_embed_size = knowledge_embed_size
        
        self.train_epochs = epoch_size
        self.batch_size = batch_size
        self.lr = lr

        self.gpu_available = gpu_available
        self.model_type = model_type
        self.model = AGCDM(student_n, item_n, knowledge_n, knowledge_embed_size)

        
          # gpu
        if args.gpu_available:
            self.model = self.model.to(args.device)

        self.loss_func = loss_func
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # meta-leaner hyperparameters
        self.meta = False
        self.num_tasks = 11
        self.num_samples = 32
        self.task_epochs = 200
        self.alpha = 1e-3
        self.beta = 1e-3
        self.lam = 1e-3

        
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
    


    def sample_task_data(self, data):
        dataloader = Data.DataLoader(MyDataset(data), batch_size=self.num_samples, shuffle=True, num_workers=0) 
        task_data = next(iter(dataloader))
        return task_data
    
    def show_params_grad(self):
        for params in self.model.parameters():
            print(params.grad)
            break
        
        
    def train_task(self, task_data):

        stu, item, knowledge, label = task_data[0], task_data[1], task_data[2], task_data[3]

        if self.gpu_available:
            stu, item, knowledge, label = \
            stu.to(args.device), item.to(args.device), knowledge.to(args.device), label.to(args.device)

        self.optimizer.zero_grad()
        out = self.model(stu, item, knowledge)
        loss_task = self.loss_func(out.view(-1), label)
        loss_task.backward() 
        self.optimizer.step()


    def new_model(self):

        if self.model_type == 'AGCDM':
            model = AGCDM(student_n, item_n, knowledge_n, knowledge_embed_size)
        elif self.model_type == 'ACDM':
            model = ACDM(student_n, item_n, knowledge_n, knowledge_embed_size)
        else:
            raise ValueError('No models')

        if self.gpu_available:
            model = model.to(self.args.device)

        return model
        
    def reset_model(self):
        self.model = self.new_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.meta = False
        del self.train_losses[:]
        del self.val_losses[:]
        del self.test_losses[:]
        
    def learn_algorithm(self):
        
        logging.info("Learning an algorithm from current dataset....")
        self.meta = True
        
        for e in range(self.task_epochs):        
            
            self.opti_params_ = []

            #1. for train task i in batch of tasks
            for i in range(self.num_tasks):
                
                task_data = self.sample_task_data(self.train_data)
    
                self.train_task(task_data)
                
                opti_params = deepcopy(self.model.state_dict())
                
                self.opti_params_.append(opti_params)
            
                
            meta_grad_dict = deepcopy(self.model.state_dict())
            meta_grad_dict = {name: nn.init.constant_(meta_grad_dict[name], 0.) for name in meta_grad_dict} 
            
            
            #2. Add each tasks loss, backprogate to get a "fitness" parameters
            for i in range(self.num_tasks):
                
                task_data = self.sample_task_data(train_data)
                stu, item, knowledge, label = task_data[0], task_data[1], task_data[2], task_data[3]
                if self.gpu_available:
                    stu, item, knowledge, label = \
                    stu.to(args.device), item.to(args.device), knowledge.to(args.device), label.to(args.device)
                
                net_optim = self.new_model()

                if self.gpu_available:
                    net_optim = net_optim.to(args.device)

                net_optim.load_state_dict(self.opti_params_[i])
                
                out = net_optim(stu, item, knowledge)
                
                loss = self.loss_func(out, label)
                
                loss.backward()
                
                #update meta gradient bt net_optim_params's grad
                net_optim_params_grad = {}
                for name, params in zip(net_optim.state_dict(), net_optim.parameters()):
                    net_optim_params_grad[name] = params.grad.data
                #print(net_optim_params_grad)
                meta_grad_dict = {name: meta_grad_dict[name] + net_optim_params_grad[name] / self.num_samples for name in meta_grad_dict} 
                #meta_grad_dict = {name: meta_grad_dict[name] + net_optim_params[name].grad.data / self.num_samples for name in meta_grad_dict} 
            
            
            #update net params by meta gradient
            net_params = self.model.state_dict()
            net_params_new = {name: net_params[name] + self.beta * meta_grad_dict[name] / self.num_samples for name in net_params} 
            self.model.load_state_dict(net_params_new)
    
    
    def evaluate(self, data):
        self.model.eval()
        error = 0.
        with torch.no_grad():
            dataset = MyDataset(data)
            dataloader = Data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            for batch_stu_id, batch_exer_id, batch_knowledge_id, batch_label in dataloader:
                # gpu
                if self.gpu_available:
                    batch_stu_id, batch_exer_id, batch_knowledge_id, batch_label = \
                    batch_stu_id.to(args.device), batch_exer_id.to(args.device), batch_knowledge_id.to(args.device), batch_label.to(args.device)

                predict = self.model(batch_stu_id, batch_exer_id, batch_knowledge_id, knowledge_n)
                batch_error = self.loss_func(predict.view(-1), batch_label)
                error += batch_error #/ len(data)
        self.model.train()
        return error.item()
    
        
    def train(self):
        #warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        if self.meta == True:
            self.args.lr /= 10
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
            self.train_epochs = int(self.train_epochs / 5)
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.5) 
        train_dataset = MyDataset(self.train_data)
        dataloader = Data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=0)   
        
        for epoch in range(self.train_epochs):
            loss_epoch = 0.
            for batch_stu_id, batch_item_id, batch_knowledge_id, batch_label in dataloader:
                # gpu
                if self.gpu_available:
                    batch_stu_id, batch_item_id, batch_knowledge_id, batch_label = \
                    batch_stu_id.to(args.device), batch_item_id.to(args.device), \
                        batch_knowledge_id.to(args.device), batch_label.to(args.device)

                self.optimizer.zero_grad()
                batch_out = self.model(batch_stu_id, batch_item_id, batch_knowledge_id, knowledge_n)
                
                loss_batch = self.loss_func(batch_out.view(-1), batch_label)
                loss_batch.backward()
                loss_epoch += loss_batch
                self.optimizer.step()
            #loss_epoch = loss_epoch / len(self.train_data)
            self.train_losses.append(loss_epoch.item())    

            # test on validation data
            val_loss = self.evaluate(self.val_data)
            self.val_losses.append(val_loss)

            if len(self.val_losses) == 0 or val_loss < min(self.val_losses):
                if self.meta == False:
                    torch.save(self.model.state_dict(), './results/models/Experiment1/'+args.data_file.split('/')[-2]+'AGCDM.pt')
                else:
                    torch.save(self.model.state_dict(), './results/models/Experiment2/'+args.data_file.split('/')[-2]+'Meta_AGCDM.pt')
            else:
                scheduler.step()
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                
            #print("epoch: ", epoch+1, "| loss: ", loss_epoch.data.item())
            
    def binary_classify(self, data):
        data[data <= 0.5] = 0
        data[data > 0.5] = 1
        return data.astype(np.int64)
    
    def get_scores(self, true_scores, pred_scores):

        true_scores = self.binary_classify(true_scores)
        pred_scores = self.binary_classify(pred_scores)
    

        accuracy = accuracy_score(true_scores, pred_scores)
        precision = precision_score(true_scores, pred_scores)
        recall = recall_score(true_scores, pred_scores)
        f1 = f1_score(true_scores, pred_scores)
        roc_auc = roc_auc_score(true_scores, pred_scores)

        return accuracy, precision, recall, f1, roc_auc
    
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
                stu_id.to(args.device), item_id.to(args.device), knowledge_id.to(args.device), true_scores.to(args.device)

            true_scores = true_scores.view(-1).cpu().detach().numpy()
            pred_scores = self.model(stu_id, item_id, knowledge_id, knowledge_n).view(-1).cpu().detach().numpy()
            #print(true_scores.shape, pred_scores.shape)
            accuracy, precision, recall, f1, roc_auc = self.get_scores(true_scores, pred_scores)
        self.model.train()
        return accuracy, precision, recall, f1, roc_auc
    
    def show_train_val(self, dataname='Math'):
        _, (ax1, ax2) = plt.subplots(2, 1)

        x_loss = range(len(self.train_losses))
        ax1.plot(x_loss, self.train_losses, label='train loss', color = 'g', linewidth=2)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        #ax1.set_facecolor('lightsteelblue')
        ax1.grid(b=True, color='gray', linestyle='--', linewidth=1, alpha=0.8)
        ax1.legend()

        x_rmse = range(len(self.val_losses))
        ax2.plot(x_rmse, self.val_losses, label='val loss', color = 'r', linewidth=2)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('error')
        ax2.grid(b=True, color='gray', linestyle='--', linewidth=1, alpha=0.8)
        ax2.legend()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AGCDM model (For Ablation Study).')
    ## Required parameters

    parser.add_argument("--data_file", default='./dataset/Math/', type=str, required=True,
                        help="The input data file.")

    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size for training.")

    parser.add_argument("--embedding_size", default=32, type=int,
                        help="Embedding Size for student, skills, and tests.") 
    
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for the optimizer.")

    parser.add_argument("--log_interval", default=1, type=int,
                        help="Log interval steps for epoch and loss.")

    parser.add_argument("--valid_freq", default=1, type=int,
                        help="Log interval steps for epoch and loss.")
    
    parser.add_argument("--save_model", default=True, type=bool,
                        help="Save model to the checkpoint path.")
    
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    
    parser.add_argument("--resume", default=False, type=bool,
                        help="Resume training")

    parser.add_argument("--start_epoch", default=0, type=int,
                        help="start of epoch, optional if resume a interval trained model.")     
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  

    parser.add_argument("--momentum", default=0, type=int,
                        help="Training momentum for the optimizer.")

    parser.add_argument("--cpkt_model_path", default="./results/cpkts/", type=str,
                        help="Checkpoint path for saving model and optimizer.")
                        
    parser.add_argument("--eps", default=1e-8, type=float,
                        help="The eps for Adam.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--weight_decay', type=float, default=0.,
                        help="Weight decay")

    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")

    # parser.add_argument('--local_rank', default=-1, type=int,
    #                 help='node rank for distributed training')

    # parser.add_argument('--world_size', default=-1, type=int,
    #             help='node rank for distributed training')

    args = parser.parse_args()

        #set log
    logging.basicConfig(filename='results/logs/AGCDM.log', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.DEBUG)
    
    # print(args)
    
    # args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.gpu_available = True if args.device=='cuda' else False
    args.nprocs = torch.cuda.device_count()
    # logger.info("process number: ", args.nprocs)
    # logger.info("local rank: ", args.local_rank)

    data = load_data(path=args.data_file, val_ratio=0.8, test_ratio=0.8)
    # print(data['train_data'].shape, data['val_data'].shape, data['test_data'].shape)
    # print(data['train_data'][1], '\n', data['val_data'][1], '\n', data['test_data'][1])

    train_data, val_data, test_data = data['train_data'], data['val_data'], data['test_data']
    student_n, item_n, knowledge_n, knowledge_embed_size = data['students_num'], data['items_num'], data['skills_num'], args.embedding_size

    train_dataset = MyDataset(train_data)
    dataloader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


    # AGCDM Train
    loss_func = nn.MSELoss()
    learner = Learner('AGCDM', train_data, val_data, test_data, \
                    student_n, item_n, knowledge_n, loss_func, \
                    knowledge_embed_size=args.embedding_size, epoch_size=args.epochs, \
                    batch_size=args.batch_size, lr = args.learning_rate, gpu_available=args.gpu_available, args=args)

    learner.reset_model()
    learner.train()

    learner.evaluate(test_data)

    accuracy, precision, recall, f1, roc_auc = learner.get_test_score(test_data)
    print("AGCDM | Accuracy: {:4.6f} | Precision: {:4.6f} | Recall: {:4.6f} | F1: {:4.6f} | AUC: {:4.6f}"\
        .format(accuracy, precision, recall, f1, roc_auc))

    train_losses, val_losses = learner.train_losses.copy(), learner.val_losses.copy()
    normal_train = {'loss': train_losses, 'rmse': val_losses}