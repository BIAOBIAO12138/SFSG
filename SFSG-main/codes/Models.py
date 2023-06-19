import os
import numpy as np
from time import time
from torch.nn import TransformerEncoder,TransformerEncoderLayer
import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import math
import scipy.sparse as sp
from utility.parser import parse_args
from scipy.sparse import csr_matrix 
args = parse_args()


def compute_normalized_laplacian(adj):
    adj=adj.to_dense() 
    rowsum = torch.sum(adj, -1) +1e-7 
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SFSG(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats,interaction_matrix,i_i_matrix):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)

        self.n_nodes = self.n_users + self.n_items
        self.dropout= 0.8
        self.interaction_matrix = interaction_matrix
        self.adj = self.get_norm_adj_mat().cuda()
        self.quant_layer = Quant_layer()
        self.i_i_matrix = i_i_matrix
        self.i_i_matrix = self.get_i_i_matrix().cuda()
        self.sim =0
        self.sim1 =0
        self.knn_k = 10
        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats).cuda(), freeze=False).cuda()
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats).cuda(), freeze=False).cuda()

        if os.path.exists('../data/%s/%s-core/image_adj_%d.pt'%(args.dataset, args.core, args.topk)):
            image_adj = torch.load('/home/WeiYiBiao/LA-baseline/data/%s/%s-core/image_adj_%d.pt'%(args.dataset, args.core, args.topk))
        else:

            indices,adj_size, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            image_adj = self.compute_normalized_laplacian(indices,adj_size)
            torch.save(image_adj, '/home/WeiYiBiao/LA-baseline/data/%s/%s-core/image_adj_%d.pt'%(args.dataset, args.core, args.topk))

        if os.path.exists('../data/%s/%s-core/text_adj_%d.pt'%(args.dataset, args.core, args.topk)):
            text_adj = torch.load('/home/WeiYiBiao/LA-baseline/data/%s/%s-core/text_adj_%d.pt'%(args.dataset, args.core, args.topk))        
        else:
            indices,adj_size, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            text_adj = self.compute_normalized_laplacian(indices,adj_size)
            torch.save(text_adj, '/home/WeiYiBiao/LA-baseline/data/%s/%s-core/text_adj_%d.pt'%(args.dataset, args.core, args.topk))


        self.text_original_adj = text_adj.cuda()
        self.image_original_adj = image_adj.cuda()

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_trs = nn.Linear(image_feats.shape[1], args.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], args.feat_embed_dim)

        self.textConcept = nn.Linear(args.feat_embed_dim, args.feat_embed_dim * args.nbit)
        self.imageConcept = nn.Linear(args.feat_embed_dim, args.feat_embed_dim * args.nbit)   

        self.text_linear = nn.Linear(args.feat_embed_dim * args.nbit,args.feat_embed_dim)  
        self.image_linear = nn.Linear(args.feat_embed_dim * args.nbit,args.feat_embed_dim) 

        self.imagePosEncoder = PositionalEncoding(args.feat_embed_dim, dropout=args.dropout)
        self.textPosEncoder = PositionalEncoding(args.feat_embed_dim, dropout=args.dropout)

        imageEncoderLayer = TransformerEncoderLayer(d_model=args.feat_embed_dim,
                                                    nhead=args.nhead,
                                                    dim_feedforward=args.feat_embed_dim,
                                                    activation=args.act,
                                                    dropout=args.dropout)

        self.imageTransformerEncoder = TransformerEncoder(encoder_layer=imageEncoderLayer, num_layers=args.num_layer)
        textEncoderLayer = TransformerEncoderLayer(d_model=args.feat_embed_dim,
                                                   nhead=args.nhead,
                                                   dim_feedforward=args.feat_embed_dim,
                                                   activation=args.act,
                                                   dropout=args.dropout)
        self.textTransformerEncoder = TransformerEncoder(encoder_layer=textEncoderLayer, num_layers=args.num_layer)


        self.model_weight = nn.Parameter(torch.Tensor([0.5,0.5]))
        self.softmax = nn.Softmax(dim=0)
        self.lambdas = self.compute_concat_scaler()            
    
    
    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values
    
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).cuda()
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        return indices, adj_size,adj
    
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)


    def compute_concat_scaler(self):   #Control the scaling of each layer of the model
        lambdas = [torch.tensor((float)(x + 1) / (self.n_ui_layers + 1)) for x in range(self.n_ui_layers + 1)]
        return lambdas
    
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))
    
    def get_i_i_matrix(self):
        L=sp.coo_matrix(self.i_i_matrix)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_items, self.n_items)))
    

    def forward(self, build_item_graph=False):  

        weight = self.softmax(self.model_weight)
        image_feats = self.image_trs(self.image_embedding.weight)
        image_concept = self.imageConcept(image_feats).reshape(image_feats.size(0), args.nbit, args.feat_embed_dim).permute(1, 0, 2)
        imageSrc = self.imagePosEncoder(image_concept)
        imageMemory = self.imageTransformerEncoder(imageSrc)
        image_feats = imageMemory.permute(1, 0, 2).reshape(image_feats.size(0),args.nbit*args.feat_embed_dim)
        indices,adj_size, image_adj = self.get_knn_adj_mat(image_feats)
        image_adj = self.compute_normalized_laplacian(indices, adj_size)
        text_feats = self.text_trs(self.text_embedding.weight)
        text_concept = self.textConcept(text_feats).reshape(text_feats.size(0), args.nbit, args.feat_embed_dim).permute(1, 0, 2)
        textSrc = self.textPosEncoder(text_concept)
        textMemory = self.textTransformerEncoder(textSrc)
        text_feats = textMemory.permute(1, 0, 2).reshape(text_feats.size(0),args.nbit*args.feat_embed_dim)
        indices,adj_size, text_adj = self.get_knn_adj_mat(text_feats)
        text_adj = self.compute_normalized_laplacian(indices, adj_size)

        if args.drop_node == 1:

            if self.sim ==1:            
                random_noise_i = torch.rand_like(image_feats).cuda()
                image_feats_sub = torch.sign(image_feats) * F.normalize(random_noise_i, dim=-1)  + image_feats 
                random_noise_t = torch.rand_like(text_feats).cuda()
                text_feats_sub = torch.sign(text_feats) * F.normalize(random_noise_t, dim=-1) + text_feats

            elif self.sim1==1:
                mask = torch.zeros((text_feats.shape[0], text_feats.shape[1])).cuda()
                mask[:, :text_feats.shape[1]-256] = 1
                image_feats_sub =image_feats *mask
                text_feats_sub = text_feats*mask

            else:
                self.dropout_node = nn.Dropout(0.4)
                image_feats_sub = self.dropout_node(image_feats)
                text_feats_sub = self.dropout_node(text_feats)

            indices,adj_size, image_adj_sub = self.get_knn_adj_mat(image_feats_sub)    
            indices,adj_size, text_adj_sub = self.get_knn_adj_mat(text_feats_sub)
            image_adj_sub = image_adj+image_adj_sub
            text_adj_sub= text_adj_sub+text_adj
            image_adj_sub = compute_normalized_laplacian(image_adj_sub)
            image_adj_sub = image_adj_sub.to_sparse()
            text_adj_sub = compute_normalized_laplacian(text_adj_sub)
            text_adj_sub = text_adj_sub.to_sparse()

        h_image = self.item_id_embedding.weight
        h_text = self.item_id_embedding.weight
        h_image_sub = self.item_id_embedding.weight
        h_text_sub = self.item_id_embedding.weight
        image_embed = [self.item_id_embedding.weight*self.lambdas[0]]
        text_embed = [self.item_id_embedding.weight*self.lambdas[0]]

        if args.drop_node == 1:
            sub_image_embed = [self.item_id_embedding.weight*self.lambdas[0]]
            sub_text_embed = [self.item_id_embedding.weight*self.lambdas[0]]

        for i in range(self.n_ui_layers):
            h_image = torch.sparse.mm(image_adj, h_image)
            h_text =  torch.sparse.mm(text_adj, h_text)
            if args.drop_node == 1:
                h_image_sub = torch.sparse.mm(image_adj_sub, h_image_sub)
                h_text_sub = torch.sparse.mm(text_adj_sub, h_text_sub)
            image_embed.append(h_image*self.lambdas[i + 1])
            text_embed.append(h_text*self.lambdas[i + 1])
            if args.drop_node == 1:
                sub_image_embed.append(h_image_sub*self.lambdas[i + 1])
                sub_text_embed.append(h_text_sub*self.lambdas[i + 1])
        image_embed = torch.cat(image_embed, dim=1)
        text_embed = torch.cat(text_embed, dim=1)

        if args.drop_node == 1:
            sub_image_embed = torch.cat(sub_image_embed, dim=1)
            sub_text_embed = torch.cat(sub_text_embed,dim=1)
    
        if args.drop_node == 1:
            image_embed_1 = F.normalize(image_embed)
            sub_image_embed_1 = F.normalize(sub_image_embed)
            pos_score_image  = torch.mul(image_embed_1, sub_image_embed_1).sum(dim=1)
            ttl_score_image  = torch.matmul(image_embed_1, sub_image_embed_1.transpose(0, 1))
            pos_score_image = torch.exp(pos_score_image / args.ssl_temp)
            ttl_score_image = torch.exp(ttl_score_image / args.ssl_temp).sum(dim=1) 
            ssl_loss_image  = -torch.log(pos_score_image / ttl_score_image).sum() 
            text_embed_1 = F.normalize(text_embed)
            sub_text_embed_1 = F.normalize(sub_text_embed)
            pos_score_text  = torch.mul(text_embed_1, sub_text_embed_1).sum(dim=1)
            ttl_score_text  = torch.matmul(text_embed_1, sub_text_embed_1.transpose(0, 1))
            pos_score_text = torch.exp(pos_score_text / args.ssl_temp)
            ttl_score_text = torch.exp(ttl_score_text / args.ssl_temp).sum(dim=1) 
            ssl_loss_text  = -torch.log(pos_score_text / ttl_score_text).sum() 
            self.ssl_loss_dropout = (ssl_loss_image  + ssl_loss_text)*args.ssl_reg

        if args.contrastive_i_i ==1:
            
            image_adj_i_i = image_adj+self.i_i_matrix
            text_adj_i_i = text_adj +self.i_i_matrix
            h_image_i_i = self.item_id_embedding.weight
            h_text_i_i = self.item_id_embedding.weight
            image_embed_i_i = [self.item_id_embedding.weight*self.lambdas[0]]
            text_embed_i_i = [self.item_id_embedding.weight*self.lambdas[0]]  
     
            for i in range(self.n_ui_layers):
                h_image_i_i = torch.sparse.mm(image_adj_i_i, h_image_i_i)
                h_text_i_i =  torch.sparse.mm(text_adj_i_i, h_text_i_i)
                image_embed_i_i.append(h_image_i_i*self.lambdas[i + 1])
                text_embed_i_i.append(h_text_i_i*self.lambdas[i + 1])
            image_embed_i_i = torch.cat(image_embed_i_i,dim=1)
            text_embed_i_i = torch.cat(text_embed_i_i,dim=1)

            image_embed_1 = F.normalize(image_embed)
            image_embed_i_i = F.normalize(image_embed_i_i)
            pos_score_image  = torch.mul(image_embed_1, image_embed_i_i).sum(dim=1)
            ttl_score_image  = torch.matmul(image_embed_1,image_embed_i_i.transpose(0, 1))
            pos_score_image = torch.exp(pos_score_image / args.ssl_temp1)
            ttl_score_image = torch.exp(ttl_score_image / args.ssl_temp1).sum(dim=1) 
            ssl_loss_image_i_i  = -torch.log(pos_score_image / ttl_score_image).sum() 
            text_embed_1 = F.normalize(text_embed)
            text_embed_i_i = F.normalize(text_embed_i_i)
            pos_score_text  = torch.mul(text_embed_1, text_embed_i_i).sum(dim=1)
            ttl_score_text  = torch.matmul(text_embed_1, text_embed_i_i.transpose(0, 1))
            pos_score_text = torch.exp(pos_score_text / args.ssl_temp1)
            ttl_score_text = torch.exp(ttl_score_text / args.ssl_temp1).sum(dim=1) 
            ssl_loss_text_i_i  = -torch.log(pos_score_text / ttl_score_text).sum() 
            ssl_loss_i_i =  (ssl_loss_text_i_i+ssl_loss_image_i_i)*args.ssl_reg1          

        learned_adj = 0.1 * image_adj + 0.9 * text_adj 
        original_adj=0.5 * self.image_original_adj+ 0.5 * self.text_original_adj
        original_adj = learned_adj*0.1 +original_adj*0.9
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings* self.lambdas[0]]
        ego_embeddings_mm = self.item_id_embedding.weight
        all_embeddings_mm = [ego_embeddings_mm* self.lambdas[0]]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.adj, ego_embeddings)
            ego_embeddings_mm = torch.sparse.mm(original_adj, ego_embeddings_mm)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings*self.lambdas[i + 1])
            all_embeddings_mm.append(ego_embeddings_mm*self.lambdas[i + 1])
        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        all_embeddings_mm = torch.cat(all_embeddings_mm, dim=1)
        i_pre = torch.matmul(original_adj, i_g_embeddings)
        i_pre = i_pre.squeeze()
        i_g_embeddings = i_g_embeddings + F.normalize(all_embeddings_mm, p=2, dim=1)
        u_g_embeddings = u_g_embeddings 
        return u_g_embeddings, i_g_embeddings, self.ssl_loss_dropout+ssl_loss_i_i


