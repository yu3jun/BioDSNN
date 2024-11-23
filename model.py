import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoder,TransformerDecoderLayer
import pandas as pd
from VAE import VAE
import numpy as np

from torch_geometric.nn import SGConv

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)








class BioDSNN_Model(torch.nn.Module):
    """
    BioDSNN model

    """

    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(BioDSNN_Model, self).__init__()
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size'] # 64
        self.hidden_size = hidden_size
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        

        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        

        self.G_coexpress = args['G_coexpress'].to(args['device'])  
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        

        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        

        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        
        

        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        

        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
    def forward(self, data, mask):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx  
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())


            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device'])) 
            emb = self.bn_emb(emb) 
            base_emb = self.emb_trans(emb) 

            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1: 
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T
            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))     

            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()


            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
               
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:

                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb) 
            base_emb = self.transform(base_emb)  
            out = self.recovery_w(base_emb) 
            out = out.reshape(num_graphs, self.num_genes, -1) 
            d_model = self.hidden_size  
            
            encoder_layers = TransformerEncoderLayer(d_model=d_model, dim_feedforward = 512,nhead=2,batch_first = True)
            transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=1)   
            transformer_encoder = transformer_encoder.to(self.args['device'])
            
            output = transformer_encoder(out, mask=mask)
            out = output.unsqueeze(-1) * self.indv_w1 
            w = torch.sum(out, axis = 2) 
            out = w + self.indv_b1  
            out = out.reshape(num_graphs ,self.num_genes)
            x = x.reshape(num_graphs, self.num_genes)
            input_dim = x.size(1)
            batch_size = x.size(0)
            vae_model = VAE(input_dim = input_dim, inter_dim=32, latent_dim=16)
            vae_model.to(self.args['device'])
            recon_x, mu, logvar = vae_model(x)
            kl_loss = vae_model.KL_loss(mu, logvar)
            recon_x_clone = recon_x.clone()  
            zero_indices = torch.where(x == 0)
            recon_x_clone[zero_indices] = 0
            pred = out + recon_x_clone 
            
            return pred, kl_loss