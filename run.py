from biodsnn import BioDSNN
from pertdata import PertData
from datetime import datetime
import os
import pandas as pd
import pickle
import numpy as np

datapath = './data'
data_name = 'norman'
# data_name = 'replogle_k562_essential'
# data_name = 'replogle_rpe1_essential'
split = 'simulation'
seed = 1
batch_size = 64
test_batch_size = 64
hidden_size = 64
epochs = 15
nhead=2
num_encoder_layers=1
dim_feedforward = 512
VAE_latent_dim=16

no_perturb = False

current_time = datetime.now()

current_time = current_time.replace(second=0, microsecond=0)

formatted_time = current_time.strftime("%Y-%m-%d %H:%M")

log_str = f"datapath {datapath},data_name {data_name}, split {split}, seed {seed}, \nbatch_size {batch_size}, test_batch_size {test_batch_size}, hidden_size {hidden_size},\nepochs {epochs}, no_perturb {no_perturb}, nhead{nhead},num_encoder_layers{num_encoder_layers},\ntransformerlike encoder,dim_feedforward{dim_feedforward},VAE_latent_dim {VAE_latent_dim}"

def write_log(message, log_file_path=None):
    """Write log messages to a file."""
    if log_file_path is None:

        base_path = "./runs/BioDSNN/" + data_name + "/" + split + "/" + formatted_time
        log_file_path = os.path.join(base_path, "model_log.txt")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")

write_log(message=log_str)


pert_data = PertData(datapath)

pert_data.load(data_name)

pert_data.prepare_split(split, seed)

pert_data.get_dataloader(batch_size, test_batch_size)


BioDSNN_model = BioDSNN(pert_data,data_name,split, device = 'cuda:0')
BioDSNN_model.model_initialize(hidden_size,no_perturb=no_perturb)
BioDSNN_model.train(epochs)

BioDSNN_model.evaluate('model/BioDSNN/'+ data_name + "/" + split + "/" + formatted_time)

