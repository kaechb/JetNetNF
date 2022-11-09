from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os
from plotting import plotting
from torch.nn import functional as FF
from lit_nf import LitNF
from jetnet_dataloader import JetNetDataloader
from helpers import mass
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from scipy import stats
import datetime
import pandas as pd
import traceback
import time
import sys
def train(config, hyperopt=False,i=0,root=None):
    # This function is a wrapper for the hyperparameter optimization module called ray 
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    pl.seed_everything(42, workers=True)
    data_module = JetNetDataloader(config) #this loads the data
    model = LitNF(config,hyperopt) # the sets up the model,  config are hparams we want to optimize
    # Callbacks to use during the training, we  checkpoint our models
    print(model.config)
    callbacks = [ModelCheckpoint(monitor="val_logprob",save_top_k=10, filename='{epoch}-{val_logprob:.2f}-{val_w1m:.4f}-{val_w1efp:.6f}-{val_fpnd:.2f}',every_n_epochs=100) ]
    
    model.load_datamodule(data_module)#adds datamodule to model
    model.config = config #config are our hyperparams, we make this a class property now
    logger = TensorBoardLogger(root)

    print(model.config)
    trainer = pl.Trainer(gpus=1, logger=logger,  log_every_n_steps=1000,  # auto_scale_batch_size="binsearch",
                          max_steps=100000 , callbacks=callbacks,
                          check_val_every_n_epoch=100 ,num_sanity_val_steps=0,progress_bar_refresh_rate=0,
                         fast_dev_run=False,default_root_dir=root,max_epochs=-1)
    # This calls the fit function which trains the model
    trainer.fit(model, train_dataloaders=data_module )  


if __name__ == "__main__":
    #Select parton ["q","g","t"], type ["cc,c"] and conditions [0,1,2]
    p="q"
    typ="c"
    c=2
    hyperopt = False
    if typ=="c" and str(c)=="2" and p=="q":
        df=pd.read_csv("/home/kaechben/IML/LitJetNet/best_model/top_{}{}_{}.csv".format(typ,1,p)).set_index("path_index")
        df["context_features"]=2
        df["calc_massloss"]=False
    else:
        df=pd.read_csv("/home/kaechben/IML/LitJetNet/best_model/top_{}{}_{}.csv".format(typ,c,p)).set_index("path_index")
    
    for index,row in df.iterrows():
        config = {
            "network_layers": int(row["network_layers"]),  # sets amount hidden layers int(in transformation networks 
                "network_nodes":int(row["network_nodes"] ),  # amount nodes in hidden layers in transformation networks
                "batch_size":int(row["batch_size"] ),# sets batch size
                "coupling_layers":int(row["coupling_layers"]),# amount of invertible transformations to use
                "lr":row["lr"] , # sets learning rate 
                "batchnorm":row["batchnorm"],# use batchnorm or not 
                "bins":int(row["bins"]),  # amount of bins to use in rational quadratic splines
                "tail_bound":int(row["tail_bound"]),# splines:max value that is transformed, over this value theree is id 
                "limit": int(row["limit"]),  # how many data points to use, test_set is 10% of this -scannable in a sense use int(10 k for faster trainin)g
                "n_dim":int(row["n_dim"]  ),  # how many dimensions to use or equivalently /3 gives the amount of particles to use
                "dropout":row["dropout"],# dropout proportion, for 0 there is no dropout 
                "lr_schedule":row["lr_schedule"] ,# whether tos chedule the learning rate can be False or "smart","exp","onecycle"
                "n_sched":row["n_sched"] , # how many steps between an annealing step
                "canonical":row["canonical"], # transform data coordinates to px,py,pz
                "max_steps":50000, # how many steps to use at max - lower for quicker training
                "lambda":row["lambda"] ,# balance between massloss and nll
                "n_mse_turnoff":1000000,#int(row["n_mse_turnoff"]  ),
                "n_mse_delay":int(row["n_mse_delay"] ),# when to turn on mass loss 
                "name":row["name"] , # name for logging folder
                "calc_massloss":row["calc_massloss"], # whether to calculate mass loss, makes training slower
                "context_features":int(row["context_features"] ), #amount of variables used for conditioning, for 0 no conditioning is used, for 1 o nly the mass is used, for 2 also the number part is used
                "variable":row["variable"], #use variable amount of particles otherwise only use 30, options are true or false 
                "spline":row["spline"],#whether to use splines or not, can also be set to "autoreg" but they are unstable and slow
                "parton":row["parton"], #choose the dataset you want to train options: t for top,q for quark,g for gluon
            }
        root="/beegfs/desy/user/kaechben/bestmodels_nf/top_{}{}_{}".format(typ,c,p)
        if not hyperopt:
            train(config,hyperopt=hyperopt,root=root)
        break #only first row
