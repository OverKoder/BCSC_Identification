import numpy as np

from src.data.load_data import get_dataloaders
from src.ml.losses import LossFunction
from src.ml.models.mlp import MLP
from src.ml.models.embedder import Embedder

import pandas as pd 
import scanpy


import torch
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter

from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.miners import BatchHardMiner

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

# List containing the number of neurons and layers
EMB_ARCHITECTURES = [
    [100],[100,100],[100,100,100], 
    [250],[250,250],[250,250,250],
    [500],[500,500],[500,500,500],
    [1000],[1000,1000],[1000,1000,1000], 
    [2500],[2500,2500],[2500,2500,2500],
    [5000],[5000,5000],[5000,5000,5000],
    [10000],[10000,10000],[10000,10000,10000],
]

# GPU Device
DEVICE = 'cuda:1'
BEST_PARAMS = {}
WEIGHTS_PATH = ""

def train_embedder(embedder, loss_function, miner, loss_optimizer, emb_optimizer, device, trainloader, logger, epoch):
    """
    Trains an embedder model for 1 epoch

    Args:
        embedder: Embedder model (a torch.nn.Module)
        loss_function: Loss function.
        miner: Triplet miner for the loss function.
        loss_optimizer: Metric learning functions need an optimizer
        emb_optimizer: Optimizer for the model
        device (str): Device to use (CPU or GPU).
        trainloader (torch.utils.data.DataLoader): Training data loader.
        logger (SummaryWriter): Tensorboard logger.
        epoch (int): The epoch number

    """

    print('-------=| Epoch %d |=-------' % epoch)

    # Set models to train
    embedder.train()

    train_emb_loss = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader):

        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.long()

        # Reset gradient
        loss_optimizer.zero_grad()
        emb_optimizer.zero_grad()

        # Forward pass
        embeddings = embedder(inputs)
        
        # Mine the triplets (metric learning step)
        triplets = miner(embeddings, labels[:,0])

        # Get loss (add triplet indexes, metric learning)
        loss = loss_function(embeddings, labels[:,0], triplets)

        # Backward and optimize
        loss.backward()
        loss_optimizer.step()
        emb_optimizer.step()

        train_emb_loss += loss.item()
    
    train_emb_loss = train_emb_loss / len(trainloader)

    # Log the loss
    logger.add_scalar("Training loss", train_emb_loss, epoch)

    return train_emb_loss

def validate_embedder(embedder, loss_function, miner, device, valloader, logger, best_loss, epoch):
    """
    Trains an embedder model for 1 epoch

    Args:
        embedder: Embedder model (a torch.nn.Module)
        loss_function: Loss function.
        miner: Triplet miner for the loss function.
        device (str): Device to use (CPU or GPU).
        valloader (torch.utils.data.DataLoader): Training data loader.
        logger (SummaryWriter): Tensorboard logger
        best_loss (float): Best loss to keep track of progress and save model.
        epoch (int): Epoch number.

    """
    global BEST_PARAMS, WEIGHTS_PATH

    # Set models to evaluation
    embedder.eval()

    val_emb_loss = 0

    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(valloader):

            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()

            # Forward pass
            embeddings = embedder(inputs)
            
            # Mine the triplets (metric learning step)
            triplets = miner(embeddings, labels[:,0])
            
            # Get loss (add triplet indexes, metric learning)
            loss = loss_function(embeddings, labels[:,0], triplets)
            
            val_emb_loss += loss.item()
    
    loss = val_emb_loss / len(valloader)

    # Log the loss
    logger.add_scalar("Validation Loss", loss, epoch)
    
    # If we had the best loss until now, save the model
    if loss < best_loss:

        best_loss = loss

        BEST_PARAMS = embedder.get_params()
        WEIGHTS_PATH = 'checkpoints/' + embedder.get_name() + '.pt'

        print("Saving checkpoint...")
        torch.save(embedder.state_dict(), WEIGHTS_PATH)

    return best_loss, loss

def train_mlp(train_loader, val_loader, prefix):
    """
    Trains the MLP with the attached embedder
    """
    global BEST_PARAMS, WEIGHTS_PATH

    loss_function = BCEWithLogitsLoss()

    # Callback, after each epoch, we save based on the monitor, 'validation_f1', and save 1 model k=1 that maximizes
    # that value mode='max'
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_f1',
        dirpath='checkpoints/',
        filename= prefix + '{validation_f1:.3f}',
        save_top_k=1,
        mode='max',
    )

    print("Loading best embedder...")

    embedder = Embedder(input_size = BEST_PARAMS['input_size'], layers = BEST_PARAMS['layers'], activation = BEST_PARAMS['activation'], use_residual = BEST_PARAMS['use_residual'])
    embedder.load_state_dict(torch.load(WEIGHTS_PATH, map_location = 'cpu'))
    embedder.to(DEVICE)

    print("Done.")
    emb_size = BEST_PARAMS['layers'][-1]
    params = [ ( [emb_size] * i ) + [1] for i in range(5)]
    for layers in params:

        # Name for log directory
        run_name = prefix + '_' + str(len(layers)) + "_layers_" + str(layers[0])
        print("Running", run_name)
        logger = TensorBoardLogger(version = run_name, name=prefix, save_dir='./')

        # Trainer (devices=[1], must be the same as the global variable DEVICE='cuda:1', otherwise it will raise an error)
        trainer = Trainer(accelerator='gpu', devices=[1], max_epochs=30, enable_progress_bar=False, callbacks=[checkpoint_callback], logger=logger)
        #trainer = Trainer(accelerator='cpu',max_epochs=1, enable_progress_bar=True, callbacks=[checkpoint_callback], logger=logger)

        # MLP model
        model = MLP(input_size = emb_size, layers = layers, optimizer = AdamW, loss_function = loss_function, embedder = embedder)
        trainer.fit(model, train_loader, val_loader)
            
    return

def train_embedder_loop(train_loader, val_loader, prefix, input_size):

    # Triplet miner
    miner = BatchHardMiner()

    best_loss = 2**31
    for layers in EMB_ARCHITECTURES:

        for activation in ['relu', 'leakyrelu', 'gelu']:

            # Embedding size is the number of neurons in the last layer
            emb_size = layers[-1]

            # Loss function and optimizer
            loss_function = ArcFaceLoss(num_classes = 2, embedding_size = emb_size, margin = 15, scale = 64)
            loss_optimizer = AdamW(loss_function.parameters(), lr = 1e-3)

            # Model and optmizer
            embedder = Embedder(input_size = 41042, layers = layers, activation = activation, use_residual = True, prefix = prefix, suffix = '').to(DEVICE)
            emb_optimizer = AdamW(embedder.parameters(), lr = 1e-3)

            # Name for the loggers
            run_name = embedder.get_name()

            # Logger
            logger = SummaryWriter(
                log_dir = prefix + '/' + run_name + '/',
                comment = run_name
            )

            print("Running:", run_name)
            for epoch in range(30):

                train_loss =  train_embedder(embedder, loss_function, miner, loss_optimizer, emb_optimizer, DEVICE, train_loader, logger, epoch)
                print("Train loss", train_loss)
            
                best_loss, val_loss = validate_embedder(embedder, loss_function, miner, DEVICE, val_loader, logger, best_loss, epoch)
                print("Val loss", val_loss)
            
    return

def main():

    for input_size, name, format in zip([8547,92852,84549], ['StemMarkers', 'AllGenes', 'NonStemMarkers'],['objects/StemMarkerFormat.pk', 'objects/AllGenesFormat.pk', 'objects/NonStemMarkerFormat.pk']):

        annotation = np.load('annotations/HSD_annonation.npy', allow_pickle = True)
        # feature_names = list(np.load('annotations/bonemarrow_genes.npy', allow_pickle = True))
        train_loader, val_loader, test_loader = get_dataloaders(
        data_path = 'HSD.npz',
        annotation = annotation,
        positives = ['HSCs'],
        ratios = [0.7, 0.15, 0.15],
        feature_names = feature_names,
        batch_size = 128,
        cell_slice = None,
        gene_slice = None,
        do_format_cells=True,
        format_dict = format,
        random_state = 42

        )

        train_embedder_loop(train_loader, val_loader,input_size, prefix = name + 'HSCsEmbedder')

        train_mlp(train_loader, val_loader, input_size, prefix = name + 'HSCsMLP')
    
    print("----------=| All done |=------------")
    return

if __name__ == "__main__":
    main()