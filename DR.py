import argparse
import random
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from arch.news import NetworkNews
from dataset import RedditDRDataset
import ops
import utils
import wandb

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_queries', type=int, default=999)
    parser.add_argument('--max_queries_test', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=0.2)
    parser.add_argument('--sampling', type=str, default='random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='redditdr')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--tail', type=str, default='', help='tail message')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load checkpoint')
    parser.add_argument('--save_dir', type=str, default='./saved_redditdr/', help='save directory')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    args = parser.parse_args()
    return args

def main(args):
    # Wandb
    run = wandb.init(project="Variational-IP", name=args.name, mode=args.mode)
    model_dir = os.path.join(args.save_dir, f'{run.id}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
    utils.save_params(model_dir, vars(args))
    wandb.config.update(args)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE:', device)
    
    # Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Data
    # Step 1: build vocab from all splits to keep feature dimensions consistent
    trainset_full = RedditDRDataset(args.train_csv)
    valset_full   = RedditDRDataset(args.val_csv, vocab=trainset_full.vocab)
    testset_full  = RedditDRDataset(args.test_csv, vocab=trainset_full.vocab)
    trainloader   = DataLoader(trainset_full, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader     = DataLoader(valset_full, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testloader    = DataLoader(testset_full, batch_size=args.batch_size, shuffle=False, num_workers=4)

    N_FEATURES    = len(trainset_full.vocab)
    N_CLASSES     = 2
    N_QUERIES     = N_FEATURES
    THRESHOLD     = 0.85
    
    # Model
    classifier = NetworkNews(query_size=N_QUERIES, output_size=N_CLASSES)
    classifier = nn.DataParallel(classifier).to(device)
    querier = NetworkNews(query_size=N_QUERIES, output_size=N_QUERIES, tau=args.tau_start)
    querier = nn.DataParallel(querier).to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(classifier.parameters()) + list(querier.parameters()), amsgrad=True, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)

    # Load checkpoint
    if args.ckpt_path is not None:
        ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
        classifier.load_state_dict(ckpt_dict['classifier'])
        querier.load_state_dict(ckpt_dict['querier'])
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        scheduler.load_state_dict(ckpt_dict['scheduler'])
        print('Checkpoint Loaded!')

    # Training loop
    for epoch in range(args.epochs):
        classifier.train()
        querier.train()
        tau = tau_vals[epoch]
        for train_features, train_labels in tqdm(trainloader, desc=f"Training Epoch {epoch}"):
            train_features = train_features.to(device)
            train_labels   = train_labels.to(device)
            train_bs       = train_features.shape[0]
            querier.module.update_tau(tau)
            optimizer.zero_grad()

            # Adaptive/Random sampling
            if args.sampling == 'baised':
                mask = ops.adaptive_sampling(train_features, args.max_queries, querier).to(device).float()
            else:
                mask = ops.random_sampling(args.max_queries, N_QUERIES, train_bs).to(device).float()
            history = train_features * mask

            # Query and update
            query = querier(history, mask)
            updated_history = history + train_features * query

            train_logits = classifier(updated_history)
            loss = criterion(train_logits, train_labels)
            loss.backward()
            optimizer.step()

            # Logging
            wandb.log({
                'epoch'      : epoch,
                'loss'       : loss.item(),
                'lr'         : utils.get_lr(optimizer),
                'gradnorm_cls': utils.get_grad_norm(classifier),
                'gradnorm_qry': utils.get_grad_norm(querier)
            })
        scheduler.step()

        # Save
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'classifier': classifier.state_dict(),
                'querier'   : querier.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, os.path.join(model_dir, 'ckpt', f'epoch{epoch}.ckpt'))

        # Evaluation
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            classifier.eval()
            querier.eval()
            # Validation
            val_acc = 0
            for val_features, val_labels in tqdm(valloader, desc=f"Validation Epoch {epoch}"):
                val_features = val_features.to(device)
                val_labels   = val_labels.to(device)
                val_bs       = val_features.shape[0]
                mask_val     = torch.zeros(val_bs, N_QUERIES).to(device)
                logits = classifier(val_features * mask_val)
                val_pred = logits.argmax(dim=1)
                val_acc += (val_pred == val_labels.squeeze()).float().sum().item()

            val_acc = val_acc / len(valset_full)
            wandb.log({'val_acc': val_acc})

            # Test
            test_acc = 0
            for test_features, test_labels in tqdm(testloader, desc=f"Test Epoch {epoch}"):
                test_features = test_features.to(device)
                test_labels   = test_labels.to(device)
                test_bs       = test_features.shape[0]
                mask_test     = torch.zeros(test_bs, N_QUERIES).to(device)
                logits = classifier(test_features * mask_test)
                test_pred = logits.argmax(dim=1)
                test_acc += (test_pred == test_labels.squeeze()).float().sum().item()

            test_acc = test_acc / len(testset_full)
            wandb.log({'test_acc': test_acc})

if __name__ == '__main__':
    args = parseargs()
    main(args)
