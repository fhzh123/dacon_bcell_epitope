# Import modules
import os
import gc
import pickle
import logging
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.dataset import Seq2LabelDataset
from model.custom_transformer.transformer import Transformer
from model.custom_plm.T5 import custom_T5
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log

from transformers import BertForSequenceClassification

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start training!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    save_name = f'processed.pkl'

    with open(os.path.join(args.preprocess_path, save_name), 'rb') as f:
        data_ = pickle.load(f)
        train_src_input_ids = data_['train_src_input_ids']
        train_src_attention_mask = data_['train_src_attention_mask']
        valid_src_input_ids = data_['valid_src_input_ids']
        valid_src_attention_mask = data_['valid_src_input_ids']
        train_trg_list = data_['train_label']
        valid_trg_list = data_['valid_label']
        del data_

    gc.enable()
    write_log(logger, "Finished loading data!")

    # 2) Dataloader setting
    dataset_dict = {
        'train': Seq2LabelDataset(src_list=train_src_input_ids, src_att_list=train_src_attention_mask,
                                  trg_list=train_trg_list, min_len=args.min_len, src_max_len=args.src_max_len),
        'valid': Seq2LabelDataset(src_list=valid_src_input_ids, src_att_list=valid_src_attention_mask,
                                  trg_list=valid_trg_list, min_len=args.min_len, src_max_len=args.src_max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    if args.model_type == 'custom_transformer':
        model = Transformer(d_model=args.d_model, d_embedding=args.d_embedding, 
                            n_head=args.n_head, dim_feedforward=args.dim_feedforward,
                            num_encoder_layer=args.num_encoder_layer, src_max_len=args.src_max_len, 
                            dropout=args.dropout, embedding_dropout=args.embedding_dropout)
    elif args.model_type == 'T5':
        model = custom_T5(isPreTrain=args.isPreTrain, variational_mode=args.variational_mode, d_latent=args.d_latent,
                     emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing)
    elif args.model_type == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
    model = model.to(device)
    
    # 2) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        checkpoint = torch.load(os.path.join(args.model_save_path, 'checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

    write_log(logger, 'Traing start!')

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time_e = time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                write_log(logger, 'Validation start...')
                val_loss = 0
                val_acc = 0
                model.eval()
            for i, batch_iter in enumerate(tqdm(dataloader_dict[phase], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

                # Optimizer setting
                optimizer.zero_grad(set_to_none=True)

                # Input, output setting
                src_sequence = batch_iter[0]
                src_att = batch_iter[1]
                trg_label = batch_iter[2]

                src_sequence = src_sequence.to(device, non_blocking=True)
                src_att = src_att.to(device, non_blocking=True)
                trg_label = trg_label.to(device, non_blocking=True)

                # Train
                if phase == 'train':

                    with autocast():
                        predicted = model(input_ids=src_sequence, attention_mask=src_att)
                        loss = criterion(predicted, trg_label)
                        # total_loss = loss + dist_loss
                        total_loss = loss

                    scaler.scale(total_loss).backward()
                    if args.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    if args.scheduler in ['constant', 'warmup']:
                        scheduler.step()
                    if args.scheduler == 'reduce_train':
                        scheduler.step(loss)

                    # Print loss value only training
                    if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                        acc = (predicted.max(dim=1)[1] == trg_label).sum() / len(trg_label)
                        iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.3f | train_acc:%03.2f%% | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                            (epoch, i, len(dataloader_dict['train']), 
                            total_loss.item(), acc*100, optimizer.param_groups[0]['lr'], 
                            (time() - start_time_e) / 60)
                        write_log(logger, iter_log)
                        freq = 0
                    freq += 1

                # Validation
                if phase == 'valid':
                    with torch.no_grad():
                        predicted = model(input_ids=src_sequence, attention_mask=src_att)
                        loss = criterion(predicted, trg_label)
                        # total_loss = loss + dist_loss
                        total_loss = loss
                    val_loss += total_loss.item()
                    val_acc += (predicted.max(dim=1)[1] == trg_label).sum() / len(trg_label)

            if phase == 'valid':

                if args.scheduler == 'reduce_valid':
                    scheduler.step(val_loss)
                if args.scheduler == 'lambda':
                    scheduler.step()

                val_loss /= len(dataloader_dict[phase])
                val_acc /= len(dataloader_dict[phase])
                write_log(logger, 'Validation Loss: %3.3f' % val_loss)
                write_log(logger, 'Validation Accuracy: %3.2f%%' % (val_acc * 100))
                save_path = os.path.join(args.model_save_path, args.task, args.data_name, args.tokenizer)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_file_name = os.path.join(save_path, 
                                              f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
                if val_acc > best_val_acc:
                    write_log(logger, 'Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict()
                    }, save_file_name)
                    best_val_acc = val_acc
                    best_epoch = epoch
                else:
                    else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc.item()*100, 2)})% is better...'
                    write_log(logger, else_log)

    # 3) Print results
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_val_acc.item(), 2)}')