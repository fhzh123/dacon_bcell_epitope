# Import modules
import os
import gc
import h5py
import pickle
import logging
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
# Import PyTorch
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Import Huggingface
from transformers import BertTokenizerFast, BartTokenizerFast, T5TokenizerFast
# Import custom modules
from model.dataset import Seq2LabelTestDataset
from model.custom_transformer.transformer import Transformer
from model.custom_plm.T5 import custom_T5
from model.custom_plm.bart import custom_Bart
from utils import TqdmLoggingHandler, write_log

def testing(args):

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

    write_log(logger, 'Start testing!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    # Path setting
    save_path = os.path.join(args.preprocess_path, args.tokenizer)
    save_name = f'vocab_{args.vocab_size}_processed.pkl'

    with open(os.path.join(save_path, 'test_'+save_name), 'rb') as f:
        data_ = pickle.load(f)
        test_epitope_input_ids = data_['test_epitope_input_ids']
        test_epitope_attention_mask = data_['test_epitope_attention_mask']
        test_antigen_input_ids = data_['test_antigen_input_ids']
        test_antigen_attention_mask = data_['test_antigen_attention_mask']
        del data_

    gc.enable()
    write_log(logger, "Finished loading data!")

    #===================================#
    #===========Test setting============#
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

    # lode model
    save_path = os.path.join(args.model_save_path, args.tokenizer)
    save_file_name = os.path.join(save_path, 
                                    f'checkpoint_vocab_{args.vocab_size}.pth.tar')
    model.load_state_dict(torch.load(save_file_name)['model'])
    model = model.eval()
    write_log(logger, f'Loaded model from {save_file_name}!')

    # 2) Dataloader setting
    test_dataset = Seq2LabelTestDataset(src_list=test_epitope_input_ids, src_att_list=test_epitope_attention_mask,
                                        src_max_len=args.src_max_len)
    test_dataloader = DataLoader(test_dataset, drop_last=False, batch_size=args.test_batch_size, shuffle=False,
                                pin_memory=True, num_workers=args.num_workers)
    write_log(logger, f"Total number of trainingsets  iterations - {len(test_dataset)}, {len(test_dataloader)}")

    #===================================#
    #============Inference==============#
    #===================================#

    predicted_list = list()
    submission = pd.read_csv('/mnt/md0/dataset/dacon/bcell/sample_submission.csv')

    with torch.no_grad():
        for i, batch_iter in enumerate(tqdm(test_dataloader, bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            # Input, output setting
            src_sequence = batch_iter[0]
            src_att = batch_iter[1]

            src_sequence = src_sequence.to(device, non_blocking=True)
            src_att = src_att.to(device, non_blocking=True)

            predicted = model(input_ids=src_sequence, attention_mask=src_att)

            predicted_max = predicted.max(dim=1)[1].tolist()
            predicted_list.extend(predicted_max)

    #===================================#
    #=============Saving================#
    #===================================#
    
    # Make pandas dataframe with source_sentences, predicted_sentences, target_sentences
    df = pd.DataFrame(
        {
            'id': submission['id'].tolist(), 
            'label': predicted_list
        }
    )
    df.to_csv(args.result_path, index=False)