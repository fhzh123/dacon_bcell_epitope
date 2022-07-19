import os
import time
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
import sentencepiece as spm
from utils import TqdmLoggingHandler, write_log
from sklearn.model_selection import train_test_split

def preprocessing(args):

    start_time = time.time()

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, 'Start preprocessing!')

    train = pd.read_csv('/mnt/md0/dataset/dacon/bcell/train.csv')
    train, valid = train_test_split(train, test_size=0.1)
    test = pd.read_csv('/mnt/md0/dataset/dacon/bcell/test.csv')

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    if args.tokenizer == 'alpha_map':
        alpha_map = {
                        'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6,
                        'G':7, 'H':8, 'I':9, 'J':10, 'K':11, 'L':12,
                        'M':13, 'N':14, 'O':15, 'P':16, 'Q':17, 'R':18,
                        'S':19, 'T':20, 'U':21, 'V':22, 'W':23, 'X':24,
                        'Y':25, 'Z':26
                    }

        processed_epitope_seq_train = [list(x) for x in train['epitope_seq']]
        processed_antigen_seq_train = [list(x) for x in train['antigen_seq']]
        encoded_epitope_seq_train = [[alpha_map[i] for i in x] for x in processed_epitope_seq_train]
        encoded_antigen_seq_train = [[alpha_map[i] for i in x] for x in processed_antigen_seq_train]
        train_epitope_att_list = [[1 for _ in x] for x in processed_epitope_seq_train]
        train_antigen_att_list = [[1 for _ in x] for x in processed_antigen_seq_train]
        
        processed_epitope_seq_valid = [list(x) for x in valid['epitope_seq']]
        processed_antigen_seq_valid = [list(x) for x in valid['antigen_seq']]
        encoded_epitope_seq_valid = [[alpha_map[i] for i in x] for x in processed_epitope_seq_valid]
        encoded_antigen_seq_valid = [[alpha_map[i] for i in x] for x in processed_antigen_seq_valid]
        valid_epitope_att_list = [[1 for _ in x] for x in processed_epitope_seq_valid]
        valid_antigen_att_list = [[1 for _ in x] for x in processed_antigen_seq_train]
        
        processed_epitope_seq_test = [list(x) for x in test['epitope_seq']]
        processed_antigen_seq_test = [list(x) for x in test['antigen_seq']]
        encoded_epitope_seq_test = [[alpha_map[i] for i in x] for x in processed_epitope_seq_test]
        encoded_antigen_seq_test = [[alpha_map[i] for i in x] for x in processed_antigen_seq_test]
        test_epitope_att_list = [[1 for _ in x] for x in processed_epitope_seq_test]
        test_antigen_att_list = [[1 for _ in x] for x in processed_antigen_seq_train]

        # train['epitope_len'] = train['epitope_seq'].apply(len)
        # train['start_position2'] = train['start_position'] - 1 - ((512 - train['epitope_len']) / 2).apply(math.floor)
        # train['start_position2'] = train['start_position2'].clip(lower=0).apply(int)
        # train['end_position2'] = (train['start_position2'] + 300).apply(int)

        # train['processed_antigen_seq'] = train.apply(lambda x: x['antigen_seq'][x['start_position2']:x['end_position2']], 1)
        # train['processed_antigen_start_position'] = (train['start_position'] - train['start_position2'] - 1).apply(int)
        # train['processed_antigen_end_position'] = (train['end_position'] - train['start_position2']).apply(int)

    elif 'spm' in args.tokenizer:

        # Make text to train vocab
        preprocess_path_spm = os.path.join(args.preprocess_path, args.tokenizer)
        with open(f'{preprocess_path_spm}/bcell.txt', 'w') as f:
            for text in train['antigen_seq']:
                f.write(f'{text}\n')

        spm.SentencePieceProcessor()
        spm.SentencePieceTrainer.Train(
            f'--input={preprocess_path_spm}/bcell.txt --model_type={args.tokenizer[4:]} '
            f'--model_prefix={preprocess_path_spm}/m_{args.tokenizer[4:]}_{args.vocab_size} '
            f'--vocab_size={args.vocab_size} --character_coverage={args.character_coverage} '
            f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
            f'--split_by_whitespace=true')

        vocab_list = list()
        with open(f'{preprocess_path_spm}/m_{args.tokenizer[4:]}_{args.vocab_size}.vocab') as f:
            for line in f:
                vocab_list.append(line[:-1].split('\t')[0])

        word2id = {w: i for i, w in enumerate(vocab_list)}
        spm_src = spm.SentencePieceProcessor()
        spm_src.Load(f'{preprocess_path_spm}/m_{args.tokenizer[4:]}_{args.vocab_size}.model')

        # Encoding
        encoded_epitope_seq_train = list(
            [args.bos_id] + spm_src.encode(
                                text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
            [args.eos_id] for text in train['epitope_seq']
        )
        encoded_epitope_seq_valid = list(
            [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in valid['epitope_seq']
        )
        encoded_epitope_seq_test = list(
            [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in test['epitope_seq']
        )

        # Encoding
        encoded_antigen_seq_train = list(
            [args.bos_id] + spm_src.encode(
                                text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
            [args.eos_id] for text in train['antigen_seq']
        )
        encoded_antigen_seq_valid = list(
            [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in valid['antigen_seq']
        )
        encoded_antigen_seq_test = list(
            [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in test['antigen_seq']
        )

        # Attention mask encoding
        train_epitope_att_list = [[1 for _ in x] for x in encoded_epitope_seq_train]
        train_antigen_att_list = [[1 for _ in x] for x in encoded_antigen_seq_train]
        valid_epitope_att_list = [[1 for _ in x] for x in encoded_epitope_seq_valid]
        valid_antigen_att_list = [[1 for _ in x] for x in encoded_antigen_seq_valid]
        test_epitope_att_list = [[1 for _ in x] for x in encoded_epitope_seq_test]
        test_antigen_att_list = [[1 for _ in x] for x in encoded_antigen_seq_test]

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path setting
    save_path = os.path.join(args.preprocess_path, args.tokenizer)
    save_name = f'vocab_{args.vocab_size}_processed.pkl'

    with open(os.path.join(save_path, save_name), 'wb') as f:
        pickle.dump({
            'train_epitope_input_ids' : encoded_epitope_seq_train,
            'train_epitope_attention_mask' : train_epitope_att_list,
            'train_antigen_input_ids' : encoded_antigen_seq_train,
            'train_antigen_attention_mask' : train_antigen_att_list,
            'valid_epitope_input_ids' : encoded_epitope_seq_valid,
            'valid_epitope_attention_mask' : valid_epitope_att_list,
            'valid_antigen_input_ids' : encoded_antigen_seq_valid,
            'valid_antigen_attention_mask' : valid_antigen_att_list,
            'train_label' : train['label'].tolist(),
            'valid_label' : valid['label'].tolist(),
        }, f)

    with open(os.path.join(save_path, 'test_'+save_name), 'wb') as f:
        pickle.dump({
            'test_epitope_input_ids' : encoded_epitope_seq_test,
            'test_epitope_attention_mask' : test_epitope_att_list,
            'test_antigen_input_ids' : encoded_antigen_seq_test,
            'test_antigen_attention_mask' : test_antigen_att_list,
        }, f)


    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')