import os
import time
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
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
    submission = pd.read_csv('/mnt/md0/dataset/dacon/bcell/sample_submission.csv')

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    alpha_map = {
                    'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6,
                    'G':7, 'H':8, 'I':9, 'J':10, 'K':11, 'L':12,
                    'M':13, 'N':14, 'O':15, 'P':16, 'Q':17, 'R':18,
                    'S':19, 'T':20, 'U':21, 'V':22, 'W':23, 'X':24,
                    'Y':25, 'Z':26
                }

    processed_epitope_seq_train = [list(x) for x in train['epitope_seq']]
    encoded_epitope_seq_train = [[alpha_map[i] for i in x] for x in processed_epitope_seq_train]
    train_att_list = [[1 for _ in x] for x in processed_epitope_seq_train]
    processed_epitope_seq_valid = [list(x) for x in valid['epitope_seq']]
    encoded_epitope_seq_valid = [[alpha_map[i] for i in x] for x in processed_epitope_seq_valid]
    valid_att_list = [[1 for _ in x] for x in processed_epitope_seq_valid]
    processed_epitope_seq_test = [list(x) for x in test['epitope_seq']]
    encoded_epitope_seq_test = [[alpha_map[i] for i in x] for x in processed_epitope_seq_test]
    test_att_list = [[1 for _ in x] for x in processed_epitope_seq_test]

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    if not os.path.exists(args.preprocess_path):
        os.mkdir(args.preprocess_path)

    save_name = f'processed.pkl'

    with open(os.path.join(args.preprocess_path, save_name), 'wb') as f:
        pickle.dump({
            'train_src_input_ids' : encoded_epitope_seq_train,
            'train_src_attention_mask' : train_att_list,
            'valid_src_input_ids' : encoded_epitope_seq_valid,
            'valid_src_attention_mask' : valid_att_list,
            'train_label' : train['label'].tolist(),
            'valid_label' : valid['label'].tolist(),
        }, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')