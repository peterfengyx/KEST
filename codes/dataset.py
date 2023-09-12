from datasets import load_dataset
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, Union, Any

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
# ------------------------------------------------------------
@dataclass
class DataCollatorForCVAE:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        padding_strategy, _, max_length, _ = self.tokenizer._get_padding_truncation_strategies(
            padding=self.padding, max_length=self.max_length, verbose=False
        )
        
        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            return BatchEncoding(features, tensor_type=return_tensors)
        
        batch_size = len(features)

        need_max_length = False
        if padding_strategy == PaddingStrategy.LONGEST:
            need_max_length = True,
            padding_strategy = PaddingStrategy.MAX_LENGTH

        encoded = {}
        def pad_seqs(seqs, mode):
            padded_seqs = []
            if need_max_length:
                max_length = max(len(seq) for seq in seqs)
            if mode=='body':
                body_len=5
                pad_idx = self.tokenizer.mask_token_id
                for seq in seqs:
                    difference = max_length - body_len
                    padded_seqs.append(seq[:body_len] + [pad_idx] * difference)
                return padded_seqs
            
            if mode == 'attn_mask':
                pad_idx = 0
            elif mode == 'type_ids':
                pad_idx = seqs[0][-1]
            elif mode == "input_ids":
                pad_idx = self.tokenizer.pad_token_id
            
            for seq in seqs:
                difference = max_length - len(seq)
                padded_seqs.append(seq + [pad_idx] * difference)

            return padded_seqs
        
        
        body_ids = [ dic['body_ids'] for dic in features]
        title_ids = [ dic['title_ids'] for dic in features]
        body_mask = [ dic['body_attn_mask'] for dic in features]
        title_mask = [ dic['title_attn_mask'] for dic in features]
        body_type = [ dic['body_token_type_ids'] for dic in features]
        title_type = [ dic['title_token_type_ids'] for dic in features]
        
        seq_ids = [ dic['seq_ids'] for dic in features]
        seq_mask = [ dic['seq_attn_mask'] for dic in features]
        seq_type = [ dic['seq_token_type_ids'] for dic in features]
        
        encoded['title_ids'] = pad_seqs(title_ids, 'input_ids')
        encoded['seq_ids'] = pad_seqs(seq_ids, 'input_ids')
        
        encoded['title_attn_mask'] = pad_seqs(title_mask, 'attn_mask')
        encoded['seq_attn_mask'] = pad_seqs(seq_mask, 'attn_mask')
        
        encoded['title_token_type_ids'] = pad_seqs(title_type, 'type_ids')
        encoded['seq_token_type_ids'] = pad_seqs(seq_type, 'type_ids')
        
        encoded['body_attn_mask']=encoded['seq_attn_mask']
        encoded['body_token_type_ids']=encoded['seq_token_type_ids']
        encoded['body_ids']=pad_seqs(encoded['seq_ids'],'body')
        
        encoded['cl_labels'] = [dic['cl_label'] for dic in features]

        return BatchEncoding(encoded, tensor_type=return_tensors)

import re
def preprocess(s):
    r1=re.compile("#[\d]+;")
    s1=re.sub(r1, convert_entity,s)
    s1=s1.replace('\\',' ')
    s1=s1.replace('quot;','\"')
    s1=s1.replace('&lt;','<')
    s1=s1.replace('&gt;','>')
    s1=s1.replace('amp;','&')
    s1=re.sub('<.*?>','',s1)
    s1=s1.strip()
    return s1

def convert_entity(m) -> str:
    """
        re.sub
    """
    return chr(int(m.group(0)[1:-1]))

def build_dataset(train_args, extra_args, tokenizer):
    # Build or load gpt2 tokenizer
    
    tokenizer.model_input_names = ['cl_labels',
        'body_ids', 'body_attn_mask', 'body_token_type_ids',
        'title_ids', 'title_attn_mask', 'title_token_type_ids',
        'seq_ids', 'seq_attn_mask', 'seq_token_type_ids'
    ]

    # Load raw datasets
    print ("load raw datasets")
    raw_datasets = load_dataset(path=extra_args.corpus_dir,
        data_files={"train":extra_args.training_data})
    raw_datasets=raw_datasets.shuffle(42)
    def clean(example):
        example['text']=preprocess(example['text'])
        return example
    
    raw_datasets = raw_datasets.map(clean)
    
    split_id=int(len(raw_datasets['train'])*0.9)
    
    raw_trainset = raw_datasets['train'].select(range(split_id))
    raw_validset = raw_datasets['train'].select(range(split_id,len(raw_datasets['train'])))
    

    def preprocess_function(examples):
        titles = examples['text']
        
        title_inputs = tokenizer(titles, max_length=extra_args.max_title_len-1,
            truncation=True, return_token_type_ids=True, return_attention_mask=True)
        
        model_inputs = {}
        model_inputs['cl_label'] = examples['label']
        
        
        model_inputs['title_ids'] = title_inputs['input_ids']
        model_inputs['title_attn_mask'] = title_inputs['attention_mask']
        model_inputs['title_token_type_ids'] = [list(np.array(seq)+1) for seq in title_inputs['token_type_ids']]
        
        model_inputs['body_ids'] = model_inputs['title_ids']
        model_inputs['body_attn_mask'] = model_inputs['title_attn_mask']
        model_inputs['body_token_type_ids'] = model_inputs['title_token_type_ids']
        
        model_inputs['seq_ids'] = model_inputs['title_ids']
        model_inputs['seq_attn_mask'] = model_inputs['title_attn_mask']
        model_inputs['seq_token_type_ids'] = model_inputs['title_token_type_ids']
        return model_inputs
    
    
    # Preprocessing the datasets.
    training_set, validation_set, unlabelled_set = None, None, None
    with train_args.main_process_first(desc="tokenize sentences"):
        if train_args.do_train:
            training_set = raw_trainset.map(
                preprocess_function,
                batched=True,
                batch_size=extra_args.preprocessing_bsize,
                num_proc=extra_args.preprocessing_num_workers,
                remove_columns=['label', 'text'],
                load_from_cache_file= not extra_args.overwrite_cache,
                cache_file_name=extra_args.train_cache_file,
                desc=f"Running tokenizer on the training dataset",
            )      
    
    with train_args.main_process_first(desc="tokenize sentences"):
        if train_args.do_eval:
            validation_set = raw_validset.map(
                preprocess_function,
                batched=True,
                batch_size=extra_args.preprocessing_bsize,
                num_proc=extra_args.preprocessing_num_workers,
                remove_columns=['label', 'text'],
                load_from_cache_file= not extra_args.overwrite_cache,
                cache_file_name=extra_args.valid_cache_file,
                desc=f"Running tokenizer on the validation dataset",
            )

    return training_set, validation_set, unlabelled_set