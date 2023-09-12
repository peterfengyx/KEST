import torch
import torch.nn.functional as F


from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from graphs import GPT2VAE
from config import extra_args, GPT2Config
from dataset import DataCollatorForCVAE
from trainer import NLGTrainer
import utils
import tqdm
from torch.utils.data import Dataset,DataLoader

train_args = Seq2SeqTrainingArguments(
    generation_max_length=extra_args.max_body_len + extra_args.max_title_len + 2,
    output_dir="../ckpts/",
    seed=506,
    do_eval=True,
    per_device_eval_batch_size=32,
    local_rank=-1,
    xpu_backend='ncll',
    no_cuda=False,
    remove_unused_columns=False,
)

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

def build_dataset(tokenizer):
    # Build or load gpt2 tokenizer
    tokenizer.model_input_names = ['cl_labels',
        'body_ids', 'body_attn_mask', 'body_token_type_ids',
        'title_ids', 'title_attn_mask', 'title_token_type_ids',
        'seq_ids', 'seq_attn_mask', 'seq_token_type_ids'
    ]

    # Load raw datasets
    print ("load raw datasets")
    raw_datasets = load_dataset(path=extra_args.corpus_dir,
        data_files={'test': extra_args.testing_data})
    '''
    raw_datasets = load_dataset("ag_news")
    '''
    raw_datasets = raw_datasets.shuffle(seed=42)
    def clean(example):
        example['text']=preprocess(example['text'])
        return example
    
    raw_datasets = raw_datasets.map(clean)
    raw_testset = raw_datasets['test']

    #l2r_sep_idx = tokenizer.vocab['[L2R_SEP]']
    
    
    def preprocess_function(examples):
        # body_ids, body_attn_mask, body_token_type_ids
        # title_ids, title_attn_mask, title_token_type_ids
        # cl_labels
        # assume we use bert tokenizer
        titles = examples['text']
        #bodies = examples['text']
        '''
        bodies = examples['body']
        
        body_inputs = tokenizer(bodies, max_length=extra_args.max_body_len-1,
            truncation=True, return_token_type_ids=True, return_attention_mask=True)
        '''
        title_inputs = tokenizer(titles, max_length=extra_args.max_title_len-1,
            truncation=True, return_token_type_ids=True, return_attention_mask=True)
        
        model_inputs = {}
        model_inputs['cl_label'] = examples['label']
        '''
        model_inputs['body_ids'] = [ body[0:-1] + [l2r_sep_idx] for body in body_inputs['input_ids'] ]
        model_inputs['body_attn_mask'] = body_inputs['attention_mask']
        model_inputs['body_token_type_ids'] = body_inputs['token_type_ids']
        '''
        
        
        model_inputs['title_ids'] = title_inputs['input_ids']
        model_inputs['title_attn_mask'] = title_inputs['attention_mask']
        model_inputs['title_token_type_ids'] = [list(np.array(seq)+1) for seq in title_inputs['token_type_ids']]
        
        '''
        seq_ids = [ body[0:-1] + [l2r_sep_idx] + title[1:] for body, title in zip(body_inputs['input_ids'], title_inputs['input_ids'])]
        seq_mask = [ body + title[1:] for body, title in zip(body_inputs['attention_mask'], title_inputs['attention_mask'])]
        seq_type_ids = [ [0] * len(body) + [1]*len(title[1:]) for body, title in zip(body_inputs['input_ids'], title_inputs['input_ids'])]
        
        model_inputs['seq_ids'] = seq_ids
        model_inputs['seq_attn_mask'] = seq_mask
        model_inputs['seq_token_type_ids'] = seq_type_ids
        '''
        model_inputs['body_ids'] = model_inputs['title_ids']
        model_inputs['body_attn_mask'] = model_inputs['title_attn_mask']
        model_inputs['body_token_type_ids'] = model_inputs['title_token_type_ids']
        
        model_inputs['seq_ids'] = model_inputs['title_ids']
        model_inputs['seq_attn_mask'] = model_inputs['title_attn_mask']
        model_inputs['seq_token_type_ids'] = model_inputs['title_token_type_ids']
        return model_inputs
    
    
    # Preprocessing the datasets.
    with train_args.main_process_first(desc="tokenize sentences"):
        testing_set = raw_testset.map(
            preprocess_function,
            batched=True,
            batch_size=extra_args.preprocessing_bsize,
            num_proc=extra_args.preprocessing_num_workers,
            remove_columns=['label', 'text'],#, 'title'],
            load_from_cache_file= not extra_args.overwrite_cache,
            cache_file_name=extra_args.test_cache_file,
            desc=f"Running tokenizer on the validation dataset",
        )
    return testing_set


# ------------------------------------------
def do_evaluate(model, tokenizer, testing_set):
    
    data_collator = DataCollatorForCVAE(tokenizer=tokenizer, padding='longest')
    '''
    trainer = NLGTrainer(
        model=model, args=train_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
        
    results = trainer.predict(testing_set, metric_key_prefix="predict")
    #print(results.metrics['test_loss'])
    gen_ppl=np.exp(results.metrics['predict_loss'])
    
    raw_preds = torch.tensor(results.predictions, device=model.device)
    probs = F.softmax(raw_preds, dim=-1)
    
    preds = torch.argmax(probs, dim=-1).detach().cpu().numpy()
    pred_probs = probs[:, -1].detach().cpu().numpy()
    labels = results.label_ids
    #print(results.predictions)  
    #print(results.label_ids)  
    #print(len(results.label_ids))  

    #print(preds)
    #print(pred_probs)
    
    #print(len(preds))
    '''
    #TODO:
    #test_loader=DataLoader(testing_set,batch_size=4,collate_fn=data_collator)
    device="cuda" 
    model.to(device)
    model.eval()
    
    gen_loss=0
    logits=[]
    label=[]
    num=0
    #gen_ppl, elbo, nll, kl=utils.calc_iwnll(model,test_loader,ns=10)
    gen_ppl, elbo, nll, kl=0,0,0,0
    test_loader=DataLoader(testing_set,batch_size=16,collate_fn=data_collator)
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        label.append(batch["cl_labels"])
        with torch.no_grad():
            outs = model(batch, is_evaluate=True)#, n_sample=10)
        logits.append(outs['cl_logits'])
        
        #print(outs['gen_loss'])
        gen_loss+=float(torch.mean(outs['gen_loss']))
        
        num+=1
        if num%10==0:
            print(num)
    
    gen_ppl=np.exp(gen_loss/num)
    raw_preds=torch.cat(logits,dim=0)
    probs = F.softmax(raw_preds, dim=-1)
    preds = torch.argmax(probs, dim=-1).detach().cpu().numpy()
    pred_probs = probs.detach().cpu().numpy()
    labels = torch.cat(label,dim=0).detach().cpu().numpy()
    
    
    accu = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    auc = roc_auc_score(labels, pred_probs,multi_class="ovr")
    prec = precision_score(labels, preds,average="macro")
    recall = recall_score(labels, preds,average="macro")

    return  accu, f1, auc, prec, recall, gen_ppl, elbo, nll, kl

def evaluate(dir_path, ckpt_vec):
    
    tokenizer = utils.loadTokenizer(extra_args.tokenizer_dir)
    testing_set = build_dataset(tokenizer)
    
    config = GPT2Config(
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    )
    
    ckpt_path = dir_path + ckpt_vec
    model = GPT2VAE(config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.from_pretrained(ckpt_path, config=config)


    accu, f1, auc, prec, recall, gen_ppl,elbo, nll, kl = do_evaluate(model, tokenizer, testing_set)

    infostr = "accu: %.3f, f1: %.3f, auc: %.3f, prec: %.3f, reca: %.3f" % (
                np.round(accu, 3), np.round(f1, 3), np.round(auc, 3),
                np.round(prec, 3), np.round(recall, 3), 
            )
    
    print(infostr)
    print("gen ppl:", gen_ppl)
    print("elbo: %.3f, nll: %.3f, kl: %.3f" %(
        np.round(elbo, 3),np.round(nll, 3),np.round(kl, 3)
        ))


def main():
    evaluate("../ckpts/checkpoint-", "6480")
    #evaluate("../45008-base","")

if __name__ == "__main__":
    main()
