import torch
import torch.nn.functional as F


from datasets import load_dataset
from datasets import Dataset
from graphs import GPT2VAE
from config import extra_args, GPT2Config

import utils

import json
from gen_utils import GenerationTool
from gen_eval import evaluate_dist_scores, evaluate_cls, evaluate_ppl
import numpy as np
from diversity_eval.evaluate import Evaluator
from evaluate import build_dataset


pplm_prompt=["In summary", "This essay discusses", "Views on", "The connection", 
             "Foundational to this is", "To review", "In brief,", "An illustration of",
             "Furthermore,", "The central theme", "To conclude,", "The key aspect", 
             "Prior to this", "Emphasized are", "To summarize", "The relationship", 
             "More importantly,", "It has been shown", "The issue focused on", 
             "In this essay"]
def get_inps(body, title, label, tokenizer, bos_token_idx):

    body_inputs = tokenizer(body, max_length=extra_args.max_body_len-1,
        truncation=True, return_token_type_ids=True, return_attention_mask=True,
        return_tensors='pt')
        
    title_inputs = tokenizer(title, max_length=extra_args.max_title_len-1,
        truncation=True, return_token_type_ids=True, return_attention_mask=True,
        return_tensors='pt')
        
    model_inputs = {}
        
    model_inputs['body_ids'] = torch.cat(
        [body_inputs['input_ids'][:, 0:-1], torch.tensor(bos_token_idx).view(1,1)], dim=-1)
    
    model_inputs['body_attn_mask'] = body_inputs['attention_mask']
    model_inputs['body_token_type_ids'] = body_inputs['token_type_ids']
        
    model_inputs['title_ids'] = title_inputs['input_ids']
    model_inputs['title_attn_mask'] = title_inputs['attention_mask']
    model_inputs['title_token_type_ids'] = title_inputs['token_type_ids'] + 1
    model_inputs['cl_labels'] =torch.tensor(label, dtype=torch.int).view(1,)
        
    return model_inputs
        



def generate_file(infile, outfile, ckpt, method="pplm", do_eval=False):
    tokenizer = utils.loadTokenizer(extra_args.tokenizer_dir)
    config = GPT2Config(
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    )
    model = GPT2VAE(config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.from_pretrained("../ckpts/checkpoint-" + ckpt, config=config)
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()    
    generator = GenerationTool()
    
    bos_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    bos_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    eos_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    max_length = extra_args.max_body_len+extra_args.max_title_len+2
    max_len=max_length
    max_len=40
    min_len=10
    torch.manual_seed(100)
    
    fout = open(outfile, 'w')    
    
    num_per_class=15
    num_type=4
    #for i, d in enumerate(data[0:100]):
    all_titles=[]
    cls_logits=[]
    cls_labels=[]
    ppl_scores=0
    eval_set={'text':[],'labels':[]}
    
    
    if method=="pplm":
        prompt_raw=pplm_prompt
    elif method=="imdb":
        with open("imdb_prompt.txt",'r') as f:
            prompt_raw=eval(f.read())
    else: #all
        with open("imdb_prompt.txt",'r') as f:
            prompt_raw=eval(f.read())
        prompt_raw.extend(pplm_prompt)
    for epoch in range(len(prompt_raw)):
        
        titles=[]
        p_labels=[]
        max_len_gen=0
        for j in range(num_type):
            prompt=tokenizer(prompt_raw[epoch])
            prompt["input_ids"]=prompt["input_ids"][:-1]
            prompt["attention_mask"]=prompt["attention_mask"][:-1]
            #for sentiment generation
            #beginer=torch.tensor(prompt["input_ids"],dtype=torch.long, device=model.device).tile(num_per_class,1)
            #att=torch.tensor(prompt["attention_mask"],dtype=torch.long, device=model.device).tile(num_per_class,1)
            #token_type=att
            #for topic generation
            beginer=torch.ones((num_per_class,1),dtype=torch.long, device=model.device)*bos_idx
            att=torch.ones((num_per_class,1),dtype=torch.long, device=model.device)
            token_type=torch.ones((num_per_class,1),dtype=torch.long, device=model.device)
            pseudo_label=torch.ones((num_per_class),dtype=torch.long, device=model.device)*j
            z=model.label_embed(pseudo_label)
            
            with torch.no_grad():
                outs = generator.generate(
                        model, beginer, z,
                        attention_mask=att,
                        token_type_ids=token_type,
                        bos_token_id=bos_idx,
                        do_sample=True, top_p=0.9, temperature=1, length_penalty=1.0,
                        repetition_penalty=1.0, no_repeat_ngram_size=4,
                        use_cache=True, max_length=max_len, min_length=min_len,
                        pad_token_id=pad_idx, eos_token_id=eos_idx,
                        output_scores=False, return_dict_in_generate=True)
            
            for i in range(num_per_class):
                seq = outs['sequences'][i]
                seq=seq.tolist()
                if eos_idx in seq:
                    seq = seq[0:seq.index(eos_idx)]
                if len(seq)>max_len_gen:
                    max_len_gen=len(seq)
                titles.append(torch.tensor(seq,dtype=torch.long, device=model.device))
                all_titles.append(seq)
            p_labels.append(pseudo_label)
            decoded_seqs = tokenizer.batch_decode(outs['sequences'])
            for i in range(num_per_class):
                decoded_seqs[i] = decoded_seqs[i][5:].strip()  #remove [CLS]
                if eos_token in decoded_seqs[i]:
                    decoded_seqs[i] = decoded_seqs[i][0:decoded_seqs[i].index(eos_token)]
                
                if i==0:
                    #world, sport, business, sci/tech
                    if j==0:
                        fout.write("World: "+decoded_seqs[i]+ "\n")
                    elif j==1:
                        fout.write("Sport: "+decoded_seqs[i]+ "\n")
                    elif j==2:
                        fout.write("Business: "+decoded_seqs[i]+ "\n")
                    elif j==3:
                        fout.write("Sci/tech: "+decoded_seqs[i]+ "\n")
                    fout.write("\n")
                eval_set['text'].append(decoded_seqs[i])
                eval_set['labels'].append(j)
            fout.flush()
        
        if epoch % 1 == 0:
            print(epoch,"/500")
    fout.close()
    ds = Dataset.from_dict(eval_set)
    txt = ds['text']
    ds.to_json("../data/generation.json")
    if do_eval:
        dist_scores=evaluate_dist_scores(all_titles)
        cls_results=evaluate_cls(ds)
        ppl_scores=evaluate_ppl(ds)
        print(dist_scores)
        print(cls_results)
        print(ppl_scores)
        evaluator = Evaluator()
        evaluator.evaluate_file(txt)
    
    

def main():
    generate_file("../corpus/news_valid.json",
        "../outs/topic_dvaecls30gen1.txt", "324000", method="pplm", do_eval=True)


if __name__ == "__main__":
    main()
