# coding=utf-8
from numpy import empty
import torch
from torch import nn
import torch.nn.functional as F
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_outputs import CausalLMOutputWithPast
import math

from layers import (
    GPT2PreTrainedModel,
    GPT2Model,
    MLP,
    LMHead,
    LossWrapper,
    GPT2VAEOutputWithPast
)

from config import extra_args
from utils import log_sum_exp

class GPT2VAE(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = LMHead(config)
        #self.ag_decoder = GPT2Model(config)
    
        self.label_embed = nn.Embedding(config.n_label, config.n_label_embd)
        
        self.cl_layer = nn.Sequential(
            nn.Linear(config.n_embd, 256), nn.Tanh(),
            nn.Dropout(config.cl_pdrop),
            nn.Linear(256, 64), nn.Tanh(),
            nn.Dropout(config.cl_pdrop),
            nn.Linear(64, config.n_label),
        )

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.config = config
        
        self.loss_fct = LossWrapper(config.pad_token_id, config.kl_lambda)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_seq_rep(self, inp_ids, attn_mask, type_ids,input_embeddings=None):
        
        outputs = self.transformer(
            inp_ids,
            attention_mask=attn_mask,
            token_type_ids=type_ids,
            output_hidden_states=False,
            return_dict=True,
            with_causal_mask=False,
            input_embeddings=input_embeddings
        )
        return outputs.last_hidden_state[:, 0, :] # (B, H), cls as feature


    def get_gen_outputs(self, seq_ids, seq_attn_mask,
        seq_type_ids, latent_variable,input_embeddings=None, causal_mask=True):
        
        if causal_mask: #auto-regressive generation
            # q(x|z,y,c)
            #hidden_states = self.ag_decoder(
            hidden_states = self.transformer(
                input_ids=seq_ids,
                attention_mask=seq_attn_mask,
                token_type_ids=seq_type_ids,
                return_dict=True,
                with_causal_mask=causal_mask,
                latent_variable=latent_variable,
                input_embeddings=input_embeddings
            ).last_hidden_state
        else: #non-ar
            hidden_states = self.transformer(
                input_ids=seq_ids,
                attention_mask=seq_attn_mask,
                token_type_ids=seq_type_ids,
                return_dict=True,
                with_causal_mask=causal_mask,
                latent_variable=latent_variable,
                input_embeddings=input_embeddings
            ).last_hidden_state
        

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.device)

        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits
    
    def logit2prob(self,logit,temp=1.0):
        prob=F.softmax(logit/temp,dim=-1)
        return prob
    
    #def get_logit(self.input_dic):
        
    
    def sample_z(self, rep, z_type, task_type):
        # z_type: prior or post
        # task_type: gen or cl
        assert z_type in ['prior', 'post']
        assert task_type in ['gen', 'cl']

        if z_type == 'post':
            latent = self.posteriori(rep)
        elif z_type == 'prior' and task_type == 'gen':
            latent = self.gen_prior(rep)
        elif z_type == 'prior' and task_type == 'cl':
            latent = self.cl_prior(rep)

        (mu, log_sigma) = latent.split(self.config.n_latent, dim=-1)
        z = mu + log_sigma.exp() * torch.randn_like(mu)
        return z, mu, log_sigma

    def forward(self, input_dic, is_evaluate=False, n_sample=1): 
        
        if is_evaluate:
            return self.do_classification(input_dic, nsamples=n_sample)
        
        
        if 'past_emb' in input_dic:
            inp_emb=input_dic['past_emb']
            inp_msk_emb=input_dic['past_emb_nag']
            
            xc_h = self.get_seq_rep(None,
                input_dic['seq_attn_mask'], input_dic['seq_token_type_ids'], inp_emb)
            
            y_h = self.label_embed(input_dic['cl_labels'])
            
            # ------------------------------------------------------------------------------
            # generation direction
            gen_logits = self.get_gen_outputs(
                None, input_dic['seq_attn_mask'],
                input_dic['seq_token_type_ids'], y_h, inp_emb, causal_mask=True
            )
            
            gen_prob=F.softmax(gen_logits,dim=-1)
            gen_emb=torch.matmul(gen_prob,self.transformer.wte.weight)
            
            gen_ce_loss = self.loss_fct.mmdLoss(gen_emb,
                inp_emb, input_dic['seq_attn_mask'],
                input_dic['seq_token_type_ids'])
            

            nag_logits = self.get_gen_outputs(
                None, input_dic['seq_attn_mask'],
                input_dic['seq_token_type_ids'], y_h,inp_msk_emb, causal_mask=False
            )
            
            nag_prob=F.softmax(gen_logits,dim=-1)
            nag_emb=torch.matmul(gen_prob,self.transformer.wte.weight)
            
            nag_ce_loss = self.loss_fct.mmdLoss(nag_emb,
                inp_emb, input_dic['seq_attn_mask'],
                input_dic['seq_token_type_ids'])
            
            outs = nn.functional.gumbel_softmax(nag_logits,hard=False,dim=-1)#.to(input_dic['seq_ids'])
            out_emb = torch.matmul(outs,self.transformer.wte.weight)
            nag_ag_logits=nag_logits
            nag_ag_loss=nag_ce_loss
            
            cl_logits = self.cl_layer(xc_h)
            cl_ce_loss = self.loss_fct.clCELoss(cl_logits, input_dic['cl_labels'])

            # ------------------------------------------------------------------------------
            return GPT2VAEOutputWithPast(
                gen_loss=gen_ce_loss, cl_loss=cl_ce_loss,nag_loss=nag_ce_loss,
                nag_ag_loss=nag_ag_loss,nag_ag_logits=nag_ag_logits,
                gen_logits=gen_logits, cl_logits=cl_logits, nag_logits=nag_logits,
            )

        elif 'past_logit' in input_dic:
            raise NotImplementedError()

        # get rep of (x,c)  
        xc_h = self.get_seq_rep(input_dic['seq_ids'],
            input_dic['seq_attn_mask'], input_dic['seq_token_type_ids'])
        
        y_h = self.label_embed(input_dic['cl_labels'])
        
        # ------------------------------------------------------------------------------
        # generation direction
        
        
        gen_logits = self.get_gen_outputs(
            input_dic['seq_ids'], input_dic['seq_attn_mask'],
            input_dic['seq_token_type_ids'], y_h, causal_mask=True
        )
        
        gen_ce_loss = self.loss_fct.genCELoss(gen_logits,
            input_dic['seq_ids'], input_dic['seq_attn_mask'],
            input_dic['seq_token_type_ids'])
        

        nag_logits = self.get_gen_outputs(
            input_dic['masked_seq_ids'], input_dic['seq_attn_mask'],
            input_dic['seq_token_type_ids'], y_h, causal_mask=False
        )
        
        nag_ce_loss = self.loss_fct.genCELoss(nag_logits,
            input_dic['seq_ids'], input_dic['seq_attn_mask'],
            input_dic['seq_token_type_ids'])
        
        nag_ag_loss = nag_ce_loss
        nag_ag_logits = nag_logits
        
        # ------------------------------------------------------------------------------
        cl_logits = self.cl_layer(xc_h)
        cl_ce_loss = self.loss_fct.clCELoss(cl_logits, input_dic['cl_labels'])
        
        # ------------------------------------------------------------------------------
        return GPT2VAEOutputWithPast(
            gen_loss=gen_ce_loss, cl_loss=cl_ce_loss,nag_loss=nag_ce_loss,
            nag_ag_loss=nag_ag_loss,nag_ag_logits=nag_ag_logits,
            gen_logits=gen_logits, cl_logits=cl_logits, nag_logits=nag_logits,
        ) 
        
    def do_classification(self, input_dic, nsamples=1, with_sample=False, sample_n=None):
        # get rep 
        xc_h = self.get_seq_rep(input_dic['seq_ids'],
            input_dic['seq_attn_mask'], input_dic['seq_token_type_ids'])
        y_h = self.label_embed(input_dic['cl_labels'])
        
        gen_logits = self.get_gen_outputs(
            input_dic['seq_ids'], input_dic['seq_attn_mask'],
            input_dic['seq_token_type_ids'], y_h, causal_mask=True
        )
        
        gen_ce_loss = self.loss_fct.genCELoss(gen_logits,
            input_dic['seq_ids'], input_dic['seq_attn_mask'],
            input_dic['seq_token_type_ids'])
        
        cl_logits = self.cl_layer(xc_h)
        cl_ce_loss = self.loss_fct.clCELoss(cl_logits, input_dic['cl_labels'])
        return {'cl_loss': cl_ce_loss, 'cl_logits': cl_logits, 'gen_loss': gen_ce_loss}


    def do_generation(self, input_ids,
        token_type_ids, attn_mask, position_ids,
        past_key_values, latent_variable):
        
        # q(x|z,y,c)
        outputs = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            return_dict=True, use_cache=True,
            with_causal_mask=True,
            past_key_values=past_key_values,
            latent_variable=latent_variable
        )

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = outputs.last_hidden_state.to(self.lm_head.weight.device)
        else:
            hidden_states = outputs.last_hidden_state

        lm_logits = self.lm_head(hidden_states)
        
        return CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=outputs.past_key_values
        )
