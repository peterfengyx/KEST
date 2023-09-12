from logging import info
from numpy import negative
from transformers import Seq2SeqTrainingArguments
from transformers.configuration_utils import PretrainedConfig

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExtraArguments:
    tokenizer_dir: str = field(default="../vocab")
    corpus_dir: str = field(default="../corpus")

    training_data: Optional[str] = field(default="new_train.csv")
    validation_data: Optional[str] = field(default="news_valid.json")
    testing_data: Optional[str] = field(default="new_test.csv")
    
    train_cache_file: str = field(default="../data/train_cache.pickle")
    valid_cache_file: str = field(default="../data/valid_cache.pickle")
    test_cache_file: str = field(default="../data/test_cache.pickle")
    unlabel_cache_file: str = field(default="../data/unlabel_cache.pickle")

    max_body_len: int = field(default=10)
    max_title_len: int = field(default=100)
    min_gen_len: int = field(default=10)
    max_gen_len: int = field(default=40)
    
    preprocessing_num_workers: Optional[int] = field(default=1)
    preprocessing_bsize: Optional[int] = field(default=64)
    
    overwrite_cache: bool = field(default=False)
    
    max_train_samples: Optional[int] = field(default=None)
    max_valid_samples: Optional[int] = field(default=None)
    
    print_sample_steps: int = field(default=20000)
    print_sample_num: int = field(default=1)
    
    
    cl_weight: float = field(default=1.0)
    gen_weight: float = field(default=1.0)
    nag_weight: float = field(default=1.0)
    nag_ag_weight: float = field(default=0.0) #deprecated
    
    st_mode: str = 'nag' # ag or nag
    soft_label: bool = False
    use_prior: bool = True
    return_emb: bool = False
    self_training_steps_cls: int = 12150
    self_training_steps_gen: int = 405
    supervised_steps: int = 405
    st_start_step: int = 1
    st_sampling: str = ""
    st_select: str = "argmax"
    gen_temperature: float = 1.0
    mask_prob: float = 0.7
    

extra_args = ExtraArguments()

train_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    generation_max_length=extra_args.max_body_len+extra_args.max_title_len+2,
    output_dir="../ckpts/",
    seed=506,
    data_seed=506,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    #eval_steps=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=10.0,
    warmup_steps=12960,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=200,
    max_steps=259201,
    local_rank=-1,
    xpu_backend='ncll',
    no_cuda=False,
    logging_dir="../log/",
    ignore_data_skip=True,
    remove_unused_columns=False,
)


#-------------------------------------
class GPT2Config(PretrainedConfig):
    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        **kwargs,
    ):
        self.vocab_size = 28996
        self.n_positions = 512
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.n_inner = None
        self.activation_function = "gelu"
        self.resid_pdrop = 0.1
        self.embd_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.hidden_pdrop = 0.1
        self.cl_pdrop = 0.1
        self.layer_norm_epsilon = 1e-12
        self.initializer_range = 0.02
        self.summary_type = "cls_index"
        self.summary_use_proj = True
        self.summary_activation = None
        self.summary_first_dropout = 0.1
        self.summary_proj_to_labels = True
        self.scale_attn_weights = True
        self.use_cache = True
        self.scale_attn_by_inverse_layer_idx = False
        self.reorder_and_upcast_attn = False

        self.pad_token_id=0
        self.bos_token_id=-1
        self.eos_token_id=-1
        
        self.n_label = 4
        self.n_label_embd = 128
        self.n_latent = 128
        self.kl_lambda=0.01
        self.gen_kl_lambda=0.01

        super().__init__(bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id, **kwargs)
