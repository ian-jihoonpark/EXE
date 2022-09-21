import json
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from ofa_modeling import OFAForConditionalGeneration, OFAEncoder
from transformers import (
    top_k_top_p_filtering,
    get_linear_schedule_with_warmup,
    AutoConfig,
    OFAConfig, 
    AutoConfig, 
    top_k_top_p_filtering, 
    )
from utils import filter_and_get_scores
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
logger = logging.getLogger(__name__)


class OFA_x(LightningModule):
    def __init__(self, tokenizer, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer
        self.args = hparams
        self.model_path = hparams.model_path 
        config = OFAConfig.from_pretrained(self.model_path)
        # Add configs
        # setattr(config, 'img_size', None)
        # # setattr(config, 'max_seq_len', None)   
        config.img_size = hparams.img_size
        config.max_seq_len = hparams.max_seq_len
        config.alignment = hparams.alignment
        config.concentration_hidden = hparams.concentration_hidden
        config.concentration_attn = hparams.concentration_attn
        self.sample_patch_num = hparams.sample_patch_num

        
        # config.add_cross_attention = True
        if hparams.alignment:
                # self.align_layer = nn.ModuleList([
                # nn.Linear(config.d_model,config.vocab_size),
                # nn.Linear(config.vocab_size,config.d_model),
                # nn.ReLU(),
                # nn.Dropout(config.activation_dropout)
                # ])
                self.align_layer = nn.ModuleList([
                nn.Linear(config.d_model,config.d_model),
                nn.Dropout(hparams.dropout_rate)
                ])
        else:
            self.align_layer = None
            
            
        # Load model
        self.model = OFAForConditionalGeneration.from_pretrained(
            self.model_path,
            config = config, 
            align_layer=self.align_layer, 
            )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.weight_ckpt_pth = os.path.join(self.args.checkpoints_dir,self.args.experiment_name)
        self.pre_loss = torch.tensor(10.0)
        # .to(torch.cuda.current_device())
        torch.cuda.empty_cache()

    def setup(self,stage):
        
        if self.args.warmup_steps < 0:
            self.args.warmup_steps = self.total_steps * self.args.warmup_ratio
        if stage=="fit":
            self.total_steps = len(self.trainer.datamodule) // self.args.gradient_accumulation_steps // self.args.ngpu * self.args.max_epochs
            self.warmup_steps = self.hparams.warmup_steps
        elif stage=="test" or stage=="predict":
            self.results_full = []
            self.results_exp = []
            self.eval_results = {}
            
            SEG_TOKENS = ['<question>', '<answer>', '<explanation>']
            self.seg_token_ids = self.tokenizer.convert_tokens_to_ids(SEG_TOKENS)
            self.because_token_id = self.tokenizer.convert_tokens_to_ids('Ġbecause')
            self.eos_token_id = [self.tokenizer.eos_token_id]
            self.special_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id] + self.seg_token_ids
        
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        if self.hparams.lr_scheduler:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "name": "lr"}
        else:
            return [optimizer]
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self,  batch, batch_idx):
        outputs = self(
            input_ids=batch["inputs"],
            patch_masks = batch["patch_mask"],
            past_key_values=None, 
            patch_images=batch["img"],
            labels=batch["labels"], 
            use_cache=False, 
            return_dict=True,
            sample_patch_num = self.sample_patch_num,
            question_mask = batch["question_mask"],
                    )
        
        loss = outputs.loss
            
        self.log(f"{self.args.model_path}_train_loss", loss)
            
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["img"].shape[0]
        outputs = self(
            input_ids=batch["inputs"],
            patch_masks = batch["patch_mask"],
            past_key_values=None, 
            patch_images=batch["img"],
            labels=batch["labels"], 
            use_cache=False, 
            return_dict=True,
            sample_patch_num = self.sample_patch_num,
                    )
        loss = outputs.loss
        if self.pre_loss > loss:
            self.model.save_pretrained(self.weight_ckpt_pth)
            self.pre_loss = loss
        
        
        
        self.log(f"{self.args.model_path}_val_loss", loss)

        return loss

    def test_step(self,batch,batch_idx):
        # decoder_input_ids = decoder_input_ids.unsqueeze(0)
        batch_size = batch["inputs"][0].shape[0]
        max_len = 20
        always_exp = False
        no_sample = True
        current_output = []
        input_ids = batch["inputs"]
        decoder_input_ids = batch["decoder_input_ids"]
        current_output = self.model.generate(batch["inputs"], patch_images=batch["img"], max_length = max_len, num_beams=4)
        
        decoded_sequences = self.tokenizer.decode(current_output[0], skip_special_tokens=True).lstrip()
        question = self.tokenizer.decode(batch["inputs"][0], skip_special_tokens=True).lstrip()
        image_name = batch["image_path"][0].split("/")[-1]
        self.results_full.append({"image_id" :batch["image_id"][0] ,"image_pth": image_name, "caption": decoded_sequences, "question": question})

        
        
        if 'because' in decoded_sequences:
            if self.args.AEmode == "AE":
                cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
            else:
                cut_decoded_sequences = decoded_sequences.split('the answer is')[0].split("because")[-1].strip()
                
        else:
            cut_decoded_sequences = decoded_sequences.split('so the answer is')[0].strip()
        self.eval_results[batch["qid"][0].item()]={"question": question, "explanation": cut_decoded_sequences}    
        self.results_exp.append({"image_id" :batch["image_id"][0], "image_pth": image_name, "caption": cut_decoded_sequences})         
        return {"reults_full" : self.results_full, "results_exp": self.results_exp}

    def test_epoch_end(self, batch_parts):
        if not os.path.exists(self.hparams.output_dir):
            os.mkdir(self.hparams.output_dir)

        resFileExp = os.path.join(self.hparams.output_dir , 'captions.json')
        unf_resFileExp = os.path.join(self.hparams.output_dir , 'unf_captions.json') 
        unf_resFileFull = os.path.join(self.hparams.output_dir , 'unf_captions_full.json')
        save_scores_pathExp = os.path.join(self.hparams.output_dir , 'scores.json')
        save_eval_datasets = os.path.join(self.hparams.output_dir , 'eval_data.json')
        
        with open(unf_resFileExp, 'w') as w:
            json.dump(self.results_exp, w)
            
        with open(unf_resFileFull, 'w') as w:
            json.dump(self.results_full, w)

        with open(save_eval_datasets, 'w') as w:
            json.dump(self.eval_results, w)
        
        filter_and_get_scores(resFileExp, save_scores_pathExp, self.results_full, self.results_exp)
