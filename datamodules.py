import json, os, random
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import utils
import logging
logger = logging.getLogger(__name__)


class OFA_BaseDataset(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        hparams,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.cfg = hparams
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.resolution = hparams.img_size
        self.patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
        ])
        self.tokenizer = tokenizer
        self.nle_anno_path = self.cfg.nle_anno_path
        self.image_dir = self.cfg.nle_image_dir
        self.AEmode = hparams.AEmode
        np.random.seed(self.hparams.seed)
        self.dataset = {}
        if self.cfg.dataset_name == "vqax":
            self.dataset["train"] = self.get_data_vqax(is_train = "train")
            self.dataset["validation"] = self.get_data_vqax(is_train = "val")
            self.dataset["test"] = self.get_data_vqax(is_train = "test")
        elif self.cfg.dataset_name == "esnlive":
            self.dataset["train"] = self.get_data_esnlive(is_train = "train")
            self.dataset["validation"] = None
            self.dataset["test"] = self.get_data_esnlive(is_train = "test")
        elif self.cfg.dataset_name == "actx":
            self.dataset["train"] = self.get_data_actx(is_train = "train")
            self.dataset["validation"] = self.get_data_actx(is_train = "val")
            self.dataset["test"] = self.get_data_actx(is_train = "test")
        else:
            raise ValueError(f"There is no {self.cfg.dataset_name} dataset")
        
    def get_data_vqax(self, is_train = None):
        
        file_name = f"{self.cfg.dataset_name}_{is_train}.json" 
        data_path = os.path.join(self.nle_anno_path,file_name)
        
        cached_features_file = os.path.join(self.cfg.cached_dir, f"cached_{self.cfg.dataset_name}_total_{is_train}_mode{self.AEmode}.pt")
        # Init features and dataset from cache if it exists
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features_and_dataset = torch.load(cached_features_file)
            datasets = features_and_dataset["datasets"]
        else:
            data = json.load(open(data_path, 'r'))
        
            ids_list = list(data.keys())
            index_tracker = {k: len(v['explanation']) - 1 for k,v in data.items()}
            
            for k,v in tqdm(data.items(), desc= "Data to list and dictionary..."):   
                if len(v['explanation']) > 1:   # some questions have more than one explanation
            # duplicate them for loading. -1 because one explanation is already in ids_list
                    ids_list += [str(k)] * (len(v['explanation']) - 1)
                
                
            datasets = []
            for i in tqdm(range(len(data)), desc= f"{is_train}_{self.cfg.dataset_name} preprocessing..."):
                try:
                    question_id = ids_list[i]
                    qid = torch.LongTensor([int(question_id)])
                    sample = data[question_id]
                    img_name = sample['image_name']
                    image_id = sample["image_id"]
                    question_txt = utils.proc_ques(sample['question'])    # question
                    question_txt = f" {question_txt}?"
                    answer_txt = utils.proc_ans(sample['answers'])
                    exp_idx = index_tracker[question_id]
                    
                    if is_train == "train":
                        img_path = os.path.join(os.path.join(self.image_dir, "train2014"), img_name)
                    else:
                        img_path = os.path.join(os.path.join(self.image_dir, "val2014"), img_name)
                    
                    
                    # if one more explanations
                    if exp_idx > 0:
                        index_tracker[question_id] -= 1    # decrease usage
                        
                    explain = sample['explanation'][exp_idx]
                    if self.AEmode == "AE":
                        explain_txt = f" because {explain}"
                        tgt_txt = answer_txt + explain_txt
                    else:
                        explain_txt = f" {explain}"
                        answer_txt = f" so the answer is {answer_txt}"
                        if self.cfg.q_attn:
                            tgt_txt = question_txt + explain_txt + answer_txt
                        else:
                            tgt_txt = explain_txt + answer_txt
                        
                    question = self.tokenizer(question_txt).input_ids
                    labels = self.tokenizer(tgt_txt).input_ids
                        
                    img = Image.open(img_path)
                    img_ids = self.patch_resize_transform(img)
                    
                    input_ids = torch.tensor(question, dtype=torch.long)
                    labels = torch.tensor(labels, dtype=torch.long)
                    
                    patch_mask = torch.tensor([True])
                    if is_train == "test":
                        decoder_input_ids = torch.tensor(self.tokenizer.bos_token_id)
                        datasets.append({"image_path": img_path, "inputs" : input_ids, "labels": labels, "img": img_ids, "patch_mask" : patch_mask, "image_id": image_id, "decoder_input_ids":decoder_input_ids, "qid": qid})
                    else:
                        datasets.append({"image_path": img_path, "input_ids" : input_ids, "labels": labels, "img": img_ids, "patch_mask" : patch_mask, "image_id": image_id})
                except:
                    print(datasets[-1])
            torch.save({"datasets": datasets}, cached_features_file)
        
        return datasets
    
    def get_data_esnlive(self, is_train = None):
        
        file_name = f"{self.cfg.dataset_name}_{is_train}.json" 
        data_path = os.path.join(self.nle_anno_path,file_name)
        
        cached_features_file = os.path.join(self.cfg.cached_dir, f"cached_{self.cfg.dataset_name}_total_{is_train}_mode{self.AEmode}.pt")
        # Init features and dataset from cache if it exists
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features_and_dataset = torch.load(cached_features_file)
            datasets = features_and_dataset["datasets"]
        else:
            data = json.load(open(data_path, 'r'))
        
            ids_list = list(data.keys())
            datasets = []
            for i in tqdm(range(len(data)), desc= f"{is_train}_{self.cfg.dataset_name} preprocessing..."):
                
                question_id = ids_list[i]
                sample = data[question_id]
                img_name = sample['image_name']
                image_id = sample['image_name'].split(".jpg")[0]
                
                question_txt = utils.proc_ques(sample['hypothesis'])    # question
                question_txt = f" {question_txt}?"
                
                answer_txt = sample['answers']
                explain = sample['explanation']
                
                if self.AEmode == "AE":
                    explain_txt = f" because {explain}"
                    tgt_txt = answer_txt + explain_txt
                else:
                    explain_txt = f" {explain}"
                    answer_txt = f" so the answer is {answer_txt}"
                    tgt_txt = explain_txt + answer_txt
                    
                question = self.tokenizer(question_txt).input_ids
                labels = self.tokenizer(tgt_txt).input_ids
                
                img_path = os.path.join(self.image_dir, img_name)
                img = Image.open(img_path)
                img_ids = self.patch_resize_transform(img)
                
                input_ids = torch.tensor(question, dtype=torch.long)
                labels = torch.tensor(labels, dtype=torch.long)
                
                patch_mask = torch.tensor([True])
                if is_train == "test":
                    decoder_input_ids = torch.tensor(self.tokenizer.bos_token_id)
                    datasets.append({"image_path": img_path, "inputs" : input_ids, "labels": labels, "img": img_ids, "patch_mask" : patch_mask, "image_id": image_id, "decoder_input_ids":decoder_input_ids})
                else:
                    datasets.append({"image_path": img_path, "input_ids" : input_ids, "labels": labels, "img": img_ids, "patch_mask" : patch_mask, "image_id": image_id})
            torch.save({"datasets": datasets}, cached_features_file)
        
        return datasets
    
    def get_data_actx(self, is_train = None):
        
        file_name = f"{self.cfg.dataset_name}_{is_train}.json" 
        data_path = os.path.join(self.nle_anno_path,file_name)
        
        cached_features_file = os.path.join(self.cfg.cached_dir, f"cached_{self.cfg.dataset_name}_total_{is_train}_mode{self.AEmode}.pt")
        # Init features and dataset from cache if it exists
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features_and_dataset = torch.load(cached_features_file)
            datasets = features_and_dataset["datasets"]
        else:
            data = json.load(open(data_path, 'r'))
        
            ids_list = list(data.keys())
            index_tracker = {k: len(v['explanation']) - 1 for k,v in data.items()}
            
            for k,v in tqdm(data.items(), desc= "Data to list and dictionary..."):   
                if len(v['explanation']) > 1:   # some questions have more than one explanation
            # duplicate them for loading. -1 because one explanation is already in ids_list
                    ids_list += [str(k)] * (len(v['explanation']) - 1)
                
                
            datasets = []
            for i in tqdm(range(len(data)), desc= f"{is_train}_{self.cfg.dataset_name} preprocessing..."):
                
                question_id = ids_list[i]
                sample = data[question_id]
                img_name = sample['image_name']
                image_id = sample["image_id"]
                question_txt = f"What action is shown?"
                answer_txt = utils.proc_ans(sample['answers'])
                exp_idx = index_tracker[question_id]
                
                if is_train == "train":
                    img_path = os.path.join(os.path.join(self.image_dir, "train2014"), img_name)
                else:
                    img_path = os.path.join(os.path.join(self.image_dir, "val2014"), img_name)
                
                
                # if one more explanations
                if exp_idx > 0:
                    index_tracker[question_id] -= 1    # decrease usage
                    
                explain = sample['explanation'][exp_idx]
                if self.AEmode == "AE":
                    explain_txt = f" because {explain}"
                    tgt_txt = answer_txt + explain_txt
                else:
                    explain_txt = f" {explain}"
                    answer_txt = f" so the answer is {answer_txt}"
                    tgt_txt = explain_txt + answer_txt
                    
                question = self.tokenizer(question_txt).input_ids
                labels = self.tokenizer(tgt_txt).input_ids
                
                img_path = os.path.join(self.image_dir, img_name)
                img = Image.open(img_path)
                img_ids = self.patch_resize_transform(img)
                
                input_ids = torch.tensor(question, dtype=torch.long)
                labels = torch.tensor(labels, dtype=torch.long)
                
                patch_mask = torch.tensor([True])
                if is_train == "test":
                    decoder_input_ids = torch.tensor(self.tokenizer.bos_token_id)
                    datasets.append({"image_path": img_path, "inputs" : input_ids, "labels": labels, "img": img_ids, "patch_mask" : patch_mask, "image_id": image_id, "decoder_input_ids":decoder_input_ids})
                else:
                    datasets.append({"image_path": img_path, "input_ids" : input_ids, "labels": labels, "img": img_ids, "patch_mask" : patch_mask, "image_id": image_id})
            torch.save({"datasets": datasets}, cached_features_file)
        
        return datasets
    
    
    def collate_fn(self, batch):
        # Collate function definition
        value_lst = [list(lst.values()) for lst in batch]
        batch = list(zip(*value_lst))
        sample = {}
        
        # max len
        input_max_len = max([x.size(0) for x in batch[1]])
        output_max_len = max([x.size(0) for x in batch[2]])
        
        input_slicing = False
        output_slicing = False
        
        if self.cfg.max_seq_len < input_max_len:
            input_max_len = self.cfg.max_seq_len
            input_slicing = True
        elif self.cfg.max_seq_len < output_max_len:
            output_max_len = self.cfg.max_seq_len
            output_slicing = True
        else:
            pass
        
    # input ids and attention masking
        inputs = torch.ones((len(batch[1]), input_max_len), dtype=torch.long)
        encoder_attn_mask = torch.zeros((len(batch[1]), input_max_len), dtype=torch.long)
        
        for i, x in enumerate(batch[1]):
            if input_slicing:
                x = x[:input_max_len]
            inputs[i,:x.size(0)] = x
            encoder_attn_mask[i,:x.size(0)] = 1.0
        
        if self.cfg.q_attn:
            labels = torch.ones((len(batch[2]), output_max_len), dtype=torch.long)
            q_attn_mask = torch.ones((len(batch[2]), output_max_len), dtype=torch.long)
            for i, (x, y) in enumerate(zip(batch[2], batch[1])):
                if output_slicing:
                    x = x[:output_max_len]
                labels[i,:x.size(0)] = x
                q_attn_mask[i,1:y.size(0)] = 0.0
            patch_masks = torch.cat([btc for btc in batch[4]])
        else:
            labels = torch.ones((len(batch[2]), output_max_len), dtype=torch.long)
            q_attn_mask = None
            for i, (x, y) in enumerate(zip(batch[2], batch[1])):
                if output_slicing:
                    x = x[:output_max_len]
                labels[i,:x.size(0)] = x
            patch_masks = torch.cat([btc for btc in batch[4]]) 
        
    # Decoder ids
        if len(batch) > 6 :
            sample["decoder_input_ids"] = batch[6]
            

    # Stack
        sample["inputs"] = inputs
        sample["encoder_attention_mask"] = encoder_attn_mask
        sample["labels"] = labels
        sample["image_path"] = batch[0]
        sample["img"] = torch.stack(batch[3])
        sample["patch_mask"] = patch_masks
        sample["image_id"] = batch[5]
        sample["question_mask"] = q_attn_mask
        return sample
    
    def __len__(self):
        return len(self.dataset["train"])
        
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.cfg.train_batch_size, \
                        pin_memory=True, num_workers=self.cfg.n_train_workers, collate_fn = self.collate_fn)
    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.cfg.eval_batch_size, \
            pin_memory=True, num_workers=self.cfg.n_valid_workers, collate_fn = self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=1, pin_memory=True, num_workers=self.cfg.n_test_workers,)