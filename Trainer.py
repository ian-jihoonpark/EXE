import os
from random import random
from transformers import OFATokenizer
import torch
from pytorch_lightning import Trainer, seed_everything
from datamodules import OFA_BaseDataset, NLX_GPT_BaseDataset, distillBERT_BaseDataset, EXE_BaseDataset, Image_CaptioningDataset
from lightning import OFA_x
import opts
import datetime
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

AVAIL_GPUS = torch.cuda.device_count()

def get_checkpoint_callback(monitor, args):
    # Checkpoint call back
    now = datetime.datetime.now()
    if args.experiment_name is None:
        nowDatetime = now.strftime('%Y-%m-%d_%H:%M')
        ckpt_dir = args.checkpoints_dir + '/' + nowDatetime + "/"
    else:
        ckpt_dir = args.checkpoints_dir + "/" + args.experiment_name

    monitor = f"{monitor}_val_loss"
    filename = "{epoch:02d}-{"+monitor+":.3f}"

    checkpoint_callback = checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode = "min",
        save_top_k = 3,
        dirpath=ckpt_dir,
        filename=filename,
    )    
    if args.lr_monitor:
        lr_monitor = LearningRateMonitor(logging_interval="step")
    else:
        return checkpoint_callback
        
    return [checkpoint_callback, lr_monitor]

    
if __name__ == '__main__':
    args = opts.get_args()
    seed_everything(args.seed)

    if args.model_path == "OFA-base":
        tokenizer = OFATokenizer.from_pretrained(args.model_path)
        datamodule = OFA_BaseDataset(tokenizer, args)
    
    if args.mode == "test":
        trainer = Trainer(accelerator="gpu", gpus=1)
        model = OFA_x.load_from_checkpoint(args.load_ckpt_path, strict=False, tokenizer=tokenizer, hparams=args)
        trainer.test(model, datamodule=datamodule)
        
    else:  
        model = OFA_x(tokenizer = tokenizer, hparams =args)
        ckpt_pth = os.path.join(args.checkpoints_dir,args.experiment_name)
        if not os.path.isdir(ckpt_pth):
            os.mkdir(ckpt_pth)
    
        ckpt_callback = get_checkpoint_callback(monitor=args.model_path, args= args) 
        logger = WandbLogger(project=args.project_name, name=args.experiment_name)   
        trainer = Trainer(max_epochs=args.max_epochs,
                        accelerator = "gpu",
                        gpus= args.ngpu,
                        strategy = "ddp",
                        val_check_interval=args.val_check_interval,
                        accumulate_grad_batches = args.gradient_accumulation_steps,
                        gradient_clip_val=args.gradient_cliping,
                        check_val_every_n_epoch = 1,
                        callbacks=ckpt_callback,
                        logger=logger,
                        )
        trainer.fit(model, datamodule=datamodule)