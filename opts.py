import argparse


def get_args():
    parser = argparse.ArgumentParser(description='QA')

    """Optimization related arguments"""
    optim_args = parser.add_argument_group('Optimization related arguments')
    optim_args.add_argument('--train_batch_size', type=int,  default= 32, help='Training batch Size')
    optim_args.add_argument('--eval_batch_size', type=int,  default= 32, help='Evaluation batch Size')
    optim_args.add_argument('--test_batch_size', type=int,  default= 1, help='Test batch Size')
    optim_args.add_argument('--adam_epsilon', type=float,  default= 1e-8, help='Adam epsilon')
    optim_args.add_argument('--warmup_steps', type=int,  default= 100, help='Warmup Steps')
    optim_args.add_argument('--weight_decay', type=float,  default= 0.0, help='Warmup Steps')
    optim_args.add_argument('--learning_rate', type=float,  default=2e-5, help='Initial Learning rate')
    optim_args.add_argument( "--gradient_accumulation_steps",type=int, default=1, 
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
    optim_args.add_argument( "--val_check_interval",type=float, default=0.1, 
                            help="validation check interval ratio")
    optim_args.add_argument( "--gradient_cliping",type=float, default=0.5, 
                            help=" The value at which to clip gradients ")
    optim_args.add_argument( "--temperature",type=int, default=1, help=" Temperature ")
    optim_args.add_argument('--warmup_ratio', type=float,  default= 0.6, help='Warmup Ratio')
    optim_args.add_argument('--lr_monitor', action="store_true", help='Learning Rate Moniter')
    optim_args.add_argument('--lr_scheduler', action="store_true", help='Learning Rate schedular')
    
    """Data related arguments"""
    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--enc_max_len', type=int, default=256, help='Max sequence legnth')
    data_args.add_argument('--dec_max_len', type=int, default=128, help='Max sequence legnth')
    data_args.add_argument('--nle_anno_path', type=str, default="/media/storage/datasets/NLE_annotation/e_SNLI_VE", help='Path to annotation')
    data_args.add_argument('--nle_image_dir', type=str, default="/media/storage/datasets/image/flickr30k", help='Path to image dataset')
    
    data_args.add_argument('--img_size', type=int, default=224, help='Image size')        
    data_args.add_argument("--fewshot_ratio", type=float, default=-1, help="Ratio of few-shot data")
    data_args.add_argument("--fewshot_num", type=int, default=None, help="The number of few-shot data")
    data_args.add_argument("--cached_dir", type=str, default="/media/storage/checkpoints/OFA/base/cached", help="Directory with cached file")
    data_args.add_argument("--vis_rep_len", type=int, default=7*7, help="visual representation length")
    data_args.add_argument("--n_train_workers", type=int, default=8)
    data_args.add_argument("--n_valid_workers", type=int, default=4)
    data_args.add_argument("--n_test_workers", type=int, default=4)
    data_args.add_argument("--dataset_name", type=str, default= "vqax", help= "vqax | esnlive | actx")
    
    data_args.add_argument('--output_dir', type=str, default=None, help='Directory to store generated dataset')
    data_args.add_argument('--vqax_test_anno_path', type=str, default=None, help='Directory of vqax test annotation path')
    data_args.add_argument('--img_encoded', action="store_true", help='Encoding images beforehand')
    data_args.add_argument('--prediction', type=str, default="outputs/eval_data.json", help='Prediction dataset path')
    

    
    """Model related arguments"""
    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument("--model_path", type=str, default="OFA-base", help="Pretrained VL model")
    model_args.add_argument("--enc_model_path", type=str, default=None, help="Pretrained VL_encoder model")
    model_args.add_argument("--dec_model_path", type=str, default=None, help="Pretrained VL_decoder model")

    model_args.add_argument('--max_epochs', type=int, default=1, help='Max epoch size')
    model_args.add_argument('--load_from_epoch', type=str, default=None, help='Loading from epoch')
    model_args.add_argument("--top_k", type=float, default=0.0, help="top_k for generation")
    model_args.add_argument("--top_p", type=float, default=0.9, help="top_p for generation")
    model_args.add_argument("--mode", type=str, default="train", help="Train or Test")
    model_args.add_argument('--max_seq_len', type=int, default=40, help='Max sequence size')
    model_args.add_argument('--project_name', type=str, default="ICASSP2023", help='Project name')
    model_args.add_argument('--AEmode', type=str, default="EA", help='Answer Explanation of Explanation Answer')
    model_args.add_argument('--alignment', action="store_true", help='Alignment activation')
    model_args.add_argument('--concentration_hidden', action="store_true", help='Foreground and Background Detection about hidden states')
    model_args.add_argument('--concentration_attn', action="store_true", help='Foreground and Background Detection in attention map')
    model_args.add_argument('--sample_patch_num', type=int, default=None, help='Sample patches')
    model_args.add_argument('--tokenizer_name', type=str, default=None, help='Loading pretrained tokenizer')
    model_args.add_argument('--dropout_rate', type=float, default=0.0, help='Alignment dropout layer')
    model_args.add_argument('--q_attn', action="store_true", help='Question self attention')

    """Logging related arguments"""
    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=42, help='Random Seed')
    misc_args.add_argument('--experiment_name', type=str, default='EXE_experiments_s3', help='Experiment name for wandb')
    misc_args.add_argument('--ngpu', type=int, default=1, help='Number of gpu')
    misc_args.add_argument('--checkpoints_dir', type=str, default="/media/storage/checkpoints/OFA/base", help='Checkpoint directory')
    misc_args.add_argument('--load_ckpt_path', type=str, default=None, help='Checkpoint Loading directory')
    




    args = parser.parse_args()
    return args
