"""
MedViLL, pre-training model main run.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import time

from data.dataset_origin import create_dataset, create_loader
from utils import utils
from models.train_origin import MedViLL_Trainer
from transformers import AutoTokenizer

def train(config, args):
    utils.set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    print("Load Train dataset", config['train_dataset'])
    dset = create_dataset(tokenizer=tokenizer, args=args, config=config)

    print("Create DataLoader")
    train_data_loader = create_loader(
        [dset[0]], samplers=[None], 
        batch_size=[config['batch_size']], 
        is_trains=[True], num_workers=config['num_workers']
    )[0]

    print("Creating BERT Trainer")
    # Debug: In kiểu dữ liệu của các tham số quan trọng trong config
    print("Debug - config types:")
    for key, value in config.items():
        print(f"{key}: {value} (type: {type(value)})")
    
    start_time = time.time()
    try:
        trainer = MedViLL_Trainer(args, config, train_dataloader=train_data_loader)
        print(f"BERT Trainer created in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error creating trainer: {e}")
        raise  # Nâng lỗi lên để thấy traceback đầy đủ

    print("Training Start!")
    for epoch in range(config['epochs']):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlm_task", type=bool, default=True)
    parser.add_argument("--itm_task", type=bool, default=True)
    parser.add_argument('--BAR_attn', default=True, type=bool, help="Bidirectional Auto Regressive attn mask")
    parser.add_argument('--Mixed', default=False, type=bool, help="Mixed attn mask")
    parser.add_argument('--s2s_prob', default=1.0, type=float, help="S2S attention prob.")
    parser.add_argument('--bi_prob', default=0.0, type=float, help="Full_attention prob.")
    parser.add_argument('--disturbing_mask', default=False, type=bool, help="Baseline attn mask(I-I, T-T)")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--weight_load", type=bool, default=False, help='pre-trained_model_mid_epoch_load')
    parser.add_argument("--pre_trained_model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    
    config = yaml.load(open('/content/drive/MyDrive/MedViLL-runimage/configs/pretrain.yaml', 'r'), Loader=yaml.Loader)
    now = datetime.now()
    nowDate = now.strftime('%m%d-%H%M')
    args.output_path = f'output/{nowDate}'
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    print("Checking system resources...")
    import torch
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("VRAM allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
        print("VRAM reserved:", torch.cuda.memory_reserved() / 1024**3, "GB")

    train(config, args)