import sys
import os
sys.path.append('/content/drive/MyDrive/MedViLL-runimage')

"""
Construct CXR-BERT or BertForMaskedLM, Training and Saving with Image-only Input
"""
import tqdm
import torch
import datetime
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from models.MedViLL_origin import MedViLL
from torch.optim import AdamW
from transformers import BertConfig, AutoConfig

class MedViLL_Trainer():
    def __init__(self, args, configs, train_dataloader, test_dataloader=None):
        print("Initializing MedViLL_Trainer...")
        self.args = args
        self.configs = configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device selected: {self.device}")
        print(f"Current CUDA device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")

        if args.weight_load:
            print("Loading pre-trained model...")
            model_config = AutoConfig.from_pretrained(args.pre_trained_model_path, attn_implementation="eager")
            model_state_dict = torch.load(os.path.join(args.pre_trained_model_path, 'pytorch_model.bin'))
            self.model = MedViLL.from_pretrained(args.pre_trained_model_path, state_dict=model_state_dict,
                                model_config=model_config, args=args, configs=configs).to(self.device)
            print('Pre-trained model loaded')
        else:
            print("Creating new model from bert-base-uncased...")
            model_config = BertConfig.from_pretrained("bert-base-uncased", attn_implementation="eager")
            self.model = MedViLL(model_config, args, configs).to(self.device)
            print("Model created and moved to device")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for BERT")
            self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)

        self.model_without_ddp = self.model
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
            self.model_without_ddp = model.module
        
        print("Assigning dataloaders...")
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        print("Creating optimizer and criteria...")
        self.optimizer = AdamW(self.model.parameters(), lr=float(self.configs['lr']))  # Ép kiểu lr thành float
        self.criterion = nn.CrossEntropyLoss()
        self.step_cnt = 0
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.model.train()
        train_losses = []
        train_data_iter = tqdm.tqdm(enumerate(self.train_data),
                                    desc=f'EP_:{epoch}',
                                    total=len(self.train_data),
                                    bar_format='{l_bar}{r_bar}')
        
        total_correct, total_element = 0, 0

        for i, data in train_data_iter:
            images = data[4].to(self.device)
            labels = data[6].to(self.device)
            cls_tok = data[0].to(self.device)
            input_txt = data[1].to(self.device)
            attn_mask = data[3].to(self.device)
            segment = data[5].to(self.device)
            sep_tok = data[7].to(self.device)

            mlm_output, itm_output = self.model(cls_tok, input_txt, attn_mask, segment, images, sep_tok)
            loss = self.criterion(itm_output, labels)

            train_losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            correct = itm_output.argmax(dim=-1).eq(labels).sum().item()
            total_correct += correct
            total_element += labels.nelement()

        print("Avg loss per epoch:", np.mean(train_losses))
        print("Avg accuracy per epoch:", round(total_correct / total_element * 100, 3))

        if self.test_data:
            self.model.eval()
            test_losses = []
            test_data_iter = tqdm.tqdm(enumerate(self.test_data),
                                       desc=f'EP_:{epoch} Test',
                                       total=len(self.train_data),
                                       bar_format='{l_bar}{r_bar}')
            total_test_correct, total_test_element = 0, 0

            with torch.no_grad():
                for i, data in test_data_iter:
                    images = data[4].to(self.device)
                    labels = data[6].to(self.device)
                    cls_tok = data[0].to(self.device)
                    input_txt = data[1].to(self.device)
                    attn_mask = data[3].to(self.device)
                    segment = data[5].to(self.device)
                    sep_tok = data[7].to(self.device)

                    mlm_output, itm_output = self.model(cls_tok, input_txt, attn_mask, segment, images, sep_tok)
                    loss = self.criterion(itm_output, labels)
                    test_losses.append(loss.item())

                    correct = itm_output.argmax(dim=-1).eq(labels).sum().item()
                    total_test_correct += correct
                    total_test_element += labels.nelement()

            print("Avg loss in testset:", np.mean(test_losses))
            print("Avg accuracy in testset:", round(total_test_correct / total_test_element * 100, 3))

    def save(self, epoch, file_path):
        save_path_per_ep = os.path.join(file_path, str(epoch))
        os.makedirs(save_path_per_ep, exist_ok=True)
        
        if torch.cuda.device_count() > 1:
            self.model.module.save_pretrained(save_path_per_ep, safe_serialization=False)
            print(f'Multi_EP: {epoch} Model saved on {save_path_per_ep}')
        else:
            self.model.save_pretrained(save_path_per_ep, safe_serialization=False)
            print(f'Single_EP: {epoch} Model saved on {save_path_per_ep}')
        os.chmod(save_path_per_ep + '/pytorch_model.bin', 0o777)