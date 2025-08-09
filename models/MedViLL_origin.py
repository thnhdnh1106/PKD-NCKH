import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": lambda x: x * torch.sigmoid(x)}

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Img_patch_embedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=2048):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, dim))

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        batch_size = x.size(0)
        pos_ids = torch.arange(self.num_patches, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = x + self.position_embeddings.expand(batch_size, -1, -1)
        return embeddings, pos_ids

class ImageBertEmbeddings(nn.Module):
    def __init__(self, embeddings, img_dim=2048, hidden_size=768):
        super().__init__()
        self.img_embeddings = nn.Linear(img_dim, hidden_size)
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(0.1)
        self.position_embeddings = embeddings.position_embeddings

    def forward(self, input_imgs, img_pos, token_type_ids):
        imgs_embeddings = self.img_embeddings(input_imgs)
        img_pos = img_pos.long()[:, :imgs_embeddings.shape[1]]
        position_embeddings = self.position_embeddings(img_pos)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = imgs_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MedViLLEncoder(BertPreTrainedModel):
    def __init__(self, model_config, args, configs):
        super().__init__(model_config)
        self.args = args
        self.configs = configs
        bert = BertModel(model_config)
        
        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(self.txt_embeddings, img_dim=2048, hidden_size=configs['hidden_size'])
        
        if configs['img_encoder'] == 'ViT':
            img_size = configs['img_size']
            patch_sz = 16
            self.img_encoder = Img_patch_embedding(image_size=img_size, patch_size=patch_sz, dim=2048)
        
        self.encoder = bert.encoder
        self.pooler = bert.pooler

        self.mlm_head = nn.Sequential(
            nn.Linear(configs['hidden_size'], configs['hidden_size']),
            nn.GELU(),
            BertLayerNorm(configs['hidden_size'], eps=1e-5),
            nn.Linear(configs['hidden_size'], configs['vocab_size'])
        )
        self.itm_head = nn.Linear(configs['hidden_size'], 2)

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):
        batch_size = cls_tok.size(0)
        
        # Process image embeddings
        img_out, img_pos = self.img_encoder(input_img)
        img_segment = torch.ones(batch_size, self.configs['num_image_embeds']).long().to(input_img.device)
        img_out = self.img_embeddings(img_out, img_pos, img_segment)
        
        # Process text embeddings
        cls_segment = torch.zeros(batch_size, 1).long().to(input_img.device)
        cls_out = self.txt_embeddings(cls_tok, cls_segment)
        
        sep_segment = torch.ones(batch_size, 1).long().to(input_img.device)
        sep_out = self.txt_embeddings(sep_tok, sep_segment)
        
        txt_out = self.txt_embeddings(input_txt, segment)
        
        # Concatenate all embeddings
        seq_output = torch.cat([cls_out, img_out, sep_out, txt_out], dim=1)
        
        # Calculate expected sequence length
        total_seq_len = 1 + self.configs['num_image_embeds'] + 1 + input_txt.size(1)
        
        # Create proper attention mask
        if attn_mask.dim() == 2:
            # If mask is 2D [batch, seq_len], expand to 4D
            if attn_mask.size(1) != total_seq_len:
                # Resize mask if needed
                new_mask = torch.ones(batch_size, total_seq_len, device=attn_mask.device)
                text_len = input_txt.size(1)
                new_mask[:, -text_len:] = attn_mask[:, -min(text_len, attn_mask.size(1)):]
                attn_mask = new_mask
            extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        elif attn_mask.dim() == 3:
            # If mask is 3D [batch, seq_len, seq_len], ensure correct size
            if attn_mask.size(1) != total_seq_len or attn_mask.size(2) != total_seq_len:
                raise ValueError(f"3D attention mask must have shape [batch, {total_seq_len}, {total_seq_len}]")
            extended_attn_mask = attn_mask.unsqueeze(1)
        else:
            raise ValueError(f"Attention mask must be 2D or 3D, got {attn_mask.dim()}D")
        
        # Convert mask to BERT format
        extended_attn_mask = extended_attn_mask.to(dtype=torch.float32)
        extended_attn_mask = (1.0 - extended_attn_mask) * -10000.0
        
        # Pass through encoder
        encoder_outputs = self.encoder(seq_output, attention_mask=extended_attn_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # Calculate outputs
        mlm_output = self.mlm_head(sequence_output[:, self.configs['num_image_embeds'] + 2:])
        itm_output = self.itm_head(pooled_output)

        return mlm_output, itm_output

class MedViLL(MedViLLEncoder):
    def __init__(self, model_config, args, configs):
        super().__init__(model_config, args, configs)
        self.enc = MedViLLEncoder(model_config, args, configs)

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):
        return self.enc(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)