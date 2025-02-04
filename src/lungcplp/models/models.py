from collections import OrderedDict
from typing import Tuple, Union

import numpy as np, math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from einops import rearrange, repeat

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.drop1(x)
        x = self.act2(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)

class DynamicMLP(nn.Module):
    def __init__(self, input_size, layers: list, activation=nn.ReLU, dropout=0.1):
        """
        layers: list of layer dimensions after input layer [layer_1, layer_2, ..., n_classes]
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size in layers:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            if activation is not None:
                self.layers.append(activation())
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
                
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AbsTimeEncoding(nn.Module):
    def __init__(self, dim, dropout=0.0, n_patches=1):
        super().__init__()
        self.n_patches = n_patches
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(torch.arange(0, dim, 2) *-(math.log(10000.0) / dim))
        self.register_buffer('div_term', div_term)
        self.dim = dim
        
    def forward(self, x, t):
        device, dtype = x.device, x.dtype
        pe = torch.zeros(x.shape, device=device, dtype=dtype)
        
        # repeat times into shape [b, t, dim]
        time_position = repeat(t, 'b t -> b t d', d=int(self.dim/2))
        time_position = time_position.repeat_interleave(self.n_patches, dim=1)
        pe[:, :, 0::2] = torch.sin(time_position * self.div_term.expand_as(time_position))
        pe[:, :, 1::2] = torch.cos(time_position * self.div_term.expand_as(time_position))
        x = x + Variable(pe, requires_grad=False)
        return self.dropout(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
    # def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, key_padding_mask, attn_mask):
        # self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]

    def forward(self, x, key_padding_mask, attn_mask):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, key_padding_mask=None, attn_mask=None):
        for block in self.resblocks:
            x = block(x, key_padding_mask, attn_mask)
        return x

class TDViT(nn.Module):
    def __init__(self,
                vision_width: int=512,
                transformer_width: int=256,
                transformer_heads: int=4,
                transformer_layers: int=8,
                classifier_depth: int=2,
                n_classes: int=2,
                **kwargs,
                ):
        super().__init__()
        # Vision
        self.vision_width = vision_width
        self.visual_embed = nn.Linear(vision_width, transformer_width)

        self.tdvit = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )
        self.time_encoding = AbsTimeEncoding(transformer_width, dropout=0., n_patches=1)
        self.cls_token = nn.Parameter(torch.empty(1, 1, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        layers = [transformer_width*2 for _ in range(classifier_depth-1)] + [n_classes]
        self.mlp_final = DynamicMLP(transformer_width, layers, activation=nn.ReLU, dropout=0.1)
    
    @property
    def dtype(self):
        return self.visual_embed.weight.dtype
    
    def forward(self, images, padding, times):
        # images: b t d, padding: b t, times: b t
        B, T, *_ = images.shape
        x = self.visual_embed(images.type(self.dtype))        
        x = self.time_encoding(x, times)

        # Create attention mask from padding
        padding = torch.where(padding > 0, False, True) # True value indicates position not allowed to attend

        x = x.permute(1, 0, 2)  # btd -> tbd
        x = self.tdvit(x, key_padding_mask=padding, attn_mask=None)
        x = x.permute(1, 0, 2)  # tbd -> btd
        x = self.ln_final(x).type(self.dtype)
        # take features from the most recent embedding, which is the first in the sequence
        x = x[torch.arange(x.shape[0]), 0]
        x = self.mlp_final(x)
        return x 
    
class ImageClassifier(nn.Module):
    def __init__(self,
                vision_width: int=512,
                classifier_depth: int=2,
                n_classes: int=2,
                **kwargs,
                ):
        super().__init__()
        # Vision
        layers = [vision_width*2 for _ in range(classifier_depth-1)] + [n_classes]
        self.mlp_final = DynamicMLP(vision_width, layers, activation=nn.ReLU, dropout=0.1)
    
    def forward(self, x):
        # images: b t d, padding: b t, times: b t
        return self.mlp_final(x[:, 0])

class CLIP(nn.Module):
    def __init__(self,
                vision_width: int=512, # img dim
                vocab_size: int=13, # expr dim
                embed_dim: int=64,
                transformer_width: int=256,
                transformer_heads: int=4,
                transformer_layers: int=8,
                ):
        super().__init__()

        # Vision
        self.vision_width = vision_width
        self.visual_embed = nn.Linear(vision_width, transformer_width)

        self.tdvit = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )
        self.time_encoding = AbsTimeEncoding(transformer_width, dropout=0., n_patches=1)
        self.cls_token = nn.Parameter(torch.empty(1, 1, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.vision_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # Text
        self.vocab_size = vocab_size
        self.textual = MLP(vocab_size, vocab_size*4, vocab_size)
        self.text_projection = nn.Parameter(torch.empty(vocab_size, embed_dim))
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.visual_embed.weight, std=self.vision_width ** -0.5)
        nn.init.normal_(self.cls_token, std=self.tdvit.width ** -0.5)
        nn.init.normal_(self.vision_projection, std=self.tdvit.width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.vocab_size ** -0.5)

        proj_std = (self.tdvit.width ** -0.5) * ((2 * self.tdvit.layers) ** -0.5)
        attn_std = self.tdvit.width ** -0.5
        fc_std = (2 * self.tdvit.width) ** -0.5
        for block in self.tdvit.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    @property
    def dtype(self):
        return self.visual_embed.weight.dtype

    def encode_image(self, images, padding, times):
        # images: b t d, padding: b t, times: b t
        B, T, *_ = images.shape
        x = self.visual_embed(images.type(self.dtype))        
        x = self.time_encoding(x, times)

        # Create attention mask from padding
        padding = torch.where(padding > 0, False, True) # True value indicates position not allowed to attend

        x = x.permute(1, 0, 2)  # btd -> tbd
        x = self.tdvit(x, key_padding_mask=padding, attn_mask=None)
        x = x.permute(1, 0, 2)  # tbd -> btd
        x = self.ln_final(x).type(self.dtype)
        # take features from the most recent embedding, which is the first in the sequence
        x = x[torch.arange(x.shape[0]), 0] 
        return x 

    def encode_text(self, text):
        x = text[:, 0] # b 1 d -> b d
        x = self.textual(x)
        return x
    
    def forward(self, image, text, padding, times):
        image_features = self.encode_image(image, padding, times)
        text_features = self.encode_text(text)

        # project to contrastive space
        image_features = image_features @ self.vision_projection
        text_features = text_features @ self.text_projection

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t() # cos(img, txt)
        logits_per_text = logits_per_image.t() # transpose to get text on first axis

        return logits_per_image, logits_per_text
    
    def load_encoders(self, ckpt: str):
        state_dict = torch.load(ckpt)['state_dict']
        self.load_state_dict(state_dict, strict=False)

class LinearClassifier(CLIP):
    """
    Concatenates image and text features from pretrained CLIP encoders
    Single linear layer maps concatenated features to classes (ie. multivariable regression)
    """
    def __init__(self,
        vision_width: int=512, # img dim
        vocab_size: int=13, # expr dim
        embed_dim: int=64,
        transformer_width: int=256,
        transformer_heads: int=4,
        transformer_layers: int=8,
        classifier_depth: int=2,
        **kwargs,
    ):
        super().__init__(vision_width, vocab_size, embed_dim, transformer_width, transformer_heads, transformer_layers, **kwargs)
        classifier_dim = transformer_width + vocab_size # concatenated image and text dim
        self.linear = nn.Linear(classifier_dim, 1)

    def forward(self, image, text, padding, times):
        image_features = self.encode_image(image, padding, times)
        text_features = self.encode_text(text)
        x = torch.cat([image_features, text_features], dim=-1) # b 2d
        x = self.linear(x)
        return x
    
    def freeze_encoders(self):
        for p in self.parameters():
            p.requires_grad = False
        # do not feeze mlp final
        for p in self.linear.parameters():
            p.requires_grad = True
    
    def set_encoder_lr_param_groups(self, lr: float):
        return [
            {'params': self.visual_embed.parameters(), 'lr': lr},
            {'params': self.tdvit.parameters(), 'lr': lr}, 
            {'params': self.time_encoding.parameters(), 'lr': lr}, 
            {'params': self.ln_final.parameters(), 'lr': lr}, 
            {'params': self.textual.parameters(), 'lr': lr},
            {'params': self.linear.parameters()}, # classifier uses default lr
        ]

class ZeroShotClassifier(CLIP):
    """
    Computes cosine similarity between image features and text features of each class
    Input expects each subject to have text from each class of shape (b c)
    The most probable (image, text) pair according to cosine similarity is the predicted class
    """
    def __init__(self,
        vision_width: int=512, # img dim
        vocab_size: int=13, # expr dim
        embed_dim: int=64,
        transformer_width: int=256,
        transformer_heads: int=4,
        transformer_layers: int=8,
        classifier_depth: int=2,
        **kwargs,
    ):
        super().__init__(vision_width, vocab_size, embed_dim, transformer_width, transformer_heads, transformer_layers, **kwargs)
        
    def encode_text(self, text):
        # text is (b c) where c is number of possible classes
        x = self.textual(text)
        return x
    
    def forward(self, image, text, padding, times):
        image_features = self.encode_image(image, padding, times) # b d
        text_features = self.encode_text(text) # b c d

        # project to contrastive space
        image_features = image_features @ self.vision_projection # b p
        text_features = torch.einsum('bcd, dp -> bcp', text_features, self.text_projection) # c=num classes
        
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=2, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = torch.einsum('bp, acp -> bac', logit_scale*image_features, text_features) # b b c
        logits = logits.diagonal(dim1=0, dim2=1).t() # b c
        
        return logits
    
    def freeze_encoders(self):
        for p in self.parameters():
            p.requires_grad = False

class Classifier(CLIP):
    def __init__(self,
        vision_width: int=512, # img dim
        vocab_size: int=13, # expr dim
        embed_dim: int=64,
        transformer_width: int=256,
        transformer_heads: int=4,
        transformer_layers: int=8,
        classifier_depth: int=2,
        n_classes: int=2,
        **kwargs,
    ):
        super().__init__(vision_width, vocab_size, embed_dim, transformer_width, transformer_heads, transformer_layers, **kwargs)
        classifier_dim = transformer_width + vocab_size # concatenated image and text dim
        # self.mlp_final = MLP(classifier_dim, classifier_dim, n_classes)
        layers = [classifier_dim*2 for _ in range(classifier_depth-1)] + [n_classes]
        self.mlp_final = DynamicMLP(classifier_dim, layers, activation=nn.ReLU, dropout=0.1)

    def forward(self, image, text, padding, times):
        image_features = self.encode_image(image, padding, times)
        text_features = self.encode_text(text)
        x = torch.cat([image_features, text_features], dim=-1) # b 2d
        x = self.mlp_final(x)
        return x
    
    def freeze_encoders(self):
        for p in self.parameters():
            p.requires_grad = False
        # do not feeze mlp final
        for p in self.mlp_final.parameters():
            p.requires_grad = True

    def set_encoder_lr_param_groups(self, lr: float):
        return [
            {'params': self.visual_embed.parameters(), 'lr': lr},
            {'params': self.tdvit.parameters(), 'lr': lr}, 
            {'params': self.time_encoding.parameters(), 'lr': lr}, 
            {'params': self.ln_final.parameters(), 'lr': lr}, 
            {'params': self.textual.parameters(), 'lr': lr},
            {'params': self.mlp_final.parameters()}, # classifier uses default lr
        ]

class JointConcatenated(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, image, text):
        x = torch.concat([image, text], dim=-1)
        return x

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "vision_projection", "cls_token"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()

def clip_base(**kwargs):
    return CLIP(
        transformer_heads=4,
        transformer_layers=8,
        **kwargs)
def clip_classifier(**kwargs):
    return Classifier(
        transformer_heads=4,
        transformer_layers=8,
        n_classes=2,
        **kwargs)
def clip_linear_classifier(**kwargs):
    return LinearClassifier(
        transformer_heads=4,
        transformer_layers=8,
        **kwargs)
def clip_zeroshot_classifier(**kwargs):
    return ZeroShotClassifier(
        transformer_heads=4,
        transformer_layers=8,
        **kwargs)

# single modality baselines
def tdvit(**kwargs):
    return TDViT(
        transformer_heads=4,
        transformer_layers=8,
        n_classes=2,
        **kwargs)
def img_classifier(**kwargs):
    return ImageClassifier(
        n_classes=2,
        **kwargs)
