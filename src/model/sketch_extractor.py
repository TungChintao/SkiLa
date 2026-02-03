from transformers import CLIPVisionModel, CLIPProcessor, SiglipVisionModel
import torch
import torch.nn as nn

class SketchExtractor(nn.Module):
    def __init__(self, model_path, sketch_token_num, config, torch_dtype, attn_implementation, llm_hidden_dim=2048):
        super().__init__()
        # CLIP frozen
        self.clip_vision = CLIPVisionModel.from_pretrained(
            model_path,
            config=config.vision_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        ).eval()

        for param in self.clip_vision.parameters():
            param.requires_grad = False
        
        self.sketch_token_num = sketch_token_num

        # Projector trainable
        self.sketch_proj = nn.Linear(config.vision_config.hidden_size, llm_hidden_dim, dtype=torch_dtype)

    @torch.no_grad()
    def encode_clip(self, pixel_values):
        """
        Only return CLIP output's patch tokens, w/o CLS token
        """
        # outputs = self.clip_vision(pixel_values, output_hidden_states=True).hidden_states[-2]

        # return outputs[:,1:,:]

        outputs = self.clip_vision(pixel_values)
        if self.sketch_token_num == 1:
            return outputs.last_hidden_state[:, :1, :]
        return outputs.last_hidden_state[:, 1:, :]  # [B, num_patches, hidden_dim]

    def forward(self, pixel_values, n_sketch_tokens=576):
        """
        pixel_values: [B, 3, H, W] PIL->tensor
        n_sketch_tokens: map to sketch token
        """
        clip_tokens = self.encode_clip(pixel_values)  # [B, P, hidden_dim]
        
        sketch_tokens = self.sketch_proj(clip_tokens)  # [B, P, llm_hidden_dim]

        B, P, D = sketch_tokens.shape
      
        return sketch_tokens#.view(-1,D)


class SketchExtractor_Siglip(nn.Module):
    def __init__(self, model_path, sketch_token_num, config, torch_dtype, attn_implementation, llm_hidden_dim=2048):
        super().__init__()
        # Siglip frozen
        self.siglip2_vision = SiglipVisionModel.from_pretrained(
            model_path,
            config=config.vision_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation).eval()

        for param in self.siglip2_vision.parameters():
            param.requires_grad = False
        
        self.sketch_token_num = sketch_token_num

        # Projector trainable
        self.sketch_proj = nn.Linear(config.vision_config.hidden_size, llm_hidden_dim, dtype=torch_dtype)

    @torch.no_grad()
    def vision_encode(self, pixel_values):
        """
        Only return CLIP output's patch tokens, w/o CLS token
        """
        # outputs = self.siglip2_vision(pixel_values, output_hidden_states=True).hidden_states[-2]

        # return outputs[:,1:,:]

        outputs = self.siglip2_vision(pixel_values)

        if self.sketch_token_num == 1:
            return outputs.pooler_output.unsqueeze(1)
   
        return outputs.last_hidden_state  # [B, num_patches, hidden_dim]

    def forward(self, pixel_values, n_sketch_tokens=576):
        """
        pixel_values: [B, 3, H, W] PIL->tensor
        n_sketch_tokens: map to sketch token
        """
        clip_tokens = self.vision_encode(pixel_values)  # [B, P, hidden_dim]
        
        sketch_tokens = self.sketch_proj(clip_tokens)  # [B, P, llm_hidden_dim]

        B, P, D = sketch_tokens.shape

      
        return sketch_tokens#.view(-1,D)