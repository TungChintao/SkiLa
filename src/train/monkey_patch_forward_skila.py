import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl


def replace_qwen2_5_with_skila_forward():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_skila_forward


from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None
    sketch_hidden_state: Optional[Tuple[torch.FloatTensor]] = None


def qwen2_5_skila_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,  # ===================================
    sketch_embedding: Optional[torch.Tensor] = None, # Sketch Tokens
    sketch_image_mask: Optional[torch.Tensor] = None, # Sketch Token Mask
    in_sketch_mode: Optional[torch.Tensor] = None,
    sketch_hidden_state: Optional[torch.Tensor] = None
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    ```"""
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)

        if in_sketch_mode is not None and in_sketch_mode.any():
            sketch_hidden_state = sketch_hidden_state.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds[in_sketch_mode, -1, :] = sketch_hidden_state[in_sketch_mode]

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
    
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        
        # inject sketch embedding
        if sketch_embedding is not None:
            sketch_embedding = sketch_embedding.type(self.visual.dtype)
            B, N, D = sketch_embedding.shape  
            sketch_image_embeds = sketch_embedding.view(-1, D)  
            
            each_n_image_tokens = (input_ids == self.config.skila_id).sum(dim=1)  
            n_image_tokens = each_n_image_tokens.sum().item()                   
            n_image_features = sketch_image_embeds.shape[0]                     
            
            # reasoning image token num != sketch token num
            if n_image_tokens != n_image_features:
                assert n_image_tokens < n_image_features
                batch_size = input_ids.shape[0]
                each_n_images = each_n_image_tokens // self.model.config.sketch_token_num  # [8,12] // 4 = [2, 3]
                each_n_features = each_n_images * N  # [2*576, 3*576]
                
                cumsum_features = torch.cat([torch.tensor([0], device=each_n_features.device), 
                                            each_n_features.cumsum(dim=0)])
                
                sketch_embeds_list = []
                for i in range(batch_size):
                    start_idx = cumsum_features[i].item()
                    end_idx = cumsum_features[i + 1].item()
                    sketch_features = sketch_image_embeds[start_idx:end_idx] 
                    
                    num_target_tokens = each_n_image_tokens[i].item()
                    
                    if self.model.config.compress_strategy == "average":
                        res = sketch_features.shape[0] % num_target_tokens
                        if res > 0:
                            sketch_features = sketch_features[:-res]
                        
                        group_size = sketch_features.shape[0] // num_target_tokens
                        grouped_features = sketch_features.reshape(num_target_tokens, group_size, D).mean(dim=1)
                        sketch_embeds_list.append(grouped_features)
                    else:
                        sketch_embeds_list.append(sketch_features[:num_target_tokens])
                
                sketch_image_embeds = torch.cat(sketch_embeds_list, dim=0)

                n_image_features = sketch_image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Sketch Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

            mask = input_ids == self.config.skila_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            sketch_image_embeds = sketch_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, sketch_image_embeds)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
    
    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    
    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    sketch_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)
  
    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    
    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
        inputs_embeds=inputs_embeds,
        sketch_hidden_state =sketch_hidden_state
    )