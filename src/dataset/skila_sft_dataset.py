import re
import os
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from qwen_vl_utils import process_vision_info
from .skila_data_utils import *


# ========== Dataset Class ==========
class SupervisedDatasetSkiLa(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.exclude = ('3D', 'Chemistry', 'Competitive', 'Graph', 'Physics', 'ARC-AGI', 'Ciphers', 'Tetris')
        self.raw_dataset = self.load_zebra_cot_dataset(data_root)

        
    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, i):
        """Simply return raw sample - no preprocessing here."""
        return self.raw_dataset[i]
    
    def load_zebra_cot_dataset(self, data_root):
        """Load all parquet files from Zebra-CoT dataset directory."""
        all_datasets = []
        
        for subdir in os.listdir(data_root):
            if any(x in subdir for x in self.exclude):
                continue
            subdir_path = os.path.join(data_root, subdir)
            
            if not os.path.isdir(subdir_path):
                continue
            
            logging.info(f"Loading dataset from: {subdir}")
            
            try:
                dataset = load_dataset(subdir_path, split='train')
                all_datasets.append(dataset)
                logging.info(f"  Loaded {len(dataset)} samples from {subdir}")
            except Exception as e:
                logging.warning(f"  Error loading {subdir}: {e}")
                continue
        
        if all_datasets:
            combined_dataset = concatenate_datasets(all_datasets)
            logging.info(f"Total samples: {len(combined_dataset)}")
            return combined_dataset
        else:
            raise ValueError("No datasets loaded!")

# ========== data processing ==========

def parse_reasoning_trace(text_reasoning_trace):
    """Parse reasoning trace to extract thoughts and image placeholders."""
    parts = []
    segments = re.split(r'(<image_start>\[reasoning_image_\d+\]<image_end>)', text_reasoning_trace)
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        image_match = re.match(r'<image_start>\[(reasoning_image_\d+)\]<image_end>', segment)
        if image_match:
            parts.append(('image', image_match.group(1)))
        else:
            cleaned_text = re.sub(r'THOUGHT \d+:\s*', '', segment)
            if cleaned_text:
                parts.append(('think', cleaned_text))
    return parts


def zebra_cot_preprocess_function(sample, max_pixels=1920*28*28, pattern='vl'):
    """Convert Zebra-CoT format to training format."""
    user_content = []
    
    # Extract problem images
    question_text = sample['Question']
    problem_image_pattern = r'<image_start>\[(problem_image_\d+)\]<image_end>'
    problem_images = re.findall(problem_image_pattern, question_text)
    clean_question = re.sub(problem_image_pattern, '', question_text)#+USER_PROMPT
    user_content.append({"type": "text", "text": clean_question})
    
    for img_key in problem_images:
        if img_key in sample and sample[img_key] is not None:
            user_content.append({
                "type": "image", 
                "image": sample[img_key],
                "max_pixels": max_pixels
            })
    
    # Process assistant output
    assistant_content = []
    reasoning_parts = parse_reasoning_trace(sample['Text Reasoning Trace'])
    

    if pattern == 'vl':
        for part_type, content in reasoning_parts:
            if part_type == 'think':
                content = content.replace('\n\n', ' ').replace('\n', ' ')
                assistant_content.append({
                    "type": "text",
                    "text": f"<think>{content}</think>"
                })
            elif part_type == 'image':
                if content in sample and sample[content] is not None:
                    assistant_content.append({
                        "type": "text",
                        "text": f"\n"
                    })
                    assistant_content.append({
                        "type": "image",
                        "image": sample[content]
                    })
                    assistant_content.append({
                        "type": "text",
                        "text": f"\n"
                    })
    # ======================= Think Process ==============================
    elif pattern == 'v':
        for part_type, content in reasoning_parts:
            if part_type == 'think':
                continue
            elif part_type == 'image':
                if content in sample and sample[content] is not None:
                    assistant_content.append({
                        "type": "image",
                        "image": sample[content]
                    })
                    assistant_content.append({
                        "type": "text",
                        "text": f"\n"
                    })
    # ====================================================================
    final_answer = sample.get('Final Answer', '')
    assistant_content.append({
        "type": "text",
        "text": f"<answer>{final_answer}</answer>"
    })
    
    return [
        # {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]


# ==========  Collator ==========

class SkiLaDataCollator:
    """Collate examples for Stage 1 training with batch-level optimization."""
    
    def __init__(self, processor, sketch_processor, args):
        self.processor = processor
        self.sketch_processor = sketch_processor
        self.args = args
        
        # Precompute token IDs once
        self.sketch_token_idx = processor.tokenizer("<|skila|>", return_tensors="pt")["input_ids"][0]
        self.sketch_start_idx = processor.tokenizer("<|sketch_start|>", return_tensors="pt")["input_ids"][0]
        self.sketch_end_idx = processor.tokenizer("<|sketch_end|>", return_tensors="pt")["input_ids"][0]
        self.pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
        self.answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    
    def __call__(self, raw_examples):
        """Process batch of raw examples."""

        examples = [zebra_cot_preprocess_function(ex, self.args.image_max_pixels, self.args.pattern) for ex in raw_examples]

        texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in examples]

        texts = replace_visual_spectial_tokens(texts)

        image_inputs, _ = process_vision_info(examples)
        
        user_examples = remove_assistant_images(examples)
        user_texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in user_examples]
        user_image_inputs, _ = process_vision_info(user_examples)
        
        assistant_examples = remove_user_images(examples)
        assistant_texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in assistant_examples]
        assistant_texts = replace_visual_spectial_tokens(assistant_texts)
        assistant_image_inputs, _ = process_vision_info(assistant_examples)
        
        user_batch = self.processor(text=user_texts, images=user_image_inputs, return_tensors="pt", padding=True)
        assistant_batch = self.processor(text=assistant_texts, images=assistant_image_inputs, return_tensors="pt", padding=True)
        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        
        batch['pixel_values'] = user_batch.get('pixel_values', None)
        batch['image_grid_thw'] = user_batch.get('image_grid_thw', None)
        sketch_pixel_values = assistant_batch.get('pixel_values', None)
        sketch_image_grid_thw = assistant_batch.get('image_grid_thw', None)

        new_input_ids, new_attention_mask = process_batch(
            batch["input_ids"], batch["attention_mask"], 
            self.sketch_start_idx, self.sketch_end_idx,
            self.sketch_token_idx, self.args.sketch_token_num, 
            self.answer_start_token_pattern, self.pad_token_idx
        )
        batch["input_ids"] = new_input_ids
        batch["attention_mask"] = new_attention_mask

        labels = generate_labels_after_multi_token_start(
            batch["input_ids"], self.answer_start_token_pattern, 
            self.pad_token_idx, self.sketch_token_idx
        )
        batch["labels"] = labels
        
        if sketch_pixel_values is not None:
            sketch_image_mask = mask_image_output_tokens(
                batch["input_ids"], self.sketch_start_idx, self.sketch_token_idx
            )
            batch["sketch_image_mask"] = sketch_image_mask
            
            sketch_images = [
                item["image"] 
                for conv in examples 
                for msg in conv 
                if msg["role"] == "assistant" 
                for item in msg["content"] 
                if item["type"] == "image"
            ]
      
            sketch_inputs = self.sketch_processor(images=sketch_images, do_convert_rgb=True, return_tensors="pt")
            batch['sketch_pixel_values'] = sketch_inputs['pixel_values']
            
        return batch


def make_supervised_data_module_skila(processor, sketch_processor, args):
    """Make dataset and collator for Zebra-CoT training."""
    
    dataset = SupervisedDatasetSkiLa(data_root=args.data_path)

    data_collator = SkiLaDataCollator(
        processor=processor,
        sketch_processor=sketch_processor,
        args=args
    )
    
    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator
    )

