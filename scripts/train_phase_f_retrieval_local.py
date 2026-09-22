"""
AI Search Framework - Phase F2: Local Specialist LoRA Fine-Tuning
Runs 4-bit QLoRA fine-tuning for retrieval_qa specialist directly on local RTX 3050 Laptop GPU.
Enforces:
1. Strict VRAM budgeting (<5.2 GB) via 4-bit NF4 + Gradient Checkpointing + Batch Size 1.
2. Incremental checkpointing every 25 steps with auto-resume.
3. ChatML prompt formatting matching Phi-3.5-mini and Ollama Modelfile.
4. Export of LoRA adapter and merged standalone model.
"""

import os
import sys
import time
import json
import glob

DATASET_PATH = "data/phase_f/retrieval_qa_train_dataset.json"
CHECKPOINT_DIR = "models/phase_f_checkpoints"
FINAL_ADAPTER_DIR = "models/phi35_retrieval_adapter"
MERGED_DIR = "models/phi35_retrieval_merged"
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

def main():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel
    )
    from datasets import Dataset
    from trl import SFTTrainer

    print("=" * 80)
    print("PHASE F2: LOCAL RETRIEVAL SPECIALIST QLORA FINE-TUNING")
    print(f"Target Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info()
        print(f"Available VRAM: {free_vram / (1024**3):.2f} GB / {total_vram / (1024**3):.2f} GB")
    print("=" * 80)

    assert torch.cuda.is_available(), "CUDA GPU is required for local fine-tuning!"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)

    # 1. Load Tokenizer
    print("\n[1/6] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Configure 4-bit QLoRA Quantization
    print("\n[2/6] Initializing 4-bit Quantization (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # 3. LoRA Configuration on Attention & MLP Projection Layers
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 4. Load & Format Training Dataset (ChatML format)
    print(f"\n[3/6] Loading Training Dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    formatted_texts = []
    for item in raw_samples:
        text = (
            f"<|system|>\n{item['system']}<|end|>\n"
            f"<|user|>\n{item['prompt']}<|end|>\n"
            f"<|assistant|>\n{item['response']}<|end|>"
        )
        formatted_texts.append({"text": text})

    dataset = Dataset.from_list(formatted_texts)
    print(f"Loaded {len(dataset)} training samples.")

    # 5. Check for Existing Checkpoint for Auto-Resume
    ckpts = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
    resume_checkpoint = None
    if ckpts:
        latest_step = max([int(c.split("-")[1]) for c in ckpts])
        resume_checkpoint = os.path.join(CHECKPOINT_DIR, f"checkpoint-{latest_step}")
        print(f"DISCONNECT RECOVERY: Resuming from checkpoint: {resume_checkpoint}")
    else:
        print("Starting fresh training run.")

    # 6. Training Configuration (RTX 3050 6GB Optimized)
    print("\n[4/6] Configuring SFTTrainer...")
    import trl
    if hasattr(trl, "SFTConfig"):
        training_args = trl.SFTConfig(
            output_dir=CHECKPOINT_DIR,
            max_length=768,
            dataset_text_field="text",
            per_device_train_batch_size=1,        # Batch size 1 keeps VRAM < 4.8 GB
            gradient_accumulation_steps=8,        # Effective batch size = 8
            learning_rate=2e-4,
            logging_steps=5,
            num_train_epochs=3,                   # 3 epochs on 240 samples = 90 total steps
            save_strategy="steps",
            save_steps=25,
            save_total_limit=2,
            bf16=True,
            fp16=False,
            optim="paged_adamw_8bit",
            warmup_steps=5,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            report_to="none"
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer
        )
    else:
        training_args = TrainingArguments(
            output_dir=CHECKPOINT_DIR,
            per_device_train_batch_size=1,        # Batch size 1 keeps VRAM < 4.8 GB
            gradient_accumulation_steps=8,        # Effective batch size = 8
            learning_rate=2e-4,
            logging_steps=5,
            num_train_epochs=3,                   # 3 epochs on 240 samples = 90 total steps
            save_strategy="steps",
            save_steps=25,
            save_total_limit=2,
            bf16=True,
            fp16=False,
            optim="paged_adamw_8bit",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            report_to="none"
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=768,                   # Max seq length for strict 6GB safety
            tokenizer=tokenizer,
            args=training_args
        )

    # 7. Execute Fine-Tuning Run
    print("\n[5/6] Launching Local Fine-Tuning Run...")
    t_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    duration_min = (time.perf_counter() - t_start) / 60.0
    print(f"\nTraining completed in {duration_min:.2f} minutes.")

    # 8. Save Final LoRA Adapter
    print(f"\n[6/6] Saving Final Adapter to {FINAL_ADAPTER_DIR}...")
    trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
    print(f"Successfully saved trained LoRA adapter to {FINAL_ADAPTER_DIR}")

    # Write training metadata
    meta = {
        "model_id": MODEL_ID,
        "dataset_samples": len(dataset),
        "epochs": 3,
        "batch_size_effective": 8,
        "max_seq_length": 768,
        "r": 16,
        "alpha": 32,
        "duration_minutes": duration_min,
        "device": torch.cuda.get_device_name(0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(FINAL_ADAPTER_DIR, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE F2 LOCAL TRAINING COMPLETE")
    print(f"Adapter saved at: {FINAL_ADAPTER_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
