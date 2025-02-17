from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trainer import TTCGRPOTrainer

from reward import ThinkingCompletionRewardFunction
from dataloader import SelfAdaptingDataset

def main():
    # Load base model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load and prepare dataset
    dataset = load_dataset("lordspline/arc-agi")
    adapted_dataset = SelfAdaptingDataset(dataset)
    transformed_dataset = adapted_dataset.get_dataset()

    # Initialize reward function
    reward_function = ThinkingCompletionRewardFunction().calculate_reward

    # Training arguments
    training_args = GRPOConfig(
        output_dir="./arc_agi_grpo_output",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_completion_length=4096,
        max_prompt_length=18000,
        learning_rate=1e-4,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        num_generations=14,  # Number of generations per prompt
        temperature=0.7,
        beta=0.04,  # GRPO beta parameter
        use_vllm=True,  # Enable vLLM for faster generation
        vllm_gpu_memory_utilization=0.85,
        report_to="wandb",
        remove_unused_columns=False,
        log_completions=True,  # Log completions to wandb
        fp16=True,
        bf16=False,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=transformed_dataset["training"],
        eval_dataset=transformed_dataset["evaluation"],
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model("./arc_agi_grpo_final")

if __name__ == "__main__":
    main()