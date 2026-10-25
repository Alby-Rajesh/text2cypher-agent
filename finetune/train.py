import argparse
import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

log = logging.getLogger("train")


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA fine-tuning for Text-to-Cypher.")
    ap.add_argument("--config", default="finetune/config.yaml")
    ap.add_argument("--max-steps", type=int, default=-1, help="Override for a quick smoke run.")
    ap.add_argument("--no-push", action="store_true", help="Do not push the adapter to the Hub.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_dir = Path(cfg["data"]["output_dir"])
    data = load_dataset(
        "json", data_files={"train": str(data_dir / "train.jsonl"), "eval": str(data_dir / "eval.jsonl")}
    )

    compute_dtype = torch.bfloat16 if cfg["training"].get("bf16") else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    lora = LoraConfig(task_type="CAUSAL_LM", bias="none", **cfg["lora"])
    hub = cfg.get("hub", {})
    training = SFTConfig(
        **cfg["training"],
        max_steps=args.max_steps,
        max_length=cfg["model"]["max_length"],
        completion_only_loss=True,
        model_init_kwargs={"dtype": compute_dtype, "device_map": "auto"},
        push_to_hub=bool(hub.get("push")) and not args.no_push,
        hub_model_id=hub.get("model_id"),
        hub_private_repo=hub.get("private", False),
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    trainer = SFTTrainer(
        model=cfg["model"]["base"],
        args=training,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        processing_class=tokenizer,
        peft_config=lora,
        quantization_config=quantization,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    log.info("final eval: %s", trainer.evaluate())
    trainer.save_model(training.output_dir)
    if training.push_to_hub:
        trainer.push_to_hub(commit_message="Text2Cypher QLoRA adapter")
        log.info("pushed adapter to https://huggingface.co/%s", training.hub_model_id)


if __name__ == "__main__":
    main()
