import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import ClapModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_CLAP_LoRa():
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "intermediate.dense",
            "output.dense"
        ]
    )
    model = ClapModel.from_pretrained("laion/clap-htsat-fused")
    model.eval()
    model = get_peft_model(model, lora_config)
    model.to(DEVICE)
    return model