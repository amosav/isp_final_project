
from peft import get_peft_model, LoraConfig

def get_CLAP_LORa():
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_fc"],
        lora_dropout=0.1,
        modules_to_save=['trainable_adapters'],  # Add your list of trainable adapters here
    )