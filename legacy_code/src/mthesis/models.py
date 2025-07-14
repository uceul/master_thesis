import lightning.pytorch as pl
import torch.nn.functional as F
import torch
from jsonformer import Jsonformer
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from torch.optim import AdamW


class JsonformerHFModel(PreTrainedModel):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

class JsonformerModel(pl.LightningModule):
    def __init__(
        self,
        model_path: str | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
        prompt: str  = "",
        temperature: float = 0.1,
        tokenizer_path: str | None = None,
        load_params: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.model_name = model_name
        if load_params and model_path is not None:
            model_kwargs = {
                "torch_dtype": torch.float16,
                "load_in_8bit": False,
                "trust_remote_code": True,
                "device_map": "auto",
            }

            # Add the attn_implementation parameter conditionally
            if self.model_name == "Phi 3 Mini 4k Instruct":
                model_kwargs["attn_implementation"] = "flash_attention_2"

            if self.temperature > 0.0:
                model_kwargs["do_sample"] = True
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
            else:
                model_kwargs["do_sample"] = False
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
            print("model kwargs: " + str(model_kwargs))
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path if tokenizer_path else model_path,
                use_fast=True,
                return_tensors="pt",
            )

            if self.model_name == "Phi 3 Mini 4k Instruct":
                # We have to use the max vocab size of 32064 for Phi 3 to work with JSONFormer
                # For some reason only 32011 tokens are defined by default => Add 53 unused tokens 
                for i in range(53):
                    self.tokenizer.add_special_tokens({'additional_special_tokens': [f'[PAD{i}]']})
        else:
            self.model = None
            self.tokenizer = None
            log.warning(
                "JsonformerModel initialized without underlying generative model or tokenizer."
            )

        self.schema = {
            "type": "object",
            "properties": {
                "additive": {"type": "string"},
                "solvent": {"type": "string"},
                "temperature": {"type": "number"},
                "temperature_unit": {"type": "string"},
                "time": {"type": "number"},
                "time_unit": {"type": "string"},
            },
        }
        self.instruction = prompt 
        
    def forward(self, text: str) -> dict:
        # Combine instruction with the actual text
        full_prompt = f"{self.instruction}\n{text}"
        # Create Jsonformer instance and generate
        with torch.inference_mode():
            if self.temperature > 0.0:
                jsonformer = Jsonformer(self.model, self.tokenizer, self.schema, full_prompt, temperature=self.temperature)
            else:
                jsonformer = Jsonformer(self.model, self.tokenizer, self.schema, full_prompt)
        return jsonformer()

    def forward2(self, text: str):
        input_tokens = self.tokenizer.encode(text, return_tensors="pt").to(
            self.model.device
        )
        return self.model(text)

    def training_step(self, batch, batch_idx):
        # TODO: FIXME: calculate loss properly, and only for labels
        x, l = batch["input"], batch["label"]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.trainer.model.parameters,
            lr=5e-4,
        )
        return optimizer
