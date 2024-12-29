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
        if load_params and model_path is not None:
            if self.temperature > 0.0:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    trust_remote_code=True,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    trust_remote_code=True,
                    device_map="auto",
                    do_sample=True,
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path if tokenizer_path else model_path,
                use_fast=True,
                return_tensors="pt",
            )
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
