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
    """Provides interface for a LLM wrapped with JsonFormer for material Science applications.

    PARAMS:
        model_path [str]: path to both model and tokenizer in pytorch / huggingface format.
        load_params [bool]: if the model should load parameters. if they will be restored from a checkpoint anyways.

    Use `JsonformerModel.load_from_checkpoint(PATH)`
    to load from a checkpoint.
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
        tokenizer_path: str | None = None,
        load_params: bool = True,
    ):
        super().__init__()
        if load_params and model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                trust_remote_code=True,
                device_map="auto",
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
                "JsonformerModel is initialized without underlying generative model or tokenizer."
            )

        # self.prompt = "Generate the information of used parameters for the reaction based on the following schema:"
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

    def forward(self, text: str) -> dict:
        # https://github.com/1rgs/jsonformer/blob/main/jsonformer/main.py#L240C13-L240C13
        return Jsonformer(self.model, self.tokenizer, self.schema, text)()

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
