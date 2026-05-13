import tinker
import dotenv
import uuid
import hashlib
from pydantic import BaseModel
from tinker.types import Datum, AdamParams, ForwardBackwardOutput, OptimStepResponse

from ip.utils import file_utils
from ip.external.tinker_driver.common import datum_from_tokens_weights
from ip.external.tinker_driver.renderers import get_renderer, TrainOnWhat, get_renderer_name_for_model
from ip.external.tinker_driver.tokenizer_utils import get_tokenizer
from ip.llm.data_models import Model
from ip.config import ROOT_DIR

dotenv.load_dotenv(ROOT_DIR / ".env")

class TrainConfig(BaseModel):
    base_model: str
    dataset_path: str
    batch_size: int
    n_epochs: int
    learning_rate: float
    lora_rank: int
    seed: int = 0
    slug: str | None = None
    
    def get_unsafe_hash(self, max_length: int = 16) -> str:
        dataset_hash = file_utils.get_hash(self.dataset_path)
        return hashlib.sha256(str((
            self.base_model,
            dataset_hash,
            self.batch_size,
            self.n_epochs,
            self.learning_rate,
            self.lora_rank,
            self.seed,
        )).encode()).hexdigest()[:max_length]
    
class TinkerModel(BaseModel):
    model_path: str
    train_config: TrainConfig
    
    @property
    def base_model(self) -> str:
        return self.train_config.base_model

def process_messages_to_datum(
    base_model: str,
    messages: list[dict[str, str]], 
    *,
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
) -> Datum:
    """Convert OpenAI-style messages to Tinker Datum format.
    """
    tokenizer = get_tokenizer(base_model)
    renderer = get_renderer(get_renderer_name_for_model(base_model), tokenizer)
    tokens, weights = renderer.build_supervised_example(messages, train_on_what=train_on_what)
    return datum_from_tokens_weights(tokens, weights)

async def _run_training_loop(
    training_client: "tinker.TrainingClient",
    config: TrainConfig,
    base_model_for_renderer: str,
) -> None:
    dataset = file_utils.read_jsonl(config.dataset_path)
    num_batches = (len(dataset) + config.batch_size - 1) // config.batch_size

    for _ in range(config.n_epochs):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, len(dataset))
            batch_samples = dataset[start_idx:end_idx]

            batch_data = []
            for sample in batch_samples:
                datum = process_messages_to_datum(base_model_for_renderer, sample["messages"])
                batch_data.append(datum)

            await training_client.forward_backward_async(batch_data, loss_fn="cross_entropy")
            await training_client.optim_step_async(
                AdamParams(learning_rate=config.learning_rate)
            )


async def train(
    config: TrainConfig,
    *,
    service_client: tinker.ServiceClient | None = None,
) -> Model:
    """Train a fresh LoRA from base. Returns a Model whose id is the sampler path."""
    model, _ = await train_with_state(config, service_client=service_client)
    return model


async def train_with_state(
    config: TrainConfig,
    *,
    service_client: tinker.ServiceClient | None = None,
) -> tuple[Model, str]:
    """Like train() but ALSO saves training state so the run can be continued
    (e.g. for BCT-style stacked finetunes). Returns (sampler Model, state_path).
    """
    service_client = service_client or tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.base_model, rank=config.lora_rank
    )

    await _run_training_loop(training_client, config, config.base_model)

    h = config.get_unsafe_hash(max_length=16)
    sampling_future = await training_client.save_weights_for_sampler_async(name=h)
    state_future = await training_client.save_state_async(name=f"{h}_state")
    sampler_path = sampling_future.result().path
    state_path = state_future.result().path

    model = Model(
        id=sampler_path,
        type="tinker",
        parent_model=Model(id=config.base_model, type="tinker"),
    )
    return model, state_path


async def train_resume(
    config: TrainConfig,
    *,
    state_path: str,
    service_client: tinker.ServiceClient | None = None,
) -> tuple[Model, str]:
    """Continue training from a previously-saved state. config.base_model is
    used only for the renderer (Tinker reads the actual base model from state).
    Returns (sampler Model, new state_path).
    """
    service_client = service_client or tinker.ServiceClient()
    training_client = await service_client.create_training_client_from_state_async(
        path=state_path,
    )

    await _run_training_loop(training_client, config, config.base_model)

    h = config.get_unsafe_hash(max_length=16)
    sampling_future = await training_client.save_weights_for_sampler_async(name=h)
    state_future = await training_client.save_state_async(name=f"{h}_state")
    sampler_path = sampling_future.result().path
    new_state_path = state_future.result().path

    model = Model(
        id=sampler_path,
        type="tinker",
        parent_model=Model(id=config.base_model, type="tinker"),
    )
    return model, new_state_path
    
if __name__ == "__main__":
    
    dataset = [{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}]
    file_utils.save_jsonl(dataset, "dataset.jsonl")
    
    import asyncio
    model_path = asyncio.run(train(
        config=TrainConfig(
            base_model="Qwen/Qwen3-4B-Instruct-2507", 
            dataset_path="dataset.jsonl", 
            batch_size=16, 
            n_epochs=1, 
            learning_rate=1e-4, 
            lora_rank=16,
            slug="gsm8k-spanish-french___frac-french=0__inoc-type=french",
        ),
    ))
    # Should print the appropriate path to the saved weights
    print(model_path)
    
    # cleanup 
    import os
    os.remove("dataset.jsonl")