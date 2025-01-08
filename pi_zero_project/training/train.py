"""Train the model"""

import jax
import jax.numpy as jnp
import optax
import wandb

from pi_zero_project.training.data_utils.jax_dataloader import NaiveDataLoader
from pi_zero_project.training.data_utils.jax_lerobot_dataset import JaxLeRobotDataset
from pi_zero_project.training.model_from_config import get_model
from pi_zero_project.training.objectives.flow_matching_action import train_step

checkpoint_dir = "../../checkpoints/"

# Configuration for model selection
config = {
    "model_name": "PiZero",
    "vit_variant": "B/16",
    "llm_vocab_size": 30522,
    "gemma_mlp_dim": 2048,
    "gemma_embed_dim": 768,
    "action_expert_mlp_dim": 2048,
    "action_expert_embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "num_kv_heads": 12,
    "head_dim": 64,
    "dropout": 0.1,
}

log_every_n_steps = 1

# Load dataset
prng_key = jax.random.PRNGKey(0)

delta_timestamps = {
    "observation.image": [-0.1, 0.0],
    "observation.state": [-0.2, -0.1, 0.0],  # 3 for acceleration
    "action": [0.1, 0.2, 0.3, 0.4, 0.5],
}

dataset = JaxLeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
dataloader = NaiveDataLoader(prng_key, dataset, batch_size=128)

# Initialize wandb
wandb.init(project="pi_zero_project", config=config)


model = get_model(config)

# Optimizer setup
optimizer = optax.adam(learning_rate=1e-3)

# Initialize parameters and optimizer state
params = model.init(
    jax.random.PRNGKey(0),
    jnp.ones((128, 2, 96, 96, 3)),  # images
    jnp.empty((128, 0), dtype=jnp.int32),  # text
    jnp.ones((128, 3, 2)),  # proprio
    jnp.ones((128, 5, 2)),  # action
    jnp.ones((128,)),  # timesteps
)
opt_state = optimizer.init(params)

# Example training loop
prng_key = jax.random.PRNGKey(0)
for epoch in range(10):  # Number of epochs
    for i, batch in enumerate(dataloader):
        prng_key, params, opt_state, loss = train_step(
            prng_key=prng_key,
            params=params,
            opt_state=opt_state,
            images=batch["image"],
            text=batch["text"],
            proprio=batch["proprio"],
            action=batch["action"],
            model=model,
            optimizer=optimizer,
        )

        # Log the loss to wandb
        if i % log_every_n_steps == 0:
            wandb.log({"epoch": epoch, "loss": loss})
            print(f"Epoch {epoch}, Loss: {loss}")

# Finish the wandb run
wandb.finish()
