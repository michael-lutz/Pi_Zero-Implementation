"""Model instantiation based on config"""

from typing import Any, Dict


def get_model(config: Dict[str, Any]):
    """Barebones config-based model instantiation"""

    if config["model_name"] == "PiZero":
        from pi_zero_project.model.policy.pi_zero import PiZero

        return PiZero(
            vit_variant=config["vit_variant"],
            llm_vocab_size=config["llm_vocab_size"],
            gemma_mlp_dim=config["gemma_mlp_dim"],
            gemma_embed_dim=config["gemma_embed_dim"],
            action_expert_mlp_dim=config["action_expert_mlp_dim"],
            action_expert_embed_dim=config["action_expert_embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            num_kv_heads=config["num_kv_heads"],
            head_dim=config["head_dim"],
            dropout=config["dropout"],
        )
    else:
        raise ValueError("Unknown model name in config")
