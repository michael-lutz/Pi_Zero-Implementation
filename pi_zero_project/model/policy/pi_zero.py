"""Pi Zero Policy Implementation

There are many dimensions... For the reader's benefit:
- B: batch size
- I: number of input images
- T: number of input tokens
- A: number of actions to predict
- L: overall sequence length (L = I + T + 1 + A)
- D: hidden dimension
- H: hidden dimension per head

- num_heads: number of attention heads
- h_i, w_i: height and width of the image
- p: number of proprioceptive features
- a: size of action dimension
"""

from typing import Any, Dict, List, Sequence, Tuple, TypedDict
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from pi_zero_project.model.components.attention import Attention, apply_attention, make_attn_mask
from pi_zero_project.model.components.linear import FeedForward
from pi_zero_project.model.components.norms import RMSNorm
from pi_zero_project.model.components.token_embed import Embedder
from pi_zero_project.model.policy.action_embedding import ActionEmbedder
from pi_zero_project.model.vlm.img_model import vit


class MixtureSpec(TypedDict):
    name: str
    mlp_dim: int
    embed_dim: int


class MoEBlock(nn.Module):
    """Mixture of Experts Transformer block.

    Attributes:
        mixture_specs: List of dictionaries containing specifications for each mixture.
                       Each dictionary should have keys: 'name', 'mlp_dim', 'embed_dim'.
        num_heads: number of attention heads
        num_kv_heads: number of key/value heads
        head_dim: dimension of each attention head
        query_pre_attn_norm: normalization strategy before attention
        attn_logits_softcap: soft cap for attention logits
        post_norms: whether to apply post-attention norms
        dropout: dropout rate
        dropout_bdims: dimensions for dropout
        cache_dtype: data type for cache
    """

    mixture_specs: List[MixtureSpec]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    query_pre_attn_norm: str
    attn_logits_softcap: float | None
    post_norms: bool
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    cache_dtype: str | None = None

    def setup(self) -> None:
        self.pre_attention_norms = {
            spec["name"]: RMSNorm(name=f"{spec['name']}_pre_attn_norm")
            for spec in self.mixture_specs
        }

        self.attentions = {
            spec["name"]: Attention(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                features=spec["embed_dim"],
                head_dim=self.head_dim,
                cache_dtype=self.cache_dtype,
                query_pre_attn_norm=self.query_pre_attn_norm,
                attn_logits_softcap=self.attn_logits_softcap,
                name=f"{spec['name']}_attn",
            )
            for spec in self.mixture_specs
        }

        self.pre_ffw_norms = {
            spec["name"]: RMSNorm(name=f"{spec['name']}_pre_ffw_norm")
            for spec in self.mixture_specs
        }
        self.mlps = {
            spec["name"]: FeedForward(features=spec["embed_dim"], hidden_dim=spec["mlp_dim"])
            for spec in self.mixture_specs
        }

        if self.dropout:
            self.drop = nn.Dropout(self.dropout, self.dropout_bdims)
        else:
            self.drop = lambda x, _: x
        if self.post_norms:
            self.post_attention_norms = {
                spec["name"]: RMSNorm(name=f"{spec['name']}_post_attn_norm")
                for spec in self.mixture_specs
            }
            self.post_ffw_norms = {
                spec["name"]: RMSNorm(name=f"{spec['name']}_post_ffw_norm")
                for spec in self.mixture_specs
            }

    def visualize_attention_mask(self, attn_mask: jax.Array) -> None:
        """Visualize the attention mask.

        Args:
            attn_mask: [B, 1, L, S] attention mask
        """
        pass

    def _process_attention_output(
        self, attn_output: jax.Array, residual: jax.Array, deterministic: bool, mixture_name: str
    ) -> jax.Array:
        """Standard attention processing with residual connections, norms, and MLPs.

        Args:
            attn_output: [N, L, D] attention output
            residual: [N, L, D] residual connection
            deterministic: whether to use dropout
            name: name of the mixture
        Returns:
            [N, L, D] processed attention output
        """
        if self.post_norms:
            attn_output = self.post_attention_norms[mixture_name](attn_output)
        attn_output = self.drop(attn_output, deterministic)
        attn_output += residual
        attn_output = self.pre_ffw_norms[mixture_name](attn_output)
        attn_output = self.mlps[mixture_name](attn_output)
        attn_output = self.drop(attn_output, deterministic)
        if self.post_norms:
            attn_output = self.post_ffw_norms[mixture_name](attn_output)
        return attn_output + residual

    def __call__(
        self,
        x: List[Tuple[str, jax.Array]],
        attn_mask: jax.Array,
        use_kv_cache: bool,
        deterministic: bool,
    ) -> List[Tuple[str, jax.Array]]:
        """Transformer block forward pass.

        Args:
            x: list of (mixture name, [B, L, D] input embeddings)
            attn_mask: [B, 1, L, S] attention mask
            use_kv_cache: whether to use kv-cache
            deterministic: whether to use dropout
        Returns:
            [B, L, D] output embeddings
        """
        mixtures_shapes = []
        all_q, all_k, all_v, all_positions = [], [], [], []
        for i, (mixture_name, x_mixture) in enumerate(x):
            mixtures_shapes.append(x_mixture.shape[1])
            inputs_normalized = self.pre_attention_norms[mixture_name](x_mixture)  # [B, L, D]

            start_idx = sum(mixtures_shapes[:i])
            end_idx = start_idx + mixtures_shapes[i]
            positions = jnp.arange(start_idx, end_idx).astype(jnp.int32)[None, :]

            q_mixture, k_mixture, v_mixture = self.attentions[mixture_name].get_qkv(
                inputs_normalized,
                positions,
                cache_size=attn_mask.shape[-1],
                decode=use_kv_cache,
            )  # 3 x [B, L, H]

            all_q.append(q_mixture)
            all_k.append(k_mixture)
            all_v.append(v_mixture)
            all_positions.append(positions)

        all_q = jnp.concatenate(all_q, axis=1)
        all_k = jnp.concatenate(all_k, axis=1)
        all_v = jnp.concatenate(all_v, axis=1)
        # all [B, L, H]
        all_positions = jnp.concatenate(all_positions, axis=1)  # [B, L]

        # Reorder the attention mask according to the positions
        sorted_indices = jnp.argsort(all_positions, axis=1)
        reordered_attn_mask = jnp.take_along_axis(
            attn_mask, sorted_indices[:, None, :, None], axis=2  # reordering rows
        )

        # Apply attention using the reordered attention mask in a single operation
        attn_output = apply_attention(
            all_q,
            all_k,
            all_v,
            reordered_attn_mask,
            num_kv_heads=self.num_kv_heads,
            attn_logits_softcap=self.attn_logits_softcap,
        )

        # Breaking down to the individual mixtures, projecting back to respective embed dims
        output = []
        for i, (mixture_name, x_mixture) in enumerate(x):
            # isolate for mixture and project back to embed dim
            start_idx = sum(mixtures_shapes[:i])
            end_idx = start_idx + mixtures_shapes[i]
            attn_output_mixture = self.attentions[mixture_name].proj_to_embed_dim(
                attn_output[:, start_idx:end_idx, :]
            )

            # residual connection + MLP
            output.append(
                (
                    mixture_name,
                    self._process_attention_output(
                        attn_output_mixture, x_mixture, deterministic, mixture_name
                    ),
                )
            )

        return output


class PiZero(nn.Module):
    """Pi_Zero Policy Implementation"""

    # Initialization parameters
    vit_variant: str
    llm_vocab_size: int

    # Attention parameters
    gemma_mlp_dim: int
    gemma_embed_dim: int
    action_expert_mlp_dim: int
    action_expert_embed_dim: int
    depth: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    query_pre_attn_norm: str = "rsqrt_head_dim"
    attn_logits_softcap: float = 0.0
    post_norms: bool = False

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    cache_dtype: str | None = None

    embed_dtype: str = "float32"
    scan: bool = False
    remat_policy: str = "none"

    @nn.compact
    def __call__(
        self,
        images: jax.Array,
        text: jax.Array,
        proprio: jax.Array,
        action: jax.Array,
        timesteps: jax.Array,
        *,
        inference_mode: bool = False,
        deterministic: bool = True,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Primary call function for Pi_Zero

        Args:
            images: [B, I, h_i, w_i, 3] images
            text: [B, T] text tokens
            proprio: [B, P, p] proprioceptive features (P is assumed to be 1 in the paper)
            action: [B, A, a] action to predict
            timesteps: [B] timesteps
            inference_mode: bool, whether to return the output of the policy

        Returns:
            [B, A, a] output of the policy
        """
        assert images.shape[0] == text.shape[0] == proprio.shape[0] == action.shape[0]
        assert len(images.shape) == 5, "images must be [B, I, h_i, w_i, 3]"
        assert len(text.shape) == 2, "text must be [B, T]"
        assert len(proprio.shape) == 3, "proprio must be [B, P, p]"
        assert len(action.shape) == 3, "action must be [B, A, a]"
        assert len(timesteps.shape) == 1, "timesteps must be [B]"
        out = {}
        B = images.shape[0]
        A = action.shape[1]
        a = action.shape[2]
        P = proprio.shape[1]
        I = images.shape[1]
        T = text.shape[1]

        action_token_embed = ActionEmbedder(
            embed_dim=self.action_expert_embed_dim, name="action_embedder"
        )(
            action, timesteps
        )  # [B, A, D]

        if (
            inference_mode
            and self.has_variable("cache", "o_keys")
            and self.has_variable("cache", "o_values")
        ):
            # Can skip ahead to attention if we have cache
            # TODO: implement fast and cached inference
            pass

        proprio_token_embed = nn.Dense(
            features=self.action_expert_embed_dim, name="proprio_embedder"
        )(proprio).astype(
            self.embed_dtype
        )  # [B, P, D]

        images = images.reshape(B * I, *images.shape[2:])
        vit_config = vit.decode_variant(self.vit_variant)
        image_token_embed, aux = vit.ViT(
            num_classes=self.gemma_embed_dim, name="img", **vit_config
        )(
            images
        )  # [B * I, D]
        del aux
        images = images.reshape(B, I, *images.shape[1:])
        image_token_embed = image_token_embed.reshape(B, I, self.gemma_embed_dim)
        image_token_embed = image_token_embed.astype(self.embed_dtype)

        if text.shape[1] > 0:
            embedder = Embedder(
                vocab_size=self.llm_vocab_size,
                embed_dim=self.gemma_embed_dim,
                name="gemma_embedder",
            )
            text_token_embed = embedder.encode(text).astype(self.embed_dtype)  # [B, T, D]
        else:
            text_token_embed = jnp.zeros((B, 0, self.gemma_embed_dim))

        # run attention (need to pass in)
        L = I + T + P + A

        # Applying block-wise causal mask as in the paper
        input_mask = np.ones([B, L])
        # NOTE: if images are missing, assume we populate them with all 0
        img_mask = images.any(axis=(-3, -2, -1))
        input_mask[:, :I] = input_mask[:, :I] * img_mask
        mask_ar = np.zeros([B, L])
        mask_ar[:, I + T] = 1
        mask_ar[:, I + T + P] = 1
        input_mask = jnp.array(input_mask)
        mask_ar = jnp.array(mask_ar)
        attn_mask = make_attn_mask(input_mask, mask_ar)
        attn_mask = attn_mask[:, None, :, :]

        if self.remat_policy == "none":
            block_cls = MoEBlock
        else:
            block_cls = nn.remat(
                MoEBlock,
                prevent_cse=not self.scan,
                static_argnums=(8, 9),  # 0=self, 8=use_kv_cache, 9=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy),
            )

        block_kw = dict(
            mixture_specs=[
                {"name": "gemma", "mlp_dim": self.gemma_mlp_dim, "embed_dim": self.gemma_embed_dim},
                {
                    "name": "action_expert",
                    "mlp_dim": self.action_expert_mlp_dim,
                    "embed_dim": self.action_expert_embed_dim,
                },
            ],
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            query_pre_attn_norm=self.query_pre_attn_norm,
            attn_logits_softcap=self.attn_logits_softcap,
            post_norms=self.post_norms,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            cache_dtype=self.cache_dtype,
        )
        layers = self.scope.push("layers")  # type: ignore
        if self.scan:
            blocks = [
                nn.scan(
                    block_cls,
                    variable_axes={"params": 0, "cache": 1},
                    split_rngs={"params": True, "dropout": True},
                    in_axes=nn.broadcast,  # type: ignore
                    length=self.depth,
                )(
                    parent=layers, **block_kw  # type: ignore
                )
            ]
        else:
            blocks = [
                block_cls(parent=layers.push(str(layer)), **block_kw)  # type: ignore
                for layer in range(self.depth)
            ]

        x = [
            ("gemma", jnp.concatenate([image_token_embed, text_token_embed], axis=1)),
            ("action_expert", jnp.concatenate([proprio_token_embed, action_token_embed], axis=1)),
        ]

        out["pre_attention_embeddings"] = x
        for block in blocks:
            x = block(
                x=x,
                attn_mask=attn_mask,
                use_kv_cache=inference_mode,
                deterministic=deterministic,
            )

        out["post_attention_embeddings"] = x

        for i, (mixture_name, x_mixture) in enumerate(x):
            x[i] = (mixture_name, RMSNorm(name=f"{mixture_name}_final_norm")(x_mixture))
        out["final_normed_embeddings"] = x

        action_embeddings = x[-1][1][:, -A:, :]
        action_field_pred = nn.Dense(features=a, name="proj_action_dim")(action_embeddings)

        return action_field_pred, out
