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

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from pi_zero_project.model.components.attention import Attention, apply_attention, make_attn_mask
from pi_zero_project.model.components.linear import FeedForward
from pi_zero_project.model.components.norms import RMSNorm
from pi_zero_project.model.components.token_embed import Embedder
from pi_zero_project.model.policy.action_embedding import ActionEmbedder
from pi_zero_project.model.vlm.img_model import vit


class MoEBlock(nn.Module):
    """Mixture of Experts Transformer block.

    Attributes:
        gemma_num_heads: number of attention heads
        gemma_num_kv_heads: number of key/value heads
        gemma_embed_dim: embedding dimension
        gemma_head_dim: dimension of each attention head
        gemma_hidden_dim: hidden dimension
        embed_dim: embedding dimension
        action_expert_num_heads: number of attention heads
        action_expert_num_kv_heads: number of key/value heads
        action_expert_embed_dim: embedding dimension
        action_expert_head_dim: dimension of each attention head
        action_expert_hidden_dim: hidden dimension
    """

    # Following the paper, "width" and "mlp dim" can be changed to be different for gemma and action expert
    gemma_mlp_dim: int
    gemma_embed_dim: int
    action_expert_mlp_dim: int
    action_expert_embed_dim: int

    # Joining other parameters for simplicity (and necessity in the case of head count)
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
        self.pre_attention_norm = RMSNorm(name="gemma_layers_pre_attn_norm")
        self.gemma_attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            features=self.gemma_embed_dim,
            head_dim=self.head_dim,
            cache_dtype=self.cache_dtype,
            query_pre_attn_norm=self.query_pre_attn_norm,
            attn_logits_softcap=self.attn_logits_softcap,
            name="gemma_layers_attn",
        )

        self.action_expert_attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            features=self.action_expert_embed_dim,
            head_dim=self.head_dim,
            cache_dtype=self.cache_dtype,
            query_pre_attn_norm=self.query_pre_attn_norm,
            attn_logits_softcap=self.attn_logits_softcap,
            name="action_expert_layers_attn",
        )
        self.pre_ffw_norm = RMSNorm(name="gemma_layers_pre_ffw_norm")
        self.gemma_mlp = FeedForward(features=self.gemma_embed_dim, hidden_dim=self.gemma_mlp_dim)
        self.action_expert_mlp = FeedForward(
            features=self.action_expert_embed_dim, hidden_dim=self.action_expert_mlp_dim
        )
        if self.dropout:
            self.drop = nn.Dropout(self.dropout, self.dropout_bdims)
        else:
            self.drop = lambda x, _: x
        if self.post_norms:
            self.post_attention_norm = RMSNorm(name="gemma_layers_post_attn_norm")
            self.post_ffw_norm = RMSNorm(name="gemma_layers_post_ffw_norm")

    def _process_attention_output(
        self, attn_output: jax.Array, residual: jax.Array, deterministic: bool, mlp: FeedForward
    ) -> jax.Array:
        """Standard attention processing with residual connections, norms, and MLPs.

        Args:
            attn_output: [N, L, D] attention output
            residual: [N, L, D] residual connection
            deterministic: whether to use dropout

        Returns:
            [N, L, D] processed attention output
        """
        if self.post_norms:
            attn_output = self.post_attention_norm(attn_output)
        attn_output = self.drop(attn_output, deterministic)
        attn_output += residual
        attn_output = self.pre_ffw_norm(attn_output)
        attn_output = mlp(attn_output)
        attn_output = self.drop(attn_output, deterministic)
        if self.post_norms:
            attn_output = self.post_ffw_norm(attn_output)
        return attn_output + residual

    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        attn_mask: jax.Array,
        use_kv_cache: bool,
        deterministic: bool,
        separation_idx: int,
    ) -> jax.Array:
        """Transformer block forward pass.

        Args:
            x: [B, L, D] input embeddings
            positions: [B, L] absolute positions of the tokens
            attn_mask: [B, 1, L, S] attention mask
            use_kv_cache: whether to use kv-cache
            deterministic: whether to use dropout
            separation_idx: index to separate gemma and action expert
        Returns:
            [B, L, D] output embeddings
        """
        x_gemma = x[:, :separation_idx, :]  # [B, I + T, D]
        x_action = x[:, separation_idx:, :]  # [B, 1 + A, D]
        x_gemma = nn.with_logical_constraint(x_gemma, ("act_batch", "act_len", "act_emb"))  # type: ignore
        x_action = nn.with_logical_constraint(x_action, ("act_batch", "act_len", "act_emb"))  # type: ignore
        concat_idx = x_gemma.shape[1]

        # NOTE: Reusing RMSNorm for both gemma and action expertπ
        inputs_normalized_gemma = self.pre_attention_norm(x_gemma)  # [B, I + T, D]
        inputs_normalized_action = self.pre_attention_norm(x_action)  # [B, 1 + A, D]
        positions_gemma = positions[:, : x_gemma.shape[1]]
        positions_action = positions[:, x_gemma.shape[1] + 1 :]

        # TODO: thoroughly think through how to handle kv-cache
        q_gemma, k_gemma, v_gemma = self.gemma_attn.get_qkv(
            inputs_normalized_gemma,
            positions_gemma,
            cache_size=attn_mask.shape[-1],
            decode=use_kv_cache,
        )  # 3 x [B, I + T, D]

        q_action, k_action, v_action = self.action_expert_attn.get_qkv(
            inputs_normalized_action,
            positions_action,
            cache_size=attn_mask.shape[-1],
            decode=use_kv_cache,
        )  # 3 x [B, 1 + A, D]

        q = jnp.concatenate([q_gemma, q_action], axis=1)  # [B, L, D]
        k = jnp.concatenate([k_gemma, k_action], axis=1)  # [B, L, D]
        v = jnp.concatenate([v_gemma, v_action], axis=1)  # [B, L, D]

        attn_output = apply_attention(
            q,
            k,
            v,
            attn_mask,
            num_kv_heads=self.num_kv_heads,
            attn_logits_softcap=self.attn_logits_softcap,
        )

        gemma_attn_output = self.gemma_attn.proj_to_embed_dim(attn_output[:, :concat_idx, :])
        action_attn_output = self.action_expert_attn.proj_to_embed_dim(
            attn_output[:, concat_idx:, :]
        )

        gemma_attn_output = self._process_attention_output(
            gemma_attn_output, x_gemma, deterministic, self.gemma_mlp
        )
        action_attn_output = self._process_attention_output(
            action_attn_output, x_action, deterministic, self.action_expert_mlp
        )

        outputs = jnp.concatenate([gemma_attn_output, action_attn_output], axis=1)  # [N, L, D]
        return outputs


class PiZero(nn.Module):
    """Pi_Zero Policy Implementation"""

    max_images: int
    vit_variant: str
    llm_vocab_size: int

    gemma_mlp_dim: int
    gemma_embed_dim: int
    action_expert_mlp_dim: int
    action_expert_embed_dim: int
    depth: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    query_pre_attn_norm: str = "rsqrt_head_dim"
    final_logits_softcap: float = 0.0
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

        images = images[:, : self.max_images, :, :, :]
        vit_config = vit.decode_variant(self.vit_variant)
        image_token_embed, aux = vit.ViT(
            num_classes=self.gemma_embed_dim, name="img", **vit_config
        )(
            images
        )  # [B, I, D]
        del aux
        image_token_embed = image_token_embed.astype(self.embed_dtype)

        embedder = Embedder(
            vocab_size=self.llm_vocab_size, embed_dim=self.gemma_embed_dim, name="gemma_embedder"
        )
        text_token_embed = embedder.encode(text).astype(self.embed_dtype)  # [B, T, D]

        # run attention (need to pass in)
        L = I + T + P + A
        positions = jnp.arange(L).astype(jnp.int32)[None, :]

        # Applying block-wise causal mask as in the paper
        input_mask = jnp.ones([B, L])
        # NOTE: if images are missing, assume we populate them with all 0
        img_mask = images.any(axis=(-3, -2, -1))
        input_mask[:, : self.max_images] = input_mask[:, : self.max_images] * img_mask
        mask_ar = jnp.zeros([B, L])
        mask_ar[:, I + T] = 1
        mask_ar[:, I + T + P] = 1
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
            gemma_mlp_dim=self.gemma_mlp_dim,
            gemma_embed_dim=self.gemma_embed_dim,
            action_expert_mlp_dim=self.action_expert_mlp_dim,
            action_expert_embed_dim=self.action_expert_embed_dim,
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

        x = jnp.concatenate(
            [image_token_embed, text_token_embed, proprio_token_embed, action_token_embed], axis=1
        )
        for block in blocks:
            x = block(
                x=x,
                positions=positions,
                attn_mask=attn_mask,
                use_kv_cache=inference_mode,
                deterministic=deterministic,
                separation_idx=I + T,
            )

        assert x.dtype == jnp.dtype(self.embed_dtype)  # Sanity check.
        out["encoded"] = x

        x = RMSNorm(name="final_norm")(x)
        out["pre_logits"] = x

        x = embedder.decode(x)
        out["logits_pre_norm"] = x
        if self.final_logits_softcap:
            x = jnp.tanh(x / self.final_logits_softcap) * self.final_logits_softcap
        out["logits"] = x

        return x, out
