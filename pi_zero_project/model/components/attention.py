"""Base implementation of attention"""

from typing import Optional, Tuple
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from pi_zero_project.model.components.linear import Einsum, MlpBlock
from pi_zero_project.model.components.pos_embed import apply_rope


def update_kv_cache(
    module: nn.Module,
    k: jax.Array,
    v: jax.Array,
    cache_size: int,
    cache_dtype: jnp.dtype | None,
) -> Tuple[jax.Array, jax.Array]:
    """Updates KV cache and returns its current contents.

    Args:
        module: flax module
        k: key array
        v: value array
        cache_size: size of the cache
        cache_dtype: dtype of the cache

    Returns:
        Tuple[jax.Array, jax.Array]: current contents of the cache
    """
    initialized = module.has_variable("cache", "idx")
    batch_size, update_len, num_heads, head_dim = k.shape
    cache_dtype = cache_dtype or k.dtype

    # Idx of which cache row to update next is the same for all examples, so that
    # it allows to update with dynamic_update_slice. But in order to keep things
    # nicely partitioned we store it with leading batch dimension and use only
    # the first entry.
    idx = module.variable("cache", "idx", jnp.zeros, (batch_size,), jnp.int32)

    kv_shape = (batch_size, cache_size, num_heads, head_dim)
    k_cache = module.variable("cache", "k_cache", jnp.zeros, kv_shape, cache_dtype)
    v_cache = module.variable("cache", "v_cache", jnp.zeros, kv_shape, cache_dtype)

    if initialized:  # write k, v in the next cache position.
        assert update_len == 1, update_len
        # Note: idx is the same for all examples. Use value from example 0.
        indices = (0, idx.value[0], 0, 0)
        k_cache.value = jax.lax.dynamic_update_slice(k_cache.value, k.astype(cache_dtype), indices)
        v_cache.value = jax.lax.dynamic_update_slice(v_cache.value, v.astype(cache_dtype), indices)
        idx.value = idx.value + 1
    else:  # init cache with k, v after padding to cache_size.
        prefill_len = k.shape[1]
        pad_width = ((0, 0), (0, cache_size - prefill_len), (0, 0), (0, 0))
        k_cache.value = jnp.pad(k.astype(cache_dtype), pad_width)
        v_cache.value = jnp.pad(v.astype(cache_dtype), pad_width)
        idx.value = idx.value + prefill_len

    return k_cache.value.astype(k.dtype), v_cache.value.astype(v.dtype)


class Attention(nn.Module):
    """Gemma's Attention module.

    Unlike nn.MultiHeadAttention, this applies RoPE to the query and key, and
    uses a different query pre-attention normalization.

    Attributes:
        num_heads: number of attention heads
        num_kv_heads: number of key/value heads
        features: embedding dimension
        head_dim: dimension of each attention head
        query_pre_attn_norm: query pre-attention normalization
        attn_logits_softcap: attention logits softcap
        cache_dtype: cache dtype
    """

    num_heads: int
    num_kv_heads: int
    features: int
    head_dim: int

    query_pre_attn_norm: str
    attn_logits_softcap: float | None

    cache_dtype: jnp.dtype | None = None

    def setup(self) -> None:
        if self.num_kv_heads == self.num_heads:
            self.qkv_einsum = Einsum(
                shape=(3, self.num_heads, self.features, self.head_dim),
                w_init=nn.initializers.variance_scaling(
                    1.0,
                    "fan_in",
                    "truncated_normal",
                    in_axis=(2,),
                    out_axis=(0, 1, 3),
                    batch_axis=(),
                ),
            )
        else:
            # MQA / GQA
            self.q_einsum = Einsum(
                shape=(self.num_heads, self.features, self.head_dim),
                w_init=nn.initializers.variance_scaling(
                    1.0, "fan_in", "truncated_normal", in_axis=(1,), out_axis=(0, 2), batch_axis=()
                ),
            )
            self.kv_einsum = Einsum(
                shape=(2, self.num_kv_heads, self.features, self.head_dim),
                w_init=nn.initializers.variance_scaling(
                    1.0,
                    "fan_in",
                    "truncated_normal",
                    in_axis=(2,),
                    out_axis=(0, 1, 3),
                    batch_axis=(),
                ),
            )
        self.attn_vec_einsum = Einsum(
            shape=(self.num_heads, self.head_dim, self.features),
            w_init=nn.initializers.variance_scaling(
                1.0, "fan_in", "truncated_normal", in_axis=(0, 1), out_axis=(2,), batch_axis=()
            ),
        )

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        attn_mask: jax.Array,
        decode: bool,
    ) -> jax.Array:
        """Attention module forward pass.

        Args:
            x: [N, L, D] input tokens
            positions: [N, L] absolute positions of the tokens
            attn_mask: [N, 1, L, S] attention mask
            decode: whether to use kv-cache

        Returns:
            [N, L, D] output embeddings
        """
        if self.num_kv_heads == self.num_heads:
            q, k, v = self.qkv_einsum("BSD,3KDH->3BSKH", x)
        else:
            q = self.q_einsum("BTD,NDH->BTNH", x)
            k, v = self.kv_einsum("BSD,2KDH->2BSKH", x)

        q = apply_rope(q, positions=positions)
        if self.query_pre_attn_norm == "rsqrt_head_dim":
            q *= self.head_dim**-0.5
        elif self.query_pre_attn_norm == "rsqrt_emb_per_head":
            q *= (self.features // self.num_heads) ** -0.5
        else:
            raise ValueError(f"Unknown query_pre_attn_norm: {self.query_pre_attn_norm}")

        k = apply_rope(k, positions=positions)
        if decode:
            k, v = update_kv_cache(
                self, k, v, cache_size=attn_mask.shape[-1], cache_dtype=self.cache_dtype
            )

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k)
        logits = logits.astype(jnp.float32)

        if self.attn_logits_softcap:
            logits = jnp.tanh(logits / self.attn_logits_softcap)
            logits = logits * self.attn_logits_softcap

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k "
                f"are: {q.shape} and {k.shape}"
            )

        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(k.dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
        attn_output = self.attn_vec_einsum("BTNH,NHD->BTD", encoded)

        return attn_output


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Multihead Attention Pooling

        Args:
            x: [B, L, D] input tokens

        Returns:
            [B, D] output tokens
        """
        n, l, d = x.shape  # pylint: disable=unused-variable
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])  # [N, 1, D]

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform()
        )(
            probe, x
        )  # [N, 1, D]

        y = nn.LayerNorm()(x)  # [N, 1, D]
        x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)  # [N, 1, D]
        return x[:, 0]  # [N, D]
