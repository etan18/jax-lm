from __future__ import annotations

import os
from collections.abc import Callable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
from flax import nnx
from flax.nnx import State
from jax import Array
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Float, Int

from data import tokenizer, train_bpe


def perpetual_rngs(seed: int):
    rngs = nnx.Rngs(seed)

    def create_wrapper(function: Callable[..., Any]):
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs, rngs=rngs)

        return wrapper

    return create_wrapper


@perpetual_rngs(0)
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Array, " d_out d_in"],
    in_features: Float[Array, " ... d_in"],
    rngs: nnx.Rngs,
) -> Float[Array, " ... d_out"]:
    raise NotImplementedError("not implemented!")


@perpetual_rngs(1)
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Array, " vocab_size d_model"],
    token_ids: Int[Array, " ..."],
    rngs: nnx.Rngs,
) -> Float[Array, " ... d_model"]:
    raise NotImplementedError("not implemented!")


@perpetual_rngs(2)
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Array, " d_ff d_model"],
    w2_weight: Float[Array, " d_model d_ff"],
    w3_weight: Float[Array, " d_ff d_model"],
    in_features: Float[Array, " ... d_model"],
    rngs: nnx.Rngs,
) -> Float[Array, " ... d_model"]:
    raise NotImplementedError("not implemented!")


def run_scaled_dot_product_attention(
    Q: Float[Array, " ... queries d_k"],
    K: Float[Array, " ... keys d_k"],
    V: Float[Array, " ... values d_v"],
    mask: Float[Array, " ... queries keys"] | None = None,
) -> Float[Array, " ... queries d_v"]:
    raise NotImplementedError("not implemented!")


@perpetual_rngs(3)
def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Array, " d_k d_in"],
    k_proj_weight: Float[Array, " d_k d_in"],
    v_proj_weight: Float[Array, " d_v d_in"],
    o_proj_weight: Float[Array, " d_model d_v"],
    in_features: Float[Array, " ... sequence_length d_in"],
    rngs: nnx.Rngs,
) -> Float[Array, " ... sequence_length d_out"]:
    raise NotImplementedError("not implemented!")


@perpetual_rngs(4)
def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Array, " d_k d_in"],
    k_proj_weight: Float[Array, " d_k d_in"],
    v_proj_weight: Float[Array, " d_v d_in"],
    o_proj_weight: Float[Array, " d_model d_v"],
    in_features: Float[Array, " ... sequence_length d_in"],
    rngs: nnx.Rngs,
    token_positions: Int[Array, " ... sequence_length"] | None = None,
) -> Float[Array, " ... sequence_length d_out"]:
    raise NotImplementedError("not implemented!")


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Array, " ... sequence_length d_k"],
    token_positions: Int[Array, " ... sequence_length"],
) -> Float[Array, " ... sequence_length d_k"]:
    raise NotImplementedError("not implemented!")


@perpetual_rngs(5)
def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Array],
    in_features: Float[Array, " batch sequence_length d_model"],
    rngs: nnx.Rngs,
) -> Float[Array, " batch sequence_length d_model"]:
    raise NotImplementedError("not implemented!")


@perpetual_rngs(6)
def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Array],
    in_indices: Int[Array, " batch_size sequence_length"],
    rngs: nnx.Rngs,
) -> Float[Array, " batch_size sequence_length vocab_size"]:
    raise NotImplementedError("not implemented!")


@perpetual_rngs(7)
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Array, " d_model"],
    in_features: Float[Array, " ... d_model"],
    rngs: nnx.Rngs,
) -> Float[Array, " ... d_model"]:
    raise NotImplementedError("not implemented!")


def run_silu(in_features: Float[Array, " ..."]) -> Float[Array, " ..."]:
    raise NotImplementedError("not implemented!")


@perpetual_rngs(8)
def run_get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    rngs: nnx.Rngs,
) -> tuple[Array, Array]:
    raise NotImplementedError("not implemented!")


def run_softmax(in_features: Float[Array, " ..."], dim: int) -> Float[Array, " ..."]:
    raise NotImplementedError("not implemented!")


def run_cross_entropy(
    inputs: Float[Array, " batch_size vocab_size"],
    targets: Int[Array, " batch_size"],
) -> Float[Array, ""]:
    raise NotImplementedError("not implemented!")


def run_gradient_clipping(gradient_state: State, max_l2_norm: float) -> State:
    raise NotImplementedError("not implemented!")


def get_adamw_cls():
    raise NotImplementedError("not implemented!")


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    raise NotImplementedError("not implemented!")


def run_save_checkpoint(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    _stub("run_save_checkpoint")
    raise NotImplementedError("not implemented!")


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nnx.Module,
    optimizer: nnx.Optimizer,
):
    _stub("run_load_checkpoint")
    raise NotImplementedError("not implemented!")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return tokenizer.Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return train_bpe.train_bpe(input_path, vocab_size, special_tokens)


def get_mesh(mesh_shape: list[int], mesh_axis_names: list[str]) -> Mesh:
    raise NotImplementedError("not implemented!")


def get_sharded_batch(
    dataset: Any,
    batch_size: int,
    context_length: int,
    mesh: Mesh,
    batch_partition_spec: P | None = None,
):
    raise NotImplementedError("not implemented!")


def get_sharded_model_and_optimizer(
    model_config: dict,
    optimizer_config: dict,
    mesh: Mesh,
    sharding_mode: str,
):
    raise NotImplementedError("not implemented!")


def _normalize_spec(spec: P) -> tuple:
    raise NotImplementedError("not implemented!")


def assert_array_sharded(array: Array, expected_spec: P, mesh: Mesh, name: str = ""):
    raise NotImplementedError("not implemented!")


def get_model_param_arrays(model_instance, num_layers: int) -> dict[str, Array]:
    raise NotImplementedError("not implemented!")


def get_expected_model_sharding_specs(num_layers: int, sharding_mode: str) -> dict[str, P]:
    raise NotImplementedError("not implemented!")


def get_expected_batch_sharding_spec(sharding_mode: str) -> P:
    raise NotImplementedError("not implemented!")


def get_optimizer_state_arrays(optimizer_instance) -> list[Array]:
    raise NotImplementedError("not implemented!")
