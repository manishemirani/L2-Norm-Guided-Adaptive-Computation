"""Universal Vision Transformer with L2 Adaptive Computation"""

from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit


class UTStochasticDepth(nn.Module):
    """Performs layer-dropout (also known as stochastic depth).
    Described in
    Huang & Sun et al, "Deep Networks with Stochastic Depth", 2016
    https://arxiv.org/abs/1603.09382
    Attributes:
      rate: the layer dropout probability (_not_ the keep rate!).
      deterministic: If false (e.g. in training) the inputs are scaled by `1 / (1
        - rate)` and the layer dropout is applied, whereas if true (e.g. in
        evaluation), no stochastic depth is applied and the inputs are returned as
        is.
    Note: This is a repeated implementation of model_lib.nn_layers.StochasticDepth
      The implementation here is to match the nn.cond in UT
    """
    rate: float = 0.0
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 deterministic: Optional[bool] = None) -> jnp.ndarray:
        """Applies a stochastic depth mask to the inputs.
        Args:
          x: Input tensor.
          deterministic: If false (e.g. in training) the inputs are scaled by `1 /
            (1 - rate)` and the layer dropout is applied, whereas if true (e.g. in
            evaluation), no stochastic depth is applied and the inputs are returned
            as is.
        Returns:
          The masked inputs reweighted to preserve mean.
        """
        if self.rate <= 0.0:
            return x
        if deterministic:
            return x
        else:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            rng = self.make_rng('dropout')
            mask = jax.random.bernoulli(rng, self.rate, shape)
            return x * (1.0 - mask)


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.
    Attributes:
      mlp_dim: Dimension of the mlp on top of attention block.
      num_heads: Number of self-attention heads.
      dtype: The dtype of the computation (default: float32).
      dropout_rate: Dropout rate.
      attention_dropout_rate: Dropout for attention heads.
      stochastic_depth: probability of dropping a layer linearly grows from 0 to
        the provided value.
    Returns:
      output after transformer encoder block.
    """
    mlp_dim: int
    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    stochastic_depth: float = 0.0
    deterministic: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Applies Encoder1DBlock module.
        Args:
          inputs: Input data.
        Returns:
          Output after transformer encoder block.
        """
        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=self.deterministic,
            dropout_rate=self.attention_dropout_rate)(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
        x = UTStochasticDepth(rate=self.stochastic_depth)(x, self.deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = attention_layers.MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            activation_fn=nn.gelu,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=self.deterministic)
        y = UTStochasticDepth(rate=self.stochastic_depth)(y, self.deterministic)
        return y + x


class LacStep(nn.Module):
    """Takes a LAC step."""
    lac_config: ml_collections.ConfigDict
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, carry_inputs: Any, layer: nn.Module) -> Any:
        (current_state, previous_norm, previous_delta_norm, delta_norm, delta_norms,
         step, alpha, previous_state, threshold) = carry_inputs

        state_norm = jnp.linalg.norm(current_state, axis=self.lac_config.axis).astype(self.dtype)

        delta_norms = delta_norms.at[step].set(previous_delta_norm)

        max_delta = jnp.max(delta_norms)
        min_delta = jnp.min(delta_norms)
        new_threshold = jnp.abs(max_delta - min_delta) * alpha

        # compute the distance between current norm and previous norm
        delta_norm = state_norm - previous_norm

        # Set the current norm to previous norm for next step
        previous_norm = state_norm

        # Set the current delta norm to previous delta norm for next step
        previous_delta_norm = delta_norm

        if self.lac_config.mode != 'B' and self.lac_config.use_mask:
            reshape_value = delta_norm.shape + (current_state.ndim - len(delta_norm.shape)) * (1,)
            new_halted = jnp.less(delta_norm, new_threshold).astype(current_state.dtype).reshape(reshape_value)

            still_running = jnp.greater_equal(previous_delta_norm, threshold).astype(current_state.dtype).reshape(
                reshape_value)

            output_state = layer(current_state)

            new_state = output_state * new_halted + previous_state * still_running
            return (output_state, previous_norm, previous_delta_norm, delta_norm, delta_norms,
                    step + 1, alpha, new_state, new_threshold)

        else:
            output_state = layer(current_state)
            return (output_state, previous_norm, previous_delta_norm, delta_norm, delta_norms,
                    step + 1, alpha, output_state, new_threshold)


class LACFunction(nn.Module):
    """L2 Adaptive Computation Function"""
    lac_config: ml_collections.ConfigDict
    stop_fn: Any
    dtype: Any = jnp.float32

    def setup(self):
        self.lac_step = LacStep(lac_config=self.lac_config, dtype=self.dtype)

    def take_a_step(self, x, layer) -> Any:
        return self.lac_step(x, layer)

    def skip_a_step(self, x, _) -> Any:  # Shunt
        return x

    @nn.compact
    def __call__(self, x, layer: nn.Module, _) -> Any:
        if self.is_mutable_collection('params'):  # Init-mode
            out = self.take_a_step(x, layer)
        else:
            decision = self.stop_fn(x)
            out = nn.cond(decision, self.skip_a_step, self.take_a_step, self, x, layer)
        return out, None


class UTEncoder(nn.Module):
    """Universal Transformer Encoder with LAC.
    Attributes:
      num_layers: Number of layers.
      mlp_dim: Dimension of the mlp on top of attention block.
      inputs_positions: Input subsequence positions for packed examples.
      dropout_rate: Dropout rate.
      stochastic_depth: probability of dropping a layer linearly grows from 0 to
        the provided value. Our implementation of stochastic depth follows timm
        library, which does per-example layer dropping and uses independent
        dropping patterns for each skip-connection.
      dtype: Dtype of activations.
    """
    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    stochastic_depth: float = 0.0
    parameter_sharing: bool = True
    lac_config: Optional[ml_collections.ConfigDict] = None
    dtype: Any = jnp.float32

    def stop_fn(self, inputs: Any):
        _, _, _, delta_norm, _, _, _, _, threshold = inputs
        return jnp.all(jnp.less(jnp.abs(delta_norm), threshold))

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, alpha: float = 1.0, *, train: bool = False):
        """Applies Transformer model on the inputs."""
        assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
        x = vit.AddPositionEmbs(
            posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
            name='posembed_input')(
            inputs)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        dtype = jax.dtypes.canonicalize_dtype(self.dtype)

        if self.lac_config.vit:
            for i in range(self.num_layers):
                x = Encoder1DBlock(
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    stochastic_depth=self.stochastic_depth,
                    deterministic=not train,
                    name='encoderblock_' + str(i),
                    dtype=dtype)(x)
            encoded = nn.LayerNorm(name='encoder_norm')(x)
            return encoded, self.num_layers

        update_shape = x.shape[self.lac_config.state_slice]
        previous_delta_norm = jnp.zeros(shape=update_shape, dtype=dtype)
        delta_norm = jnp.zeros(shape=update_shape, dtype=dtype)
        previous_norm = jnp.linalg.norm(x, axis=self.lac_config.axis).astype(dtype)
        delta_norms = jnp.zeros(shape=(self.num_layers,) + update_shape, dtype=dtype)
        previous_state = jnp.zeros_like(x, dtype=dtype)
        threshold = 0.0
        step = 0
        intermedia_output = (x, previous_norm, previous_delta_norm, delta_norm, delta_norms, step,
                             alpha, previous_state, threshold)
        
        if not self.parameter_sharing:
            lac_fn = LACFunction(self.lac_config, self.stop_fn, dtype)
            for i in range(self.num_layers):
                intermedia_output, _ = lac_fn(intermedia_output, Encoder1DBlock(
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    stochastic_depth=self.stochastic_depth,
                    deterministic=not train,
                    name=f'encoderblock_{i}',
                    dtype=dtype), None)
        else:
            encoder_block = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                stochastic_depth=self.stochastic_depth,
                deterministic=not train,
                name=f'encoderblock',
                dtype=self.dtype)
            lac_fn = LACFunction(self.lac_config, self.stop_fn)
            for i in range(self.num_layers):
                intermedia_output, _ = lac_fn(intermedia_output, encoder_block, None)

        _, _, _, _, _, step, _, x, _ = intermedia_output
        encoded = nn.LayerNorm(name='encoder_norm')(x)
        return encoded, step


class UViT(nn.Module):
    """Universall Vision Transformer model.
      Attributes:
      num_classes: Number of output classes.
      mlp_dim: Dimension of the mlp on top of attention block.
      num_layers: Number of layers.
      num_heads: Number of self-attention heads.
      patches: Configuration of the patches extracted in the stem of the model.
      ac_config: Configuration of the adaptive computation.
      hidden_size: Size of the hidden state of the output of model's stem.
      dropout_rate: Dropout rate.
      attention_dropout_rate: Dropout for attention heads.
      classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
        'token'.
      dtype: JAX data type for activations.
    """

    num_classes: int
    mlp_dim: int
    num_layers: int
    num_heads: int
    patches: ml_collections.ConfigDict
    lac_config: ml_collections.ConfigDict
    hidden_size: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    stochastic_depth: float = 0.0
    classifier: str = 'gap'
    parameter_sharing: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, alpha: float = 1.0, *, train: bool, debug: bool = False):

        fh, fw = self.patches.size
        # Extracting patches and then embedding is in fact a single convolution.
        x = nn.Conv(
            self.hidden_size, (fh, fw),
            strides=(fh, fw),
            padding='VALID',
            name='embedding')(
            x)
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        # If we want to add a class token, add it here.
        if self.classifier == 'token':
            cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
            cls = jnp.tile(cls, [n, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)

        x, auxiliary_outputs = UTEncoder(
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            lac_config=self.lac_config,
            stochastic_depth=self.stochastic_depth,
            parameter_sharing=self.parameter_sharing,
            dtype=self.dtype,
            name='UTransformer')(
            x, train=train, alpha=alpha)

        if self.classifier in ('token', '0'):
            x = x[:, 0]
        elif self.classifier in ('gap', 'gmp', 'gsp'):
            fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
            x = fn(x, axis=1)

        x = nn_layers.IdentityLayer(name='pre_logits')(x)
        x = nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.zeros,
            name='output_projection')(
            x)
        return x, auxiliary_outputs


class UViTMultiLabelClassificationModel(MultiLabelClassificationModel):
    """Universal Vision Transformer model for multi-label classification task."""

    def build_flax_model(self) -> nn.Module:
        model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
        return UViT(
            num_classes=self.dataset_meta_data['num_classes'],
            mlp_dim=self.config.model.mlp_dim,
            num_layers=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            patches=self.config.model.patches,
            lac_config=self.config.model.get('lac_config'),
            hidden_size=self.config.model.hidden_size,
            classifier=self.config.model.classifier,
            dropout_rate=self.config.model.get('dropout_rate', 0.1),
            attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                         0.1),
            stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
            parameter_sharing=self.config.model.get('parameter_sharing', True),
            dtype=model_dtype,
        )

    def init_from_train_state(
            self, train_state: Any, restored_train_state: Any,
            restored_model_cfg: ml_collections.ConfigDict) -> Any:
        """Updates the train_state with data from restored_train_state.
        This function is writen to be used for 'fine-tuning' experiments. Here, we
        do some surgery to support larger resolutions (longer sequence length) in
        the transformer block, with respect to the learned pos-embeddings.
        Args:
          train_state: A raw TrainState for the model.
          restored_train_state: A TrainState that is loaded with parameters/state of
            a  pretrained model.
          restored_model_cfg: Configuration of the model from which the
            restored_train_state come from. Usually used for some asserts.
        Returns:
          Updated train_state.
        """
        raise NotImplementedError