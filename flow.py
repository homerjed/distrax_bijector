from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from flow.bijector import ConditionalBijector
from flow.maskedcoupling import ConditionalMaskedCoupling
from flow.chain import ConditionalChain
from flow.transformed import ConditionalTransformed


flags.DEFINE_integer("flow_num_layers", 16, "Number of layers to use in the flow.") # 8 
flags.DEFINE_integer("mlp_num_layers", 2, "Number of layers to use in the MLP conditioner.")
flags.DEFINE_integer("hidden_size", 32, "Hidden size of the MLP conditioner.")
flags.DEFINE_integer("num_bins", 4, "Number of bins to use in the rational-quadratic spline.")
flags.DEFINE_integer("batch_size", 256, "Batch size for training and evaluation.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 100000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 1000, "How often to evaluate the model.")
flags.DEFINE_bool("contextualise", True, "Context input to flow.")
FLAGS = flags.FLAGS

from chex import Array, PRNGKey
Batch = Tuple[Array, Array]
OptState = Any

EVENT_SHAPE = (2,)
CONTEXT_SHAPE = (2,)


def make_conditioner(
    event_shape: Sequence[int], # event or context 
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
) -> hk.Sequential:
    """Creates an MLP conditioner for each layer of the flow."""
    return hk.Sequential(
        [
			hk.Flatten(preserve_dims=-len(event_shape)),
            hk.nets.MLP(hidden_sizes, activate_final=True),
            # We initialize this linear layer to zero so that the flow is initialized
            # to the identity function.
            hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros,
            ),
            hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
        ]
    )


def make_flow_model(
    event_shape: Sequence[int],
    context_shape: Sequence[int],
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
) -> distrax.Transformed:
    """Creates the flow model."""
    # Alternating binary mask.
    mask = jnp.arange(0, np.prod(event_shape)) % 2
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)

    def bijector_fn(params: Array):
        return distrax.RationalQuadraticSpline(
				params, range_min=0.0, range_max=1.0)

    # Number of parameters for the rational-quadratic spline:
    # - `num_bins` bin widths
    # - `num_bins` bin heights
    # - `num_bins + 1` knot slopes
    # for a total of `3 * num_bins + 1` parameters.
    num_bijector_params = 3 * num_bins + 1

    layers = []
    for _ in range(num_layers):
        layer = ConditionalMaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
			# conditioner
            conditioner=make_conditioner(
                event_shape, hidden_sizes, num_bijector_params
            ),
			# contextualiser
			conditioner_eta=make_conditioner(
				context_shape, hidden_sizes, num_bijector_params,
			),
        )
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    # We invert the flow so that the `forward` method is called with `log_prob`.
	#flow = distrax.Inverse(ConditionalChain(layers))
	#flow = ConditionalChain(layers[::-1]) # manual reversal, distrax.Inverse causes issues...
    flow = ConditionalChain(layers) # manual reversal, distrax.Inverse causes issues...
    '''
    base_distribution = distrax.Independent(
        distrax.Normal(
			loc=jnp.zeros(event_shape), scale=jnp.ones(event_shape)
		),
        reinterpreted_batch_ndims=len(event_shape),
    )
    '''
    base_distribution = distrax.MultivariateNormalDiag(
			loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))
    return ConditionalTransformed(base_distribution, flow)


def get_batch(batch_size: int, key: PRNGKey) -> Array:
    key0, key1 = jax.random.split(key)
    rosenquist = True 
    if rosenquist:
        # rosenquist distribution
        x2 = jax.random.normal(key0, shape=(batch_size,)) * 2.
        x1 = jax.random.normal(key1, shape=(batch_size,)) + (x2 ** 2. / 4.)
        batch = jnp.stack([x1, x2], axis=-1)
        labels = jnp.ones((batch_size, *CONTEXT_SHAPE))
		# get noise with y values less than 0.0
        label_idx = batch[:,1] < 0.0
		# set the labels for this noise to zeros
        labels = labels.at[label_idx].set(0.0)
		# shuffling indices
    else:
        # two gaussians 
        x0 = jax.random.multivariate_normal(
			key0, mean=jnp.zeros(EVENT_SHAPE), cov=0.5 * jnp.eye(*EVENT_SHAPE),
			shape=(batch_size // 2,))
        x1 = jax.random.multivariate_normal(
			key1, mean=jnp.ones(EVENT_SHAPE), cov=1.0 * jnp.eye(*EVENT_SHAPE),
			shape=(batch_size // 2,))
        batch = jnp.concatenate([x0, x1])
        labels = jnp.concatenate(
			[jnp.zeros((batch_size // 2, 2)), jnp.ones((batch_size // 2, 2))])
    if FLAGS.contextualise:
        idx = jax.random.randint(key, minval=0, maxval=FLAGS.batch_size, shape=(FLAGS.batch_size,))
        batch, labels = batch[idx], labels[idx]
        return batch, labels
    else:
        return batch


def prepare_data(
		batch: Array, 
		context: Array,
		prng_key: Optional[PRNGKey] = None
) -> Array:
    # your data preparation here...
    return batch, context


@hk.without_apply_rng
@hk.transform
def log_prob(data: Array, context: Array) -> Array:
    model = make_flow_model(
        event_shape=EVENT_SHAPE,
		context_shape=CONTEXT_SHAPE,
        num_layers=FLAGS.flow_num_layers,
        hidden_sizes=[FLAGS.hidden_size] * FLAGS.mlp_num_layers,
        num_bins=FLAGS.num_bins,
    )
    return model.log_prob(data, context=context)


def loss_fn(
		params: hk.Params, 
		prng_key: PRNGKey, 
		batch: Array,
		context: Array
) -> Array:
    data, context = prepare_data(batch, context, None)
    # Loss is average negative log likelihood.
    loss = -jnp.mean(log_prob.apply(params, data, context=context))
    return loss


@jax.jit
def eval_fn(
		params: hk.Params, 
	    batch: Array,
	    context: Array
) -> Array:
    data, context = prepare_data(batch, context, None)  # We don't dequantize during evaluation.
    loss = -jnp.mean(log_prob.apply(params, data, context=context))
    return loss


@hk.without_apply_rng
@hk.transform
def sample_and_log_prob(
		params: hk.Params,
		key: PRNGKey,
		context: Array
) -> Tuple[Array, Array]:
    model = make_flow_model(
        event_shape=EVENT_SHAPE,
		context_shape=CONTEXT_SHAPE,
        num_layers=FLAGS.flow_num_layers,
        hidden_sizes=[FLAGS.hidden_size] * FLAGS.mlp_num_layers,
        num_bins=FLAGS.num_bins,
    )
    n_sample = context.shape[0]
	#samples = model.sample_and_log_prob(key, (n_sample,), context=context)
    key = jax.random.PRNGKey(0)
    key, _ = jax.random.split(key)
    samples = model._sample_n(key, n_sample, context)
    return samples


def sample(
		params: hk.Params,
		key: PRNGKey,
		context: Array
) -> Tuple[Array, Array]:
	n_sample = context.shape[0]
	#return sample_and_log_prob.apply(params, key, (n_sample,), context=context)
	#return sample_and_log_prob.apply(params, key=key, context=context)
	return sample_and_log_prob.apply(params, key, (n_sample,), context)
    #return sample_and_log_prob.apply(params, key,  context)


def training_sample_plot(key, n_samples=1000):
    samples, labels = get_batch(n_samples, key)
    labels = labels[:,0]
    plt.figure(figsize=(3.,3.), dpi=100)
    plt.scatter(*samples.T, c=labels)
    plt.axis("off")
    plt.savefig("samples/training.png")
    plt.close()

	
def main(_):

    @jax.jit
    def update(
        params: hk.Params, 
		prng_key: PRNGKey, 
		opt_state: OptState, 
		batch: Array,
		context: Array
    ) -> Tuple[hk.Params, OptState]:
        """Single SGD update step."""
        grads = jax.grad(loss_fn)(params, prng_key, batch=batch, context=context)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state


    optimizer = optax.adam(FLAGS.learning_rate)

    prng_seq = hk.PRNGSequence(42)
	# training data plot 
    training_sample_plot(next(prng_seq))
	# initalise params with fixed data 
    data, context = get_batch(FLAGS.batch_size, next(prng_seq))
    params = log_prob.init(next(prng_seq), data, context=context)
    opt_state = optimizer.init(params)

    key = next(prng_seq)
	#train_ds = load_dataset(tfds.Split.TRAIN, FLAGS.batch_size)
	#valid_ds = load_dataset(tfds.Split.TEST, FLAGS.batch_size)
    print("initialised flow.\n")

    losses = []
    for step in range(FLAGS.training_steps):
        key, _ = jax.random.split(key)
        train_data, context = get_batch(FLAGS.batch_size, key)
        params, opt_state = update(
				params, 
				next(prng_seq), 
				opt_state, 
				batch=train_data, 

				context=context
		)

        if step % FLAGS.eval_frequency == 0:
            val_data, context = get_batch(FLAGS.batch_size, next(prng_seq))
            val_loss = eval_fn(params, val_data, context=context)

            logging.info("STEP: %5d; Validation loss: %.3f", step, val_loss)
            losses.append(val_loss)

	        # sample some points 
            n_sample = 1000
            context = jnp.concatenate(
                     [jnp.zeros((n_sample, 2)), jnp.ones((n_sample, 2))])
            samples = sample(params, next(prng_seq), context)
            plt.figure(figsize=(3.,3.), dpi=100)
            plt.scatter(*samples.T, c=context[:,0])
            plt.axis("off")
            plt.savefig(f"samples/sample{step}.png")
            plt.close()

    plt.figure()
    plt.plot(range(0, FLAGS.training_steps, FLAGS.eval_frequency), losses)
    plt.savefig("loss.png")

if __name__ == "__main__":
    app.run(main)
