"""
	repo dir: 
		/Users/Jed.Homer/phd/study/jax/distrax/mnistflow
"""
from absl import app
from absl import flags
from absl import logging

from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple
from chex import Array, PRNGKey

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn import datasets

from flow.bijector import ConditionalBijector
from flow.maskedcoupling import ConditionalMaskedCoupling
from flow.chain import ConditionalChain
from flow.transformed import ConditionalTransformed
from flow.inverse import ConditionalInverse


flags.DEFINE_integer("flow_num_layers", 6, "Number of layers to use in the flow.") # 8 
flags.DEFINE_integer("mlp_num_layers", 4, "Number of layers to use in the MLP conditioner.")
flags.DEFINE_integer("hidden_size", 32, "Hidden size of the MLP conditioner.") # 256
flags.DEFINE_integer("num_bins", 32, "Number of bins to use in the rational-quadratic spline.") #8 
flags.DEFINE_integer("batch_size", 64, "Batch size for training and evaluation.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 8000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 10, "How often to evaluate the model.")
flags.DEFINE_bool("contextualise", True, "Context input to flow.")
FLAGS = flags.FLAGS

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
			#hk.Flatten(preserve_dims=-len(event_shape)),
            hk.nets.MLP(
				hidden_sizes, 
				activate_final=True,
				activation=jax.nn.leaky_relu
			),
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

def make_contextualiser(
    context_shape: Sequence[int], 
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
) -> hk.Sequential:
    """Creates an MLP conditioner for each layer of the flow."""
    return hk.Sequential(
        [
			#hk.Flatten(preserve_dims=-len(event_shape)),
            hk.nets.MLP(
				hidden_sizes, 
				activate_final=True,
				activation=jax.nn.leaky_relu
			),
            # We initialize this linear layer to zero so that the flow is initialized
            # to the identity function.
            hk.Linear(2, w_init=jnp.zeros, b_init=jnp.zeros),
			hk.Reshape((2,), preserve_dims=-1)
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
				#params, range_min=-5.0, range_max=10.0) # ROSENQUIST
				params, range_min=-5.0, range_max=5.0) # MOONS


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
			conditioner_eta=make_contextualiser(
				context_shape, 
				[h // 4 for h in hidden_sizes],#hidden_sizes, 
				num_bijector_params,
			),
        )
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    # We invert the flow so that the `forward` method is called with `log_prob`.
	#flow = distrax.Inverse(ConditionalChain(layers))
	#flow = ConditionalChain(layers[::-1]) # manual reversal, distrax.Inverse causes issues...
    flow = ConditionalInverse(ConditionalChain(layers))
    base_distribution = distrax.MultivariateNormalDiag(
			loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))
    return ConditionalTransformed(base_distribution, flow)


def get_batch(batch_size: int, key: PRNGKey) -> Array:
    batch, labels = datasets.make_moons(n_samples=batch_size, noise=.05)
    batch = (batch - batch.mean()) / batch.std()
    batch = jnp.asarray(batch).astype(jnp.float32)
    labels = jnp.asarray(labels).astype(jnp.float32)
    labels = jnp.stack([labels, labels], axis=-1).astype(jnp.float32)
    return batch, labels
'''
    if FLAGS.contextualise:
		# randomly shuffle batch/labels together
        idx = jax.random.randint(key, minval=0, maxval=batch.shape[0], shape=(batch.shape[0],))
        batch = (batch - batch.mean()) / batch.std()
		#batch = 2. * (batch - batch.min()) / (batch.max() - batch.min()) - 1.0
        return batch[idx], labels[idx]
    else:
        return batch
'''


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
	#data, context = prepare_data(batch, context, None)
    data, context = batch, context
    # Loss is average negative log likelihood.
    loss = -jnp.mean(log_prob.apply(params, data, context=context))
    return loss


@jax.jit
def eval_fn(
		params: hk.Params, 
	    batch: Array,
	    context: Array
) -> Array:
	#data, context = prepare_data(batch, context, None)  # We don't dequantize during evaluation.
    data, context = batch, context
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
    samples = model._sample_n(key, n=n_sample, context=context)
    return samples


def sample(
		params: hk.Params,
		key: PRNGKey,
		context: Array
) -> Array: #Tuple[Array, Array]: # no tuple at the moment just samples...
	n_sample = context.shape[0]
	return sample_and_log_prob.apply(params, (n_sample,), key=key, context=context)


def sample_plot(train_data, gen_data, filename):
    fig, axs = plt.subplots(1, 2, figsize=(6.,3.), dpi=200)
    for ax in axs:
        ax.hlines(0.0, xmin=-5.0, xmax=9.0, colors='k', alpha=0.6, zorder=0)
        ax.vlines(0.0, ymin=-9.0, ymax=9.0, colors='k', alpha=0.6, zorder=0)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")
    data, labels = train_data
    axs[0].scatter(*data.T, s=0.5, c=labels[:,0], cmap="PiYG", alpha=0.7, zorder=10)
    data, labels = gen_data
    axs[1].scatter(*data.T, s=0.5, c=labels[:,0], cmap="PiYG", alpha=0.7, zorder=10)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(f"samples/" + filename, bbox_inches="tight")
    plt.close()


def main(_):

    optimizer = optax.adam(FLAGS.learning_rate)
	
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

    prng_seq = hk.PRNGSequence(42)

    n_sample = 4000 # for plotting
	
	# initalise params with fixed data 
    data, context = get_batch(FLAGS.batch_size, next(prng_seq))
    params = log_prob.init(next(prng_seq), data, context=context)
    opt_state = optimizer.init(params)
    print("initialised flow.\n")

    losses = []
    for step in range(FLAGS.training_steps):

        train_data, context = get_batch(FLAGS.batch_size, next(prng_seq))
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

            # batch of both different classes 
            context = jnp.concatenate(
		         [jnp.zeros((n_sample // 2, *CONTEXT_SHAPE)), 
				  jnp.ones((n_sample // 2, *CONTEXT_SHAPE))])
			
            train_data = get_batch(n_sample, next(prng_seq))
            gen_data = (sample(params, next(prng_seq), context), context) # should this have a fixed key?
            sample_plot(train_data, gen_data, filename=f"sample{step:06d}.png")

            logging.info("STEP: %5d; Validation loss: %.3f", step, val_loss)
            losses.append(val_loss)

    plt.figure()
    plt.plot(range(0, FLAGS.training_steps, FLAGS.eval_frequency), losses)
    plt.savefig("loss.png")


if __name__ == "__main__":
    app.run(main)
