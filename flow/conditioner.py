""" not in use... """

import haiku as hk
from typing import Seqeunce, Optional

class CouplingConditioner(hk.Module):

  def __init__(
      self,
      output_dim: int,
      hidden_sizes: Sequence[int],
      num_bijector_params: int,
      name: Optional[str] = '',
  ):
    super().__init__(name=name)
    self.output_dim = output_dim
    self.hidden_sizes = hidden_sizes
    self.num_bijector_params = num_bijector_params

  def __call__(self, inputs):

    out = hk.Flatten(preserve_dims=-1)(inputs)
    out = hk.nets.MLP(self.hidden_sizes, activate_final=True)(out)

    # We initialize this linear layer to zero so that the flow is initialized
    # to the identity function.
    out = hk.Linear(
        self.output_dim * self.num_bijector_params,
        w_init=jnp.zeros,
        b_init=jnp.zeros)(
            out)
    out = hk.Reshape(
        (self.output_dim,) + (self.num_bijector_params,), preserve_dims=-1)(
            out)

    return 
