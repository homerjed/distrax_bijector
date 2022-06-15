"""Conditional Masked coupling bijector."""

import distrax
from distrax._src.utils import math

import jax
import jax.numpy as jnp

from typing import Callable, Optional, Tuple, List, Any
from chex import Array

from .bijector import ConditionalBijector

BijectorParams = Any


class ConditionalMaskedCoupling(ConditionalBijector, distrax.MaskedCoupling):
    """Coupling bijector that uses a mask to specify which inputs are transformed
  	   and can input an additional context variable
	   - inherits two "conditioners" one from ConditionalBijector the other from
	     distrax.MaskedCoupling
    """

    def __init__(
			self, 
			conditioner_eta: Callable[[Array], BijectorParams], 
			**kwargs
	):
        self._conditioner_eta = conditioner_eta
        super().__init__(**kwargs)

    def forward_and_log_det(
        self, 
		x: Array, 
		context: Array,
    ) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|.
		Args:
		  x: Array
		  context: List of two elements:
			-eta: an array
			-Context: array, could be None
		  > assuming context is not a List, but an Array
	    """
        self._check_forward_input_shape(x)
        masked_x = jnp.where(self._event_mask, x, 0.0)

        conditioner_input = masked_x
        conditioner_input = jnp.concatenate([masked_x, context], axis=-1)
		# try broadcasting the context to the dimension of x
        #jnp.broadcast_to(
        #    context, x.shape[:-1] + (context.shape[-1],)
        #),

		# condition on input + context (contextualise before condition?)
        params = self._conditioner(conditioner_input) 
		#params += self._conditioner_eta(context)
		# try and broadcast contexting over params
        params += jnp.broadcast_to(
				jnp.expand_dims(self._conditioner_eta(context), axis=-1), params.shape)

        y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
        y = jnp.where(self._event_mask, x, y0)
        logdet = math.sum_last(
            jnp.where(self._mask, 0.0, log_d),
            self._event_ndims - self._inner_event_ndims,
        )
        return y, logdet

    def inverse_and_log_det(
        self, 
		y: Array, 
		context: Array,
    ) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        self._check_inverse_input_shape(y)
        masked_y = jnp.where(self._event_mask, y, 0.0)

		#conditioner_input = masked_y
		#conditioner_input = jnp.concatenate([masked_y, context], axis=-1)
        #conditioner_input = jnp.broadcast_to(context, y.shape[:-1] + (context.shape[-1],)) 
        #conditioner_input = jnp.concatenate([masked_y, context], axis=-1)
        #jnp.broadcast_to(context, y.shape[:-1] + (context.shape[-1],)),
        conditioner_input = jnp.concatenate([masked_y, context], axis=-1)

		# condition on input + context 
        params = self._conditioner(conditioner_input) 
		#params += self._conditioner_eta(context)

		# try and broadcast contexting over params
        params += jnp.broadcast_to(
				jnp.expand_dims(self._conditioner_eta(context), axis=-1), params.shape)

        x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
        x = jnp.where(self._event_mask, y, x0)
        logdet = math.sum_last(
            jnp.where(self._mask, 0.0, log_d),
            self._event_ndims - self._inner_event_ndims,
        )
        return x, logdet 
