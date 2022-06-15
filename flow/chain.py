"""Conditional Chain Bijector for composing a sequence of Bijectors."""

from typing import Optional, Tuple
import distrax

import jax
import jax.numpy as jnp
from chex import Array

from .bijector import ConditionalBijector



'''
class ConditionalChain(ConditionalBijector, distrax.Chain):
    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        """Computes y = f(x)."""
        for bijector in reversed(self._bijectors):
            if isinstance(bijector, ConditionalBijector):
                x = bijector.forward(x, context)
            else:
                x = bijector.forward(x)
        return x

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        """Computes x = f^{-1}(y)."""
        for bijector in self._bijectors:
            if isinstance(bijector, ConditionalBijector):
                y = bijector.inverse(y, context)
            else:
                y = bijector.inverse(y)
        return y

    def forward_and_log_det(
        self, x: Array, context: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""

        bijector = self._bijectors[-1]
        if isinstance(bijector, ConditionalBijector):
            x, log_det = bijector.forward_and_log_det(x, context)
        else:
            x, log_det = bijector.forward_and_log_det(x)

        for bijector in reversed(self._bijectors[:-1]):
            if isinstance(bijector, ConditionalBijector):
                x, ld = bijector.forward_and_log_det(x, context)
            else:
                x, ld = bijector.forward_and_log_det(x)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(
        self, 
		y: Array, 
		context: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""

        bijector = self._bijectors[0]
        if isinstance(bijector, ConditionalBijector):
            y, log_det = bijector.inverse_and_log_det(y, context)
        else:
            y, log_det = bijector.inverse_and_log_det(y)

        for bijector in self._bijectors[1:]:
            if isinstance(bijector, ConditionalBijector):
                y, ld = bijector.inverse_and_log_det(y, context)
            else:
                y, ld = bijector.inverse_and_log_det(y)
            log_det += ld

        return y, log_det
'''

# i think not using a distrax.Inverse like object to reverse the flow causes serious
# problems here... 
# > all the Inverse needs to do is switch forward <-> inverse for each flow step

class ConditionalChain(ConditionalBijector, distrax.Chain):
    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        """Computes y = f(x)."""
        for bijector in reversed(self._bijectors):
            x = bijector.forward(x, context=context)
        return x

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        """Computes x = f^{-1}(y)."""
        for bijector in self._bijectors:
            y = bijector.inverse(y, context=context)
        return y

    def forward_and_log_det(
        self, x: Array, context: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""

        bijector = self._bijectors[-1]
        x, log_det = bijector.forward_and_log_det(x, context=context)

        for bijector in reversed(self._bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x, context=context)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(
        self, 
		y: Array, 
		context: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""

        bijector = self._bijectors[0]
        y, log_det = bijector.inverse_and_log_det(y, context=context)

        for bijector in self._bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y, context=context)
            log_det += ld

        return y, log_det
