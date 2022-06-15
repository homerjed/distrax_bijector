"""Wrapper for inverting a Distrax Bijector."""

from typing import Tuple

import distrax
from distrax._src.bijectors import bijector as base
from distrax._src.utils import conversion

from .bijector import ConditionalBijector

Array = base.Array
BijectorLike = base.BijectorLike
BijectorT = base.BijectorT


'''
# inherit from ConditionalBijector not base.Bijector?
#class ConditionalInverse(distrax.Inverse, base.Bijector):
#class ConditionalInverse(distrax.Inverse, ConditionalBijector):
class ConditionalInverse:
  """A bijector that inverts a given bijector.
  That is, if `bijector` implements the transformation `f`, `Inverse(bijector)`
  implements the inverse transformation `f^{-1}`.
  The inversion is performed by swapping the forward with the corresponding
  inverse methods of the given bijector.
  """

  #def __init__(self, bijector: BijectorLike):
  def __init__(self, bijector: ConditionalBijector):
    self._bijector = bijector
    super().__init__()

  @property
  def bijector(self) -> BijectorT:
    """The base bijector that was the input to `Inverse`."""
    return self._bijector

  def forward(self, x: Array, context: Array) -> Array:
    """Computes y = f(x)."""
    return self._bijector.inverse(x, context=context)

  def inverse(self, y: Array, context: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    return self._bijector.forward(y, context=context)

  def forward_log_det_jacobian(self, x: Array, context: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    return self._bijector.inverse_log_det_jacobian(x, context=context)

  def inverse_log_det_jacobian(self, y: Array, context: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    return self._bijector.forward_log_det_jacobian(y, context=context)

  def forward_and_log_det(self, x: Array, context: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return self._bijector.inverse_and_log_det(x, context=context)

  def inverse_and_log_det(self, y: Array, context: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self._bijector.forward_and_log_det(y, context=context)
'''

'''
  @property
  def name(self) -> str:
    """Name of the bijector."""
    return self.__class__.__name__ + self._bijector.name

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is Inverse:  # pylint: disable=unidiomatic-typecheck
      return self.bijector.same_as(other.bijector)
    return False
'''


class ConditionalInverse(base.Bijector):
  """A bijector that inverts a given bijector.
  That is, if `bijector` implements the transformation `f`, `Inverse(bijector)`
  implements the inverse transformation `f^{-1}`.
  The inversion is performed by swapping the forward with the corresponding
  inverse methods of the given bijector.
  """

  def __init__(self, bijector: BijectorLike):
    """Initializes an Inverse bijector.
    Args:
      bijector: the bijector to be inverted. It can be a distrax bijector, a TFP
        bijector, or a callable to be wrapped by `Lambda`.
    """
    self._bijector = conversion.as_bijector(bijector)
    super().__init__(
        event_ndims_in=self._bijector.event_ndims_out,
        event_ndims_out=self._bijector.event_ndims_in,
        is_constant_jacobian=self._bijector.is_constant_jacobian,
        is_constant_log_det=self._bijector.is_constant_log_det)

  @property
  def bijector(self) -> BijectorT:
    """The base bijector that was the input to `Inverse`."""
    return self._bijector

  def forward(self, x: Array, context: Array) -> Array:
    """Computes y = f(x)."""
    return self._bijector.inverse(x, context=context)

  def inverse(self, y: Array, context: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    return self._bijector.forward(y, context=context)

  def forward_log_det_jacobian(self, x: Array, context: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    return self._bijector.inverse_log_det_jacobian(x, context=context)

  def inverse_log_det_jacobian(self, y: Array, context: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    return self._bijector.forward_log_det_jacobian(y, context=context)

  def forward_and_log_det(self, x: Array, context: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return self._bijector.inverse_and_log_det(x, context=context)

  def inverse_and_log_det(self, y: Array, context: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self._bijector.forward_and_log_det(y, context=context)

  @property
  def name(self) -> str:
    """Name of the bijector."""
    return self.__class__.__name__ + self._bijector.name

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is Inverse:  # pylint: disable=unidiomatic-typecheck
      return self.bijector.same_as(other.bijector)
    return False
