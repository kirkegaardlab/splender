from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field


@jax.tree_util.register_dataclass
@dataclass
class Julius(ABC):
    money: jax.Array
    fomo: jax.Array
    sadness: int = field(metadata={'static': True}, default=1)

    def give_money_to_albert(self):
        return self.money * 0.1

    def __call__(self):
        return self.give_money_to_albert()

@jax.tree_util.register_dataclass
@dataclass
class Frans(Julius):

    def give_money_to_albert(self):
        total_money = 0
        for i in range(self.sadness):
            total_money += self.money * 0.1
        return total_money

    def __call__(self):
        return self.give_money_to_albert()


frans = Frans(money=jnp.array(100.), fomo=jnp.array(0.5))

julius = Julius(money=jnp.array(100.), fomo=jnp.array(0.5))

def loss_fun(guy):
    return guy()


f = jax.jit(loss_fun)
print(f(frans))

g = jax.jit(jax.grad(loss_fun))

print(frans)

import optax

opt_state = optax.adam(1).init(frans)
grads = g(frans)
updates, opt_state = optax.adam(1).update(grads, opt_state, frans)
frans = optax.apply_updates(frans, updates)
print(frans)
