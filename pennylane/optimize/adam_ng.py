# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adam optimizer"""

import numpy as np

from pennylane.utils import _flatten, unflatten
from .qng import QNGOptimizer


class AdamNGOptimizer(QNGOptimizer):
    def __init__(self, stepsize=0.01, diag_approx=False, lam=0, beta1=0.9, beta2=0.99, eps=1e-8):
        super().__init__(stepsize, diag_approx=diag_approx, lam=lam)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.fm = None
        self.sm = None
        self.t = 0

    def apply_grad(self, grad, x):
        r"""Update the variables x to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (array): The gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            x (array): the current value of the variables :math:`x^{(t)}`

        Returns:
            array: the new values :math:`x^{(t+1)}`
        """

        self.t += 1

        grad_flat = np.array(list(_flatten(grad)))
        grad_flat = np.linalg.solve(self.metric_tensor, grad_flat)
        x_flat = np.array(list(_flatten(x)))

        # Update first moment
        if self.fm is None:
            self.fm = grad_flat
        else:
            self.fm = [self.beta1 * f + (1 - self.beta1) * g for f, g in zip(self.fm, grad_flat)]

        # Update second moment
        if self.sm is None:
            self.sm = [g * g for g in grad_flat]
        else:
            self.sm = [
                self.beta2 * f + (1 - self.beta2) * g * g for f, g in zip(self.sm, grad_flat)
            ]

        # Update step size (instead of correcting for bias)
        new_stepsize = (
            self._stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        )

        x_new_flat = [
            e - new_stepsize * f / (np.sqrt(s) + self.eps)
            for f, s, e in zip(self.fm, self.sm, x_flat)
        ]

        return unflatten(x_new_flat, x)

    def reset(self):
        """Reset optimizer by erasing memory of past steps."""
        self.fm = None
        self.sm = None
        self.t = 0
