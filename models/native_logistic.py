"""Independent-source flow matching with a product-logistic kernel and its score head.

The source Z has independent unit-variance logistic coordinates, so the endpoint kernel is a
genuine non-Gaussian convolution kernel: p(z|x)=p_Z does not depend on x and keeps its shape as
the scale vanishes. Training is the native rectified-flow regression on Y_t=tX+(1-t)Z with
uniform t and the benchmark RF condition 999t.

For a non-Gaussian kernel the velocity alone does not determine the density derivative. With
U=(Y-X)/lambda and psi=-grad log p_Z,

    Full = D + dlog p_lambda/dlog lambda = tr J_q + E[U|y] . E[psi(U)|y],

which follows from the scale continuity equation together with grad log p_lambda = -E[psi(U)|y]/lambda.
The Gaussian case psi(u)=u recovers Response+||q-x||^2/lambda^2. This model therefore carries two
copies of the shared field core, exactly as the DSB row does: one regresses the velocity X-Z and one
regresses psi(Z). Both are updated on the same batch, so each core receives the common step budget.
"""
import math

import torch

from models.native_tasks import NativeField, _condition

S0 = math.sqrt(3)/math.pi                      # logistic scale with unit variance
CONDITION_MULTIPLIER = 999.                    # benchmark rectified-flow time units


def logistic_noise(shape, *, device, dtype, generator):
    uniform = torch.rand(shape, device=device, dtype=dtype, generator=generator).clamp(1e-7, 1-1e-7)
    return S0*(torch.log(uniform)-torch.log1p(-uniform))


def kernel_score(z):
    """psi(z) = -d log p_Z(z) / dz for the unit-variance logistic density; |psi| <= 1/S0."""
    return torch.tanh(z/(2*S0))/S0


class NativeLogisticFlow(NativeField):
    """Velocity core plus kernel-score core; the shared backbone is used unchanged."""

    def __init__(self, architecture, family, cfg):
        super().__init__(architecture, family, cfg)
        import copy
        self.score_core = copy.deepcopy(self.core)

    def forward(self, x, condition):
        return self.raw(x, condition, self.core)

    def kernel_score_output(self, x, condition):
        return self.raw(x, condition, self.score_core)

    def _native_time(self, scale, x):
        lam = _condition(scale, x)
        return 1/(1+lam), lam

    def canonical_posterior(self, x, scale):
        """q(x,lambda)=E[X|Y=x]; lambda is the per-coordinate RMS noise, so t=1/(1+lambda)."""
        t, _ = self._native_time(scale, x)
        return t[:, None]*x+(1-t)[:, None]*self(t[:, None]*x, CONDITION_MULTIPLIER*t)

    def posterior_kernel_score(self, x, scale):
        """E[psi(Z) | Y=x] at the same query and scale."""
        t, _ = self._native_time(scale, x)
        return self.kernel_score_output(t[:, None]*x, CONDITION_MULTIPLIER*t)

    def batch(self, clean, generator):
        n = len(clean)
        noise = logistic_noise(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        time = torch.rand(n, device=clean.device, dtype=clean.dtype, generator=generator)
        point = time[:, None]*clean+(1-time)[:, None]*noise
        return point, CONDITION_MULTIPLIER*time, clean-noise, kernel_score(noise)

    def two_head_loss(self, clean, generator):
        """Unweighted coordinate-mean MSE of both heads on one shared corruption draw."""
        point, condition, velocity, score = self.batch(clean, generator)
        return ((self(point, condition)-velocity).square().mean()
                + (self.kernel_score_output(point, condition)-score).square().mean())
