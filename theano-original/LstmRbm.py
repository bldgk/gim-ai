"""
LSTM-RBM: Long Short-Term Memory + Restricted Boltzmann Machine
for polyphonic music generation.

Extends the RNN-RBM (Boulanger-Lewandowski, 2012) by replacing
the simple recurrent layer with an LSTM, which better captures
long-range temporal dependencies in music.
"""
import gim
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False


def normal(num_rows, num_cols, scale=1):
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def zeros(*shape):
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


class Rbm:
    """Restricted Boltzmann Machine component."""

    def __init__(self, v_power, h_power):
        self.v_power = v_power
        self.h_power = h_power
        self.W = normal(v_power, h_power, 0.01)
        self.c = zeros(v_power)   # visible bias
        self.b = zeros(h_power)   # hidden bias

    def build(self, v, W, c, b, k):
        """Build a k-step Gibbs chain for CD learning."""
        def gibbs_step(v):
            mean_h = T.nnet.sigmoid(T.dot(v, W) + b)
            h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                             dtype=theano.config.floatX)
            mean_v = T.nnet.sigmoid(T.dot(h, W.T) + c)
            v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                             dtype=theano.config.floatX)
            return mean_v, v

        chain, updates = theano.scan(
            lambda v: gibbs_step(v)[1], outputs_info=[v], n_steps=k)
        v_sample = chain[-1]

        mean_v = gibbs_step(v_sample)[0]
        monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
        monitor = monitor.sum() / v.shape[0]

        def free_energy(v):
            return -(v * c).sum() - T.log(1 + T.exp(T.dot(v, W) + b)).sum()

        cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]
        return v_sample, cost, monitor, updates

    def params(self):
        return (self.W, self.c, self.b)


class Lstm:
    """LSTM layer with input, forget, output gates and cell state."""

    def __init__(self, v_power, r_power):
        # Input transform
        self.Wvx = normal(v_power, r_power, 0.0001)
        self.Wyx = normal(r_power, r_power, 0.0001)
        self.bx = zeros(r_power)
        # Input gate
        self.Wxi = normal(r_power, r_power, 0.0001)
        self.Wyi = normal(r_power, r_power, 0.0001)
        self.Wci = normal(r_power, r_power, 0.0001)
        self.bi = zeros(r_power)
        # Forget gate
        self.Wxf = normal(r_power, r_power, 0.0001)
        self.Wyf = normal(r_power, r_power, 0.0001)
        self.Wcf = normal(r_power, r_power, 0.0001)
        self.bf = zeros(r_power)
        # Cell
        self.Wxc = normal(r_power, r_power, 0.0001)
        self.Wyc = normal(r_power, r_power, 0.0001)
        self.bc = zeros(r_power)
        # Output gate
        self.Wxo = normal(r_power, r_power, 0.0001)
        self.Wyo = normal(r_power, r_power, 0.0001)
        self.Wco = normal(r_power, r_power, 0.0001)
        self.bo = zeros(r_power)

    def step(self, v_t, x_prev, y_prev, c_prev):
        """One LSTM step: takes input v_t and previous states, returns new states."""
        # Input transform
        x_t = T.tanh(self.bx + T.dot(v_t, self.Wvx) + T.dot(x_prev, self.Wyx))
        # Input gate
        i_t = T.nnet.sigmoid(self.bi + T.dot(c_prev, self.Wci) + T.dot(y_prev, self.Wyi) + T.dot(x_t, self.Wxi))
        # Forget gate
        f_t = T.nnet.sigmoid(self.bf + T.dot(c_prev, self.Wcf) + T.dot(y_prev, self.Wyf) + T.dot(x_t, self.Wxf))
        # Cell state
        c_t = f_t * c_prev + i_t * T.tanh(T.dot(x_t, self.Wxc) + T.dot(y_prev, self.Wyc) + self.bc)
        # Output gate
        o_t = T.nnet.sigmoid(self.bo + T.dot(c_t, self.Wco) + T.dot(y_prev, self.Wyo) + T.dot(x_t, self.Wxo))
        # Hidden state
        y_t = o_t * T.tanh(c_t)
        return x_t, c_t, y_t

    def params(self):
        return (self.Wvx, self.Wyx, self.bx,
                self.Wxi, self.Wyi, self.Wci, self.bi,
                self.Wxf, self.Wyf, self.Wcf, self.bf,
                self.Wxc, self.Wyc, self.bc,
                self.Wxo, self.Wyo, self.Wco, self.bo)


class LstmRbm(gim.Model):
    """LSTM-RBM model for music generation."""

    def build(self, v_power, h_power, r_power):
        rbm = Rbm(v_power, h_power)
        lstm = Lstm(v_power, r_power)

        # Connection weights: LSTM hidden → RBM biases
        Wyh = normal(r_power, h_power, 0.0001)
        Wyv = normal(r_power, v_power, 0.0001)

        x0 = T.zeros((r_power,))
        y0 = T.zeros((r_power,))
        c0 = T.zeros((r_power,))

        params = rbm.params() + (Wyh, Wyv) + lstm.params()

        v = T.matrix()  # training sequence (time × visible)

        def recurrence(v_t, x_prev, y_prev, c_prev):
            # Conditional RBM biases from LSTM hidden state
            bv_t = rbm.c + T.dot(y_prev, Wyv)
            bh_t = rbm.b + T.dot(y_prev, Wyh)
            generate = v_t is None
            if generate:
                v_t, _, _, updates = rbm.build(
                    T.zeros((v_power,)), rbm.W, bv_t, bh_t, k=25)
            # LSTM step
            x_t, c_t, y_t = lstm.step(v_t, x_prev, y_prev, c_prev)
            if generate:
                return [v_t, x_t, y_t, c_t], updates
            else:
                return [x_t, y_t, c_t, bv_t, bh_t]

        # Training: deterministic recurrence to get time-varying biases
        (x_t, y_t, c_t, bv_t, bh_t), updates_t = theano.scan(
            lambda v_t, x_prev, y_prev, c_prev, *_: recurrence(v_t, x_prev, y_prev, c_prev),
            sequences=v,
            outputs_info=[x0, y0, c0, None, None],
            non_sequences=params)

        v_sample, cost, monitor, updates_rbm = rbm.build(
            v, rbm.W, bv_t[:], bh_t[:], k=15)
        updates_t.update(updates_rbm)

        # Generation: sample from RBM at each step
        (v_t, x_t, y_t, c_t), updates_g = theano.scan(
            lambda x_prev, y_prev, c_prev, *_: recurrence(None, x_prev, y_prev, c_prev),
            outputs_info=[None, x0, y0, c0],
            non_sequences=params,
            n_steps=200)

        return (v, v_sample, cost, monitor, params, updates_t, v_t, updates_g)
