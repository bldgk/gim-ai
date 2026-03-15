"""
Music generation models in PyTorch.

Four model variants combining recurrent layers with generative models:
  - RnnRbm:  RNN  + single RBM  (Boulanger-Lewandowski, 2012)
  - LstmRbm: LSTM + single RBM  (better long-range dependencies)
  - RnnDbn:  RNN  + stacked RBM (Deep Belief Network, Kratarth Goel, 2014)
  - LstmDbn: LSTM + stacked RBM (LSTM + DBN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class BaseRbmModel(nn.Module):
    """Shared RBM + recurrent base class."""

    def __init__(self, n_visible=88, n_hidden=150, n_recurrent=100):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_recurrent = n_recurrent

        # RBM parameters
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.bv = nn.Parameter(torch.zeros(n_visible))
        self.bh = nn.Parameter(torch.zeros(n_hidden))

        # Recurrent hidden → RBM bias modulation
        self.Wyv = nn.Linear(n_recurrent, n_visible, bias=False)
        self.Wyh = nn.Linear(n_recurrent, n_hidden, bias=False)
        nn.init.normal_(self.Wyv.weight, std=0.0001)
        nn.init.normal_(self.Wyh.weight, std=0.0001)

    def _cd_step(self, v_seq, lstm_shifted, k=5):
        """CD-k training on all timesteps in parallel. Returns cost, monitor."""
        bv_t = self.bv + self.Wyv(lstm_shifted)
        bh_t = self.bh + self.Wyh(lstm_shifted)

        v_sample = v_seq.clone()
        for _ in range(k):
            mean_h = torch.sigmoid(v_sample @ self.W + bh_t)
            h = torch.bernoulli(mean_h)
            mean_v = torch.sigmoid(h @ self.W.t() + bv_t)
            v_sample = torch.bernoulli(mean_v)

        def free_energy(v):
            return -(v * bv_t).sum(-1) - torch.log(1 + torch.exp(v @ self.W + bh_t)).sum(-1)

        cost = (free_energy(v_seq) - free_energy(v_sample.detach())).mean()
        monitor = (v_seq * torch.log(mean_v + 1e-10) +
                   (1 - v_seq) * torch.log(1 - mean_v + 1e-10)).sum() / v_seq.shape[-2]
        return cost, monitor

    def _cd_step_batched(self, padded, mask, lstm_shifted, k=5):
        """CD-k on a padded batch. Returns cost, monitor."""
        bv_t = self.bv + self.Wyv(lstm_shifted)
        bh_t = self.bh + self.Wyh(lstm_shifted)

        v_sample = padded.clone()
        for _ in range(k):
            mean_h = torch.sigmoid(v_sample @ self.W + bh_t)
            h = torch.bernoulli(mean_h)
            mean_v = torch.sigmoid(h @ self.W.t() + bv_t)
            v_sample = torch.bernoulli(mean_v)

        def free_energy(v):
            return -(v * bv_t).sum(-1) - torch.log(1 + torch.exp(v @ self.W + bh_t)).sum(-1)

        cost = ((free_energy(padded) - free_energy(v_sample.detach())) * mask).sum() / mask.sum()
        monitor = ((padded * torch.log(mean_v + 1e-10) +
                    (1 - padded) * torch.log(1 - mean_v + 1e-10)).sum(-1) * mask).sum() / mask.sum()
        return cost, monitor

    def _generate_loop(self, rnn_step_fn, length=200, k=25, temperature=1.0, seed=None):
        """
        Shared generation loop.
        temperature: <1 = more conservative, >1 = more random/creative
        seed: random seed for reproducibility, None = random each time
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = next(self.parameters()).device

        # Random initial input instead of zeros — gives variety
        v_t = torch.bernoulli(torch.ones(1, self.n_visible, device=device) * 0.1)
        state = self._init_gen_state(device)

        piano_roll = []
        for _ in range(length):
            h_rec = self._get_hidden(state)
            bv_t = self.bv + self.Wyv(h_rec).squeeze(0)
            bh_t = self.bh + self.Wyh(h_rec).squeeze(0)

            for _ in range(k):
                mean_h = torch.sigmoid((v_t @ self.W + bh_t) / temperature)
                h_rbm = torch.bernoulli(mean_h)
                mean_v = torch.sigmoid((h_rbm @ self.W.t() + bv_t) / temperature)
                v_t = torch.bernoulli(mean_v)

            piano_roll.append(v_t.squeeze(0).clone())
            state = rnn_step_fn(v_t, state)

        return torch.stack(piano_roll).cpu().numpy()


class RnnRbm(BaseRbmModel):
    """RNN-RBM: simple tanh RNN + RBM (Boulanger-Lewandowski, 2012)."""

    def __init__(self, n_visible=88, n_hidden=150, n_recurrent=100):
        super().__init__(n_visible, n_hidden, n_recurrent)
        self.rnn = nn.RNN(n_visible, n_recurrent, batch_first=True)
        self.rnn_cell = nn.RNNCell(n_visible, n_recurrent)

    def forward(self, v_seq, k=5):
        device = v_seq.device
        rnn_out, _ = self.rnn(v_seq.unsqueeze(0))
        rnn_out = rnn_out.squeeze(0)
        zero = torch.zeros(1, self.n_recurrent, device=device)
        shifted = torch.cat([zero, rnn_out[:-1]], dim=0)
        return self._cd_step(v_seq, shifted, k)

    def forward_batch(self, sequences, k=5):
        device = next(self.parameters()).device
        lengths = [s.shape[0] for s in sequences]
        padded = pad_sequence(sequences, batch_first=True)
        mask = torch.arange(padded.shape[1], device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
        mask = mask.float()

        rnn_out, _ = self.rnn(padded)
        zero = torch.zeros(padded.shape[0], 1, self.n_recurrent, device=device)
        shifted = torch.cat([zero, rnn_out[:, :-1, :]], dim=1)
        return self._cd_step_batched(padded, mask, shifted, k)

    def _init_gen_state(self, device):
        return torch.zeros(1, self.n_recurrent, device=device)

    def _get_hidden(self, state):
        return state

    @torch.no_grad()
    def generate(self, length=200, k=25, temperature=1.0, seed=None):
        self.eval()
        result = self._generate_loop(
            lambda v, h: self.rnn_cell(v, h), length, k, temperature, seed)
        self.train()
        return result


class LstmRbm(BaseRbmModel):
    """LSTM-RBM: LSTM + RBM (better long-range musical structure)."""

    def __init__(self, n_visible=88, n_hidden=150, n_recurrent=100):
        super().__init__(n_visible, n_hidden, n_recurrent)
        self.lstm = nn.LSTM(n_visible, n_recurrent, batch_first=True)
        self.lstm_cell = nn.LSTMCell(n_visible, n_recurrent)

    def forward(self, v_seq, k=5):
        device = v_seq.device
        lstm_out, _ = self.lstm(v_seq.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)
        zero = torch.zeros(1, self.n_recurrent, device=device)
        shifted = torch.cat([zero, lstm_out[:-1]], dim=0)
        return self._cd_step(v_seq, shifted, k)

    def forward_batch(self, sequences, k=5):
        device = next(self.parameters()).device
        lengths = [s.shape[0] for s in sequences]
        padded = pad_sequence(sequences, batch_first=True)
        mask = torch.arange(padded.shape[1], device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
        mask = mask.float()

        lstm_out, _ = self.lstm(padded)
        zero = torch.zeros(padded.shape[0], 1, self.n_recurrent, device=device)
        shifted = torch.cat([zero, lstm_out[:, :-1, :]], dim=1)
        return self._cd_step_batched(padded, mask, shifted, k)

    def _init_gen_state(self, device):
        return (torch.zeros(1, self.n_recurrent, device=device),
                torch.zeros(1, self.n_recurrent, device=device))

    def _get_hidden(self, state):
        return state[0]  # h from (h, c)

    @torch.no_grad()
    def generate(self, length=200, k=25, temperature=1.0, seed=None):
        self.eval()
        result = self._generate_loop(
            lambda v, s: self.lstm_cell(v, s), length, k, temperature, seed)
        self.train()
        return result


# ── DBN Models (stacked RBM = Deep Belief Network) ─────────

class BaseDbnModel(nn.Module):
    """
    DBN base: two stacked RBMs + recurrent layer.
    RBM1: visible ↔ hidden1 (learns note patterns)
    RBM2: hidden1 ↔ hidden2 (learns higher-level structure)
    Generation: sample h1 from RBM2, then v from h1 via W1.T
    """

    def __init__(self, n_visible=88, n_hidden=150, n_recurrent=100):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_recurrent = n_recurrent

        # RBM1: visible ↔ hidden1
        self.W1 = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.bv = nn.Parameter(torch.zeros(n_visible))
        self.bh1 = nn.Parameter(torch.zeros(n_hidden))

        # RBM2: hidden1 ↔ hidden2
        self.W2 = nn.Parameter(torch.randn(n_hidden, n_hidden) * 0.01)
        self.bh2 = nn.Parameter(torch.zeros(n_hidden))

        # Recurrent → bias modulation
        self.Wyv = nn.Linear(n_recurrent, n_visible, bias=False)
        self.Wyh1 = nn.Linear(n_recurrent, n_hidden, bias=False)
        self.Wyh2 = nn.Linear(n_recurrent, n_hidden, bias=False)

        for m in [self.Wyv, self.Wyh1, self.Wyh2]:
            nn.init.normal_(m.weight, std=0.0001)

    def forward(self, v_seq, k=5):
        device = v_seq.device
        seq_len = v_seq.shape[0]

        rnn_out = self._run_rnn(v_seq)
        zero = torch.zeros(1, self.n_recurrent, device=device)
        shifted = torch.cat([zero, rnn_out[:-1]], dim=0)

        bv_t = self.bv + self.Wyv(shifted)
        bh1_t = self.bh1 + self.Wyh1(shifted)
        bh2_t = self.bh2 + self.Wyh2(shifted)

        # Train RBM1: visible ↔ hidden1
        v_sample = v_seq.clone()
        for _ in range(k):
            mean_h = torch.sigmoid(v_sample @ self.W1 + bh1_t)
            h = torch.bernoulli(mean_h)
            mean_v = torch.sigmoid(h @ self.W1.t() + bv_t)
            v_sample = torch.bernoulli(mean_v)

        def free_energy1(v):
            return -(v * bv_t).sum(-1) - torch.log(1 + torch.exp(v @ self.W1 + bh1_t)).sum(-1)

        cost1 = (free_energy1(v_seq) - free_energy1(v_sample.detach())).mean()

        # Train RBM2: hidden1 ↔ hidden2
        h1 = torch.sigmoid(v_seq @ self.W1 + bh1_t)
        h1_sample = h1.clone()
        for _ in range(k):
            mean_h2 = torch.sigmoid(h1_sample @ self.W2 + bh2_t)
            h2 = torch.bernoulli(mean_h2)
            mean_h1 = torch.sigmoid(h2 @ self.W2.t() + bh1_t)
            h1_sample = torch.bernoulli(mean_h1)

        def free_energy2(h):
            return -(h * bh1_t).sum(-1) - torch.log(1 + torch.exp(h @ self.W2 + bh2_t)).sum(-1)

        cost2 = (free_energy2(h1) - free_energy2(h1_sample.detach())).mean()

        monitor = (v_seq * torch.log(mean_v + 1e-10) +
                   (1 - v_seq) * torch.log(1 - mean_v + 1e-10)).sum() / seq_len

        return cost1 + cost2, monitor

    def forward_batch(self, sequences, k=5):
        device = next(self.parameters()).device
        lengths = [s.shape[0] for s in sequences]
        padded = pad_sequence(sequences, batch_first=True)
        mask = torch.arange(padded.shape[1], device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
        mask = mask.float()

        rnn_out = self._run_rnn_batch(padded)
        zero = torch.zeros(padded.shape[0], 1, self.n_recurrent, device=device)
        shifted = torch.cat([zero, rnn_out[:, :-1, :]], dim=1)

        bv_t = self.bv + self.Wyv(shifted)
        bh1_t = self.bh1 + self.Wyh1(shifted)
        bh2_t = self.bh2 + self.Wyh2(shifted)

        # RBM1
        v_sample = padded.clone()
        for _ in range(k):
            h = torch.bernoulli(torch.sigmoid(v_sample @ self.W1 + bh1_t))
            mean_v = torch.sigmoid(h @ self.W1.t() + bv_t)
            v_sample = torch.bernoulli(mean_v)

        def fe1(v):
            return -(v * bv_t).sum(-1) - torch.log(1 + torch.exp(v @ self.W1 + bh1_t)).sum(-1)
        cost1 = ((fe1(padded) - fe1(v_sample.detach())) * mask).sum() / mask.sum()

        # RBM2
        h1 = torch.sigmoid(padded @ self.W1 + bh1_t)
        h1_sample = h1.clone()
        for _ in range(k):
            h2 = torch.bernoulli(torch.sigmoid(h1_sample @ self.W2 + bh2_t))
            h1_sample = torch.bernoulli(torch.sigmoid(h2 @ self.W2.t() + bh1_t))

        def fe2(h):
            return -(h * bh1_t).sum(-1) - torch.log(1 + torch.exp(h @ self.W2 + bh2_t)).sum(-1)
        cost2 = ((fe2(h1) - fe2(h1_sample.detach())) * mask).sum() / mask.sum()

        monitor = ((padded * torch.log(mean_v + 1e-10) +
                    (1 - padded) * torch.log(1 - mean_v + 1e-10)).sum(-1) * mask).sum() / mask.sum()

        return cost1 + cost2, monitor

    @torch.no_grad()
    def _generate_dbn(self, rnn_step_fn, length=200, k=25, temperature=1.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        device = next(self.parameters()).device
        v_t = torch.bernoulli(torch.ones(1, self.n_visible, device=device) * 0.1)
        state = self._init_gen_state(device)

        piano_roll = []
        for _ in range(length):
            h_rec = self._get_hidden(state)
            bh1_t = self.bh1 + self.Wyh1(h_rec).squeeze(0)
            bh2_t = self.bh2 + self.Wyh2(h_rec).squeeze(0)

            # Sample h1 from RBM2
            h1 = torch.zeros(1, self.n_hidden, device=device)
            for _ in range(k):
                mean_h2 = torch.sigmoid((h1 @ self.W2 + bh2_t) / temperature)
                h2 = torch.bernoulli(mean_h2)
                mean_h1 = torch.sigmoid((h2 @ self.W2.t() + bh1_t) / temperature)
                h1 = torch.bernoulli(mean_h1)

            # Generate visible from h1
            mean_v = torch.sigmoid((h1 @ self.W1.t() + self.bv) / temperature)
            v_t = torch.bernoulli(mean_v)

            piano_roll.append(v_t.squeeze(0).clone())
            state = rnn_step_fn(v_t, state)

        return torch.stack(piano_roll).cpu().numpy()


class RnnDbn(BaseDbnModel):
    """RNN-DBN: RNN + Deep Belief Network (Kratarth Goel, ICANN 2014)."""

    def __init__(self, n_visible=88, n_hidden=150, n_recurrent=100):
        super().__init__(n_visible, n_hidden, n_recurrent)
        self.rnn = nn.RNN(n_visible, n_recurrent, batch_first=True)
        self.rnn_cell = nn.RNNCell(n_visible, n_recurrent)

    def _run_rnn(self, v_seq):
        out, _ = self.rnn(v_seq.unsqueeze(0))
        return out.squeeze(0)

    def _run_rnn_batch(self, padded):
        out, _ = self.rnn(padded)
        return out

    def _init_gen_state(self, device):
        return torch.zeros(1, self.n_recurrent, device=device)

    def _get_hidden(self, state):
        return state

    @torch.no_grad()
    def generate(self, length=200, k=25, temperature=1.0, seed=None):
        self.eval()
        result = self._generate_dbn(
            lambda v, h: self.rnn_cell(v, h), length, k, temperature, seed)
        self.train()
        return result


class LstmDbn(BaseDbnModel):
    """LSTM-DBN: LSTM + Deep Belief Network (best long-range + deep features)."""

    def __init__(self, n_visible=88, n_hidden=150, n_recurrent=100):
        super().__init__(n_visible, n_hidden, n_recurrent)
        self.lstm = nn.LSTM(n_visible, n_recurrent, batch_first=True)
        self.lstm_cell = nn.LSTMCell(n_visible, n_recurrent)

    def _run_rnn(self, v_seq):
        out, _ = self.lstm(v_seq.unsqueeze(0))
        return out.squeeze(0)

    def _run_rnn_batch(self, padded):
        out, _ = self.lstm(padded)
        return out

    def _init_gen_state(self, device):
        return (torch.zeros(1, self.n_recurrent, device=device),
                torch.zeros(1, self.n_recurrent, device=device))

    def _get_hidden(self, state):
        return state[0]

    @torch.no_grad()
    def generate(self, length=200, k=25, temperature=1.0, seed=None):
        self.eval()
        result = self._generate_dbn(
            lambda v, s: self.lstm_cell(v, s), length, k, temperature, seed)
        self.train()
        return result
