"""
RNN-RBM and LSTM-RBM models in PyTorch for music generation.

Both models share the same RBM component but differ in the recurrent layer:
  - RnnRbm: simple tanh RNN (Boulanger-Lewandowski, 2012)
  - LstmRbm: LSTM with full gate mechanism (better long-range dependencies)
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
