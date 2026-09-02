"""
Denoising sine waves with a stacked LSTM.

A two-layer LSTM that takes a noisy 1-D signal and reconstructs the clean
signal underneath it. The LSTM cell is implemented from scratch (gates,
cell state, hidden state) rather than using torch.nn.LSTM.

Run: python lstm_noise_removal.py
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SEQUENCE_LENGTH = 80  # points per curve = time steps the LSTM walks
NUM_FUNCTIONS_TRAIN = 200  # curves used for training
NUM_FUNCTIONS_VAL = 50  # curves used for validation, never trained on
NOISE_RATIO = 0.05  # noise strength: 5% of each curve's range

TIME_AXIS = np.linspace(0, 5, SEQUENCE_LENGTH)

# ─── Data ─────────────────────────────────────────────────────────────────


class RandomSineFunction:
    """A sum of one to three sine waves with random amplitude, frequency, phase."""

    def __init__(self):
        num_waves = np.random.randint(1, 4)
        self.amplitudes = np.random.uniform(0, 2, num_waves)
        self.frequencies = np.random.uniform(0.1, 1, num_waves)
        self.phases = np.random.uniform(-np.pi, np.pi, num_waves)

    def __call__(self, time_points: np.ndarray) -> np.ndarray:
        """Evaluate the curve at the given time points."""
        total = np.zeros_like(time_points)

        for i in range(len(self.amplitudes)):
            amplitude = self.amplitudes[i]
            frequency = self.frequencies[i]
            phase = self.phases[i]

            # numpy applies sin() element-wise, so all 80 points at once
            wave = amplitude * np.sin(np.pi * frequency * time_points + phase)
            total = total + wave

        return total


def sample_sine_functions(num_functions: int) -> List[RandomSineFunction]:
    """Create a list of random curves. These are objects, not numbers yet."""
    functions = []

    for _ in range(num_functions):
        functions.append(RandomSineFunction())

    return functions


def add_noise(signal: np.ndarray, noise_ratio: float=NOISE_RATIO,
              axes: Tuple[int, ...]=None) -> np.ndarray:
    """Add Gaussian noise, scaled to the amplitude range of each curve."""
    # peak-to-peak = max - min = how tall the curve is
    curve_range = np.ptp(signal, axis=axes, keepdims=True)
    noise_strength = curve_range * noise_ratio

    # mean 0 so the noise wobbles around the signal instead of shifting it
    noise = np.random.normal(0, noise_strength, size=signal.shape)

    return signal + noise


def prepare_sequences(functions: List[RandomSineFunction]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn curve objects into a (clean, noisy) pair of (N, SEQUENCE_LENGTH, 1) tensors."""
    curves = []

    for function in functions:
        values = function(TIME_AXIS)
        # PyTorch wants three axes: (batch, seq_len, features)
        values = values.reshape(SEQUENCE_LENGTH, 1)
        curves.append(values)

    clean = np.array(curves)

    # axes=(1, 2) so each curve gets its own noise level
    noisy = add_noise(clean, NOISE_RATIO, axes=(1, 2))

    return torch.Tensor(clean), torch.Tensor(noisy)


def percentage_noise_removed(clean: np.ndarray, noisy: np.ndarray,
                             prediction: np.ndarray) -> float:
    """How much of the original noise the model removed, in percent."""
    noise_before = np.abs(clean - noisy).sum()
    noise_after = np.abs(clean - prediction).sum()

    removed = 100 * (1 - noise_after / noise_before)

    # a bad model can be worse than its input
    if removed < 0:
        return 0.0

    return removed

# ─── Model ────────────────────────────────────────────────────────────────


class LSTMCell(nn.Module):
    """One LSTM step: three gates plus a candidate value.

    All four are produced by a single linear layer of width 4 * hidden_size
    and split afterwards, which is faster than four separate matrix products.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, hidden_size * 4, bias=True)

    def forward(self, x: torch.Tensor,
                hx: Tuple[torch.Tensor, torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(batch, input_size) plus previous (h, C) -> new (h, C)."""
        if hx is None:
            hx = self._init_hidden_state(x)

        hidden_state, previous_cell_state = hx

        # [h_{t-1}, x_t] from the slides: concatenate along the feature axis
        concatenated = torch.cat((x, hidden_state), dim=1)
        gates = self.linear(concatenated)

        # order must match the order the update functions expect
        candidate_cell_state, forget_gate, input_gate, output_gate = torch.chunk(gates, 4, dim=1)

        # activations live in the update functions, not here
        current_cell_state = self.update_internal_state(
            forget_gate=forget_gate,
            previous_cell_state=previous_cell_state,
            input_gate=input_gate,
            candidate_cell_state=candidate_cell_state)
        new_hidden_state = self.update_hidden_state(current_cell_state, output_gate)

        return new_hidden_state, current_cell_state

    def update_internal_state(self, forget_gate: torch.Tensor, previous_cell_state: torch.Tensor,
                              input_gate: torch.Tensor,
                              candidate_cell_state: torch.Tensor) -> torch.Tensor:
        """C_t = f_t * C_{t-1} + i_t * C_tilde_t"""
        # sigmoid for gates (0..1 = how much to let through)
        forget_activated = torch.sigmoid(forget_gate)
        input_activated = torch.sigmoid(input_gate)

        # tanh for the candidate: it is information, not a gate, so it may be negative
        candidate_activated = torch.tanh(candidate_cell_state)

        # additive update: no weight matrix and no squashing on this path
        return forget_activated * previous_cell_state + input_activated * candidate_activated

    def update_hidden_state(self, current_cell_state: torch.Tensor,
                            output_gate: torch.Tensor) -> torch.Tensor:
        """h_t = o_t * tanh(C_t)"""
        output_activated = torch.sigmoid(output_gate)
        return output_activated * torch.tanh(current_cell_state)

    def _init_hidden_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero state for the first step of a sequence."""
        initial_hidden_state = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        initial_cell_state = torch.zeros_like(initial_hidden_state)
        return initial_hidden_state, initial_cell_state


class LSTM(nn.Module):
    """Applies one LSTMCell across a whole sequence."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm_cell = LSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, hx=None):
        """(batch, seq_len, input_size) -> (batch, seq_len, hidden_size), last (h, C)."""
        outputs = []

        for t in range(x.shape[1]):
            # x[:, t] is the whole batch at one time step
            hx = self.lstm_cell(x[:, t], hx)
            outputs.append(hx[0])  # hx[0] is h, hx[1] is C

        # stack adds a new axis: list of (batch, hidden) -> (batch, seq_len, hidden)
        return torch.stack(outputs, dim=1), hx


class NoiseRemovalModel(nn.Module):
    """Two stacked LSTMs followed by a linear projection back to one channel.

    The sequence is padded with `shift` zeros at the end and the first `shift`
    outputs are dropped, so the output for position t is produced after the
    model has already read `shift` steps beyond t.
    """

    def __init__(self, hidden_size: int, shift: int=10):
        super().__init__()
        self.shift = shift
        self.lstm1 = LSTM(1, hidden_size)
        self.lstm2 = LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, 1) -> (batch, seq_len, 1)"""
        # pad the time axis at the end: (0, 0) leaves features alone
        padded = nn.functional.pad(x, (0, 0, 0, self.shift))

        out = self.lstm1(padded)[0]  # [0] takes the stacked outputs, not (h, C)
        out = out[:, self.shift:]  # drop the lookahead steps, back to seq_len
        out = self.lstm2(out)[0]

        return self.linear(out)

# ─── Plots ────────────────────────────────────────────────────────────────


def plot_functions_with_noise(num_functions: int=6) -> None:
    """Show clean and noisy versions of several random sine functions."""
    plt.figure(figsize=(18, 8))

    for i, function in enumerate(sample_sine_functions(num_functions), start=1):
        plt.subplot(2, 3, i)
        clean = function(TIME_AXIS)
        plt.plot(TIME_AXIS, clean, label="Original")
        plt.plot(TIME_AXIS, add_noise(clean), label="Noisy")
        plt.xlabel("Time (s)")
        if i in (1, 4):
            plt.ylabel("Amplitude")
        plt.legend()

    plt.suptitle("Sine functions with and without Gaussian noise")
    plt.show()


def plot_predictions(clean: np.ndarray, noisy: np.ndarray, prediction: np.ndarray) -> None:
    """Show ground truth, noisy input and model output on the same axes."""
    plt.figure(figsize=(20, 5))

    for i in range(min(len(clean), 5)):
        plt.subplot(1, 5, i + 1)
        plt.plot(TIME_AXIS, clean[i], label="Ground truth")
        plt.plot(TIME_AXIS, noisy[i], label="Noisy input")
        plt.plot(TIME_AXIS, prediction[i], label="Model output")
        plt.xlabel("Time (s)")
        if i == 0:
            plt.ylabel("Amplitude")

    plt.legend(bbox_to_anchor=(1.04, 1), loc="best")
    plt.suptitle("Denoising results")
    plt.show()

# ─── Training ─────────────────────────────────────────────────────────────


def train(hidden_size: int=40, shift: int=10, lr: float=0.01,
          num_epochs: int=101, batch_size: int=10, plot: bool=True,
          seed: int=None) -> float:
    """Train the denoising model and return the final noise-removal percentage."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model = NoiseRemovalModel(hidden_size=hidden_size, shift=shift)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = nn.MSELoss()

    train_clean, train_noisy = prepare_sequences(sample_sine_functions(NUM_FUNCTIONS_TRAIN))
    val_clean, val_noisy = prepare_sequences(sample_sine_functions(NUM_FUNCTIONS_VAL))

    removed = 0.0
    for epoch in range(num_epochs):
        model.train()
        for start in range(0, len(train_noisy), batch_size):
            optimizer.zero_grad()
            prediction = model(train_noisy[start:start + batch_size])
            loss = loss_fn(prediction, train_clean[start:start + batch_size])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction = model(val_noisy)
            val_loss = loss_fn(prediction, val_clean)
            removed = percentage_noise_removed(val_clean.numpy(), val_noisy.numpy(),
                                               prediction.numpy())

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"epoch {epoch:3d} | val loss {val_loss:.5f} | {removed:5.2f}% noise removed")

        scheduler.step()

    if plot:
        with torch.no_grad():
            prediction = model(val_noisy).numpy()
        plot_predictions(val_clean.numpy(), val_noisy.numpy(), prediction)

    return removed


def run_hpo() -> None:
    """Compare hyperparameter settings on identical data (fixed seed)."""
    configs = [
        {"hidden_size": 40, "shift": 10, "num_epochs": 101},
        {"hidden_size": 40, "shift": 10, "num_epochs": 201},
        {"hidden_size": 20, "shift": 10, "num_epochs": 101},
        {"hidden_size": 60, "shift": 10, "num_epochs": 101},
        {"hidden_size": 40, "shift": 3, "num_epochs": 101},
    ]

    results = []
    for config in configs:
        print(f"\n--- {config} ---")
        removed = train(plot=False, seed=42, **config)
        results.append((config, removed))

    print("\nSummary")
    for config, removed in results:
        print(f"{removed:5.2f}%  {config}")


if __name__ == "__main__":
    plot_functions_with_noise()
    train()
