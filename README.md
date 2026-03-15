# GameIntoMusic — AI Music Generator

> Originally written in Theano back in 2017, ported to PyTorch in 2026 because Theano died and wouldn't run on Apple Silicon. Generates something between music and auditory schizophrenia, depending on training time and your tolerance.

Deep learning model that generates music by learning temporal patterns from MIDI files.

## How it works

```
MIDI files (Nottingham folk songs)
    │
    ▼
┌──────────────┐
│  Piano-roll  │   Convert MIDI → binary matrix (time × 88 piano keys)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LSTM-RBM    │   LSTM captures temporal dependencies,
│  or RNN-RBM  │   RBM models the distribution at each time step
└──────┬───────┘
       │
       ▼
  Generated MIDI    Sample from the learned distribution
```

Two model variants:

| Model | Parameters | Best for |
|-------|-----------|----------|
| **LSTM-RBM** | 189K | Complex structure, long phrases |
| **RNN-RBM** | 75K | Faster training, simpler patterns |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./download_data.sh
```

## Usage

```bash
./run.sh
```

```
╔═══════════════════════════════════════╗
║    GameIntoMusic — AI Music Generator  ║
╚═══════════════════════════════════════╝

Device: mps
Model [lstm/rnn]: lstm
Building LSTM-RBM...
Parameters: 189,238

[train/generate/save/load/exit] > train
  Styles:
    jigs: 340 files
    reels_simple_chords: 310 files
    waltzes: 52 files
    ...
Style [all]: waltzes
Number of epochs [200]: 1000
Gibbs steps k [5]: 5
Training on 52 files (k=5).
  Epoch 1/1000 [████████████████████] batch 4/4 (3s, ETA 0s)
  Epoch 1/1000  Cost: 0.0299  PLL: -4.3013  (3.2s)
  ...
  Epoch 1000/1000  Cost: -0.2517  PLL: -2.3854  (0.1s)

[train/generate/save/load/exit] > generate
Output filename [output.mid]: waltz
Length in steps [200]: 200
Temperature (0.5=safe, 1=normal, 2=wild) [1.0]: 0.7
Seed (number or Enter for random):
Saved to output/waltz.mid
```

### Training tips

- **Style matters**: train on one style (e.g. `waltzes`) not `all` — mixed styles = noise
- **When to stop**: watch PLL — when it stops improving for ~50 epochs, you've hit the plateau
- **Temperature**: 0.5 = conservative/safe, 1.0 = normal, 2.0 = experimental
- **Seed**: same seed = same output (reproducible), empty = random each time

### Playback

```bash
brew install fluid-synth
fluidsynth -ni /opt/homebrew/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2 output/waltz.mid -F output/waltz.wav
open output/waltz.wav
```

## Structure

```
gim.py              ← CLI: train / generate / save / load
model.py            ← LSTM-RBM & RNN-RBM in PyTorch
midi/               ← MIDI file parser/writer
data/               ← training MIDI files (downloaded via script)
checkpoints/        ← saved model weights
output/             ← generated MIDI files
theano-original/    ← original Theano implementation (2016)
```

## References

Based on the work of **Nicolas Boulanger-Lewandowski** (University of Montreal):

- Boulanger-Lewandowski, N., Bengio, Y., & Vincent, P. (2012). **Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription.** *ICML 2012.*
- RNN-RBM tutorial: http://deeplearning.net/tutorial/rnnrbm.html
- Training data: [Nottingham Music Database](https://abc.sourceforge.net/NMD/)

## Dependencies

- PyTorch (with MPS/CUDA support)
- NumPy
