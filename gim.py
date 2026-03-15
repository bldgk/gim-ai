"""
GameIntoMusic (GiM) — AI Music Generator

Trains an LSTM-RBM on MIDI files and generates new music.
PyTorch implementation.
"""
import glob
import os
import sys
import numpy as np
import torch
from midi.utils import midiread, midiwrite
from model import LstmRbm, RnnRbm, RnnDbn, LstmDbn
from player import play_realtime

CHECKPOINTS_DIR = 'checkpoints'
OUTPUT_DIR = 'output'
DEFAULT_WEIGHTS = 'weights.pt'
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(text, filename='log.txt'):
    with open(filename, 'a') as f:
        f.write(text + '\n')


def train(model, files, r=(21, 109), dt=0.3, seq_batch=16, num_epochs=200, lr=0.001, gibbs_k=5):
    import time
    device = next(model.parameters()).device

    print('Loading dataset...')
    dataset = [torch.from_numpy(midiread(f, r, dt).piano_roll.astype(np.float32)).to(device)
               for f in files]
    print(f'Loaded {len(dataset)} sequences to {device}.')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_seqs = len(dataset)
    n_batches = (total_seqs + seq_batch - 1) // seq_batch

    print(f'Training for {num_epochs} epochs ({total_seqs} seqs, {n_batches} batches of {seq_batch})...\n')
    try:
        for epoch in range(num_epochs):
            np.random.shuffle(dataset)
            costs, monitors = [], []
            epoch_start = time.time()

            for b in range(n_batches):
                batch = dataset[b * seq_batch : (b + 1) * seq_batch]
                batch = [s for s in batch if len(s) >= 2]
                if not batch:
                    continue

                optimizer.zero_grad()
                cost, monitor = model.forward_batch(batch, k=gibbs_k)
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                costs.append(cost.item())
                monitors.append(monitor.item())

                # Progress
                elapsed = time.time() - epoch_start
                pct = (b + 1) / n_batches
                eta = elapsed / pct * (1 - pct) if pct > 0 else 0
                bar = '█' * int(pct * 20) + '░' * (20 - int(pct * 20))
                print(f'\r  Epoch {epoch+1}/{num_epochs} [{bar}] batch {b+1}/{n_batches} '
                      f'({elapsed:.0f}s, ETA {eta:.0f}s)', end='', flush=True)

            elapsed = time.time() - epoch_start
            mean_cost = np.mean(costs)
            mean_mon = np.mean(monitors)
            print(f'\r  Epoch {epoch+1}/{num_epochs}  Cost: {mean_cost:.4f}  PLL: {mean_mon:.4f}  ({elapsed:.1f}s)          ')
            log(f'Epoch {epoch+1}: Cost={mean_cost:.4f} PLL={mean_mon:.4f}')

            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                cp = os.path.join(CHECKPOINTS_DIR, f'checkpoint_epoch{epoch+1}.pt')
                torch.save(model.state_dict(), cp)
                print(f'  Saved {cp}')

            sys.stdout.flush()

    except KeyboardInterrupt:
        print('\nInterrupted.')
        cp = os.path.join(CHECKPOINTS_DIR, 'checkpoint_interrupted.pt')
        torch.save(model.state_dict(), cp)
        print(f'Saved {cp}')

    print('Training finished.')


def generate(model, filename, r=(21, 109), dt=0.3, length=200, temperature=1.0, seed=None):
    seed_str = f', seed={seed}' if seed else ', random seed'
    print(f'Generating {filename} ({length} steps, temp={temperature}{seed_str})...')
    piano_roll = model.generate(length=length, temperature=temperature, seed=seed)
    midiwrite(filename, piano_roll, r, dt)
    print(f'Saved to {filename}')


def save_weights(model, filename=DEFAULT_WEIGHTS):
    torch.save(model.state_dict(), filename)
    n = sum(p.numel() for p in model.parameters())
    print(f'Weights saved to {filename} ({n:,} parameters)')


def load_weights(model, filename=DEFAULT_WEIGHTS):
    if not os.path.exists(filename):
        print(f'File not found: {filename}')
        return False
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(filename, map_location=device, weights_only=True))
    print(f'Weights loaded from {filename}')
    return True


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


if __name__ == '__main__':
    print('╔═══════════════════════════════════════╗')
    print('║    GameIntoMusic — AI Music Generator  ║')
    print('╚═══════════════════════════════════════╝\n')

    r = (21, 109)  # MIDI note range (full piano = 88 keys)
    dt = 0.3

    device = get_device()
    print(f'Device: {device}')

    models = {
        'rnn':      ('RNN-RBM',  RnnRbm),
        'lstm':     ('LSTM-RBM', LstmRbm),
        'rnn-dbn':  ('RNN-DBN',  RnnDbn),
        'lstm-dbn': ('LSTM-DBN', LstmDbn),
    }
    print('  Models: rnn, lstm, rnn-dbn, lstm-dbn')
    choice = input('Model [lstm]: ').strip().lower() or 'lstm'
    model_name, ModelClass = models.get(choice, ('LSTM-RBM', LstmRbm))
    model_name = 'LSTM-RBM' if ModelClass == LstmRbm else 'RNN-RBM'

    print(f'Building {model_name}...')
    model = ModelClass(n_visible=r[1] - r[0], n_hidden=150, n_recurrent=100).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n:,}\n')

    DEFAULT_WEIGHTS = os.path.join(CHECKPOINTS_DIR, f'weights_{choice}.pt')
    data_path = os.path.join(os.path.dirname(__file__) or '.', 'data', 'Nottingham', 'train', '*.mid')

    # Auto-load
    if os.path.exists(DEFAULT_WEIGHTS):
        load_weights(model, DEFAULT_WEIGHTS)

    while True:
        cmd = input('\n[train/generate/play/save/load/exit] > ').strip().lower()

        if cmd == 'train':
            # Show available styles
            all_files = glob.glob(data_path)
            if not all_files:
                print(f'No MIDI files found at {data_path}')
                continue
            import re
            styles = {}
            for f in all_files:
                name = re.sub(r'[\d_]+\.mid$', '', os.path.basename(f))
                styles.setdefault(name, []).append(f)
            print('  Styles:')
            for s, fs in sorted(styles.items(), key=lambda x: -len(x[1])):
                print(f'    {s}: {len(fs)} files')
            print(f'    all: {len(all_files)} files')

            style = input('Style [all]: ').strip().lower() or 'all'
            if style == 'all':
                files = all_files
            else:
                files = [f for f in all_files if style in os.path.basename(f).lower()]
                if not files:
                    print(f'No files matching "{style}"')
                    continue

            try:
                epochs = int(input('Number of epochs [200]: ').strip() or '200')
            except ValueError:
                print('Invalid number.')
                continue
            try:
                gibbs_k = int(input('Gibbs steps k [5]: ').strip() or '5')
            except ValueError:
                gibbs_k = 5
            print(f'Training on {len(files)} files (k={gibbs_k}).')
            train(model, files, r=r, dt=dt, num_epochs=epochs, gibbs_k=gibbs_k)

        elif cmd == 'generate':
            filename = (input('Output filename [output.mid]: ').strip() or 'output')
            if not filename.endswith('.mid'):
                filename += '.mid'
            filename = os.path.join(OUTPUT_DIR, filename)
            try:
                length = int(input('Length in steps [200]: ').strip() or '200')
            except ValueError:
                length = 200
            try:
                temp = float(input('Temperature (0.5=safe, 1=normal, 2=wild) [1.0]: ').strip() or '1.0')
            except ValueError:
                temp = 1.0
            seed_str = input('Seed (number or Enter for random): ').strip()
            seed = int(seed_str) if seed_str else None
            generate(model, filename, r=r, dt=dt, length=length, temperature=temp, seed=seed)

        elif cmd == 'save':
            filename = input(f'Filename [{DEFAULT_WEIGHTS}]: ').strip() or DEFAULT_WEIGHTS
            save_weights(model, filename)

        elif cmd == 'play':
            print('  First run in another terminal:')
            print('  fluidsynth -a coreaudio -m coremidi /opt/homebrew/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2\n')
            try:
                length = int(input('Length in steps [200]: ').strip() or '200')
            except ValueError:
                length = 200
            try:
                temp = float(input('Temperature [1.0]: ').strip() or '1.0')
            except ValueError:
                temp = 1.0
            seed_str = input('Seed (Enter for random): ').strip()
            seed = int(seed_str) if seed_str else None
            play_realtime(model, length=length, temperature=temp, seed=seed, dt=dt, r=r)

        elif cmd == 'load':
            filename = input(f'Filename [{DEFAULT_WEIGHTS}]: ').strip() or DEFAULT_WEIGHTS
            load_weights(model, filename)

        elif cmd == 'exit':
            break
