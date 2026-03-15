"""
Real-time MIDI player for GiM.
Generates notes step by step and plays them live via fluidsynth subprocess.
No manual MIDI routing needed — just works.
"""
import time
import subprocess
import os
import torch
import numpy as np
from midi.utils import midiwrite


# Find soundfont
SF2_PATHS = [
    '/opt/homebrew/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2',
    '/usr/share/sounds/sf2/FluidR3_GM.sf2',
    '/usr/share/soundfonts/default.sf2',
]


def find_soundfont():
    for p in SF2_PATHS:
        if os.path.exists(p):
            return p
    return None


@torch.no_grad()
def play_realtime(model, length=200, k=25, temperature=1.0, seed=None, dt=0.3, r=(21, 109)):
    """Generate music and play in real-time by writing short MIDI chunks."""

    sf2 = find_soundfont()
    if not sf2:
        print('  No soundfont found. Install fluidsynth: brew install fluid-synth')
        return

    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    model.eval()

    h = model._init_gen_state(device)
    v_t = torch.bernoulli(torch.ones(1, model.n_visible, device=device) * 0.1)

    print(f'  Generating {length} steps (temp={temperature})...')

    # Generate full piano roll first
    piano_roll = []
    for step in range(length):
        h_rec = model._get_hidden(h)
        bv_t = model.bv + model.Wyv(h_rec).squeeze(0)
        bh_t = model.bh + model.Wyh(h_rec).squeeze(0)

        for _ in range(k):
            mean_h = torch.sigmoid((v_t @ model.W + bh_t) / temperature)
            h_rbm = torch.bernoulli(mean_h)
            mean_v = torch.sigmoid((h_rbm @ model.W.t() + bv_t) / temperature)
            v_t = torch.bernoulli(mean_v)

        piano_roll.append(v_t.squeeze(0).detach().cpu().numpy())

        if hasattr(model, 'lstm_cell'):
            h = model.lstm_cell(v_t, h)
        else:
            h = model.rnn_cell(v_t, h)

        # Progress
        active = int(piano_roll[-1].sum())
        bar = ''.join(['█' if piano_roll[-1][i] > 0.5 else '░' for i in range(0, model.n_visible, 2)])
        print(f'\r  Generating [{bar}] {step+1}/{length} ({active} notes)', end='', flush=True)

    model.train()
    roll = np.stack(piano_roll)
    print(f'\n  Generated {roll.shape[0]} steps, {int(roll.sum())} total note events.')

    # Write temp MIDI
    tmp_mid = '/tmp/gim_live.mid'
    tmp_wav = '/tmp/gim_live.wav'
    midiwrite(tmp_mid, roll, r, dt)

    # Convert to WAV and play
    print('  Converting to audio...')
    subprocess.run(
        ['fluidsynth', '-ni', sf2, tmp_mid, '-F', tmp_wav, '-r', '44100'],
        capture_output=True)

    print('  Playing...\n')

    # Visualize while playing
    proc = subprocess.Popen(['afplay', tmp_wav])

    try:
        for step, frame in enumerate(piano_roll):
            active = int(frame.sum())
            bar = ''.join(['█' if frame[i] > 0.5 else '░' for i in range(0, model.n_visible, 2)])
            notes = [i + r[0] for i in range(len(frame)) if frame[i] > 0.5]
            note_names = [['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][n % 12] + str(n // 12 - 1) for n in notes]
            print(f'\r  [{bar}] {" ".join(note_names):40s}', end='', flush=True)
            time.sleep(dt)
            if proc.poll() is not None:
                break
    except KeyboardInterrupt:
        proc.kill()
        print('\n  Stopped.')
        return

    proc.wait()
    print('\n  Done.')
