"""
GameIntoMusic (GiM) — AI Music Generator

Interactive CLI for training LSTM-RBM models on MIDI datasets
and generating new music compositions.

Uses Theano for GPU-accelerated training of a deep generative model
that learns temporal patterns in piano-roll representations of music.

Original: Bachelor's diploma project, KPI (2016)
"""
import glob
import os
import sys
import numpy
from midi.utils import midiread, midiwrite
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import LstmRbm

numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False

DEFAULT_WEIGHTS = 'weights.npz'


def log(text, filename='log.txt'):
    with open(filename, 'a') as f:
        f.write(text + '\n')


class Model(object):
    """Base model class with train/generate/save/load interface."""

    def __init__(self, h_power=150, r_power=150, v_power=88, lr=0.001, dt=0.3):
        self.dt = dt
        self.r = (21, 109)  # MIDI note range (full piano)
        (v, v_sample, nll, error, self.params, updates_train, v_t,
         updates_generate) = self.build(v_power, h_power, r_power)

        gradient = T.grad(nll, self.params, consider_constant=[v_sample])
        updates_train.update(((p, p - lr * g) for p, g in zip(self.params, gradient)))
        self.train_function = theano.function([v], (error, nll), updates=updates_train)
        self.generate_function = theano.function([], v_t, updates=updates_generate)

    def train(self, files, batch_size, num_epochs, autosave=True):
        print('Loading dataset...')
        dataset = [midiread(f, self.r, self.dt).piano_roll.astype(theano.config.floatX)
                   for f in files]
        print(f'Training on {len(dataset)} sequences for {num_epochs} epochs...')

        try:
            for epoch in range(num_epochs):
                numpy.random.shuffle(dataset)
                errors, nlls = [], []
                for s, sequence in enumerate(dataset):
                    for i in range(0, len(sequence), batch_size):
                        err, nll = self.train_function(sequence[i:i + batch_size])
                        errors.append(err)
                        nlls.append(nll)

                mean_err = numpy.mean(errors)
                mean_nll = numpy.mean(nlls)
                print(f'Epoch {epoch+1}/{num_epochs}  NLL: {mean_err:.4f}  Error: {mean_nll:.4f}')
                log(f'Epoch {epoch+1}: NLL={mean_err:.4f} Error={mean_nll:.4f}')
                sys.stdout.flush()

                # Autosave every 10 epochs
                if autosave and (epoch + 1) % 10 == 0:
                    checkpoint = f'checkpoint_epoch{epoch+1}.npz'
                    self.save_weights(checkpoint)
                    print(f'  Checkpoint saved: {checkpoint}')

        except KeyboardInterrupt:
            print('\nInterrupted by user.')
            if autosave:
                self.save_weights('checkpoint_interrupted.npz')
                print('Weights saved to checkpoint_interrupted.npz')

        print('Training finished.')

    def generate(self, filename):
        print(f'Generating {filename}...')
        piano_roll = self.generate_function()
        midiwrite(filename, piano_roll, self.r, self.dt)
        print(f'Saved to {filename}')

    def save_weights(self, filename=DEFAULT_WEIGHTS):
        """Save all model parameters to a .npz file."""
        param_values = {f'param_{i}': p.get_value() for i, p in enumerate(self.params)}
        numpy.savez(filename, **param_values)
        print(f'Weights saved to {filename} ({len(self.params)} parameters)')

    def load_weights(self, filename=DEFAULT_WEIGHTS):
        """Load model parameters from a .npz file."""
        if not os.path.exists(filename):
            print(f'File not found: {filename}')
            return False
        data = numpy.load(filename)
        for i, p in enumerate(self.params):
            key = f'param_{i}'
            if key in data:
                p.set_value(data[key])
            else:
                print(f'Warning: {key} not found in {filename}')
                return False
        print(f'Weights loaded from {filename} ({len(self.params)} parameters)')
        return True


if __name__ == '__main__':
    print('╔═══════════════════════════════════════╗')
    print('║    GameIntoMusic — AI Music Generator  ║')
    print('╚═══════════════════════════════════════╝\n')

    print('Building LSTM-RBM model...')
    model = LstmRbm.LstmRbm()

    data_path = os.path.join(os.path.dirname(__file__) or '.', 'data', 'Nottingham', 'train', '*.mid')

    # Auto-load weights if available
    if os.path.exists(DEFAULT_WEIGHTS):
        model.load_weights()
        print('Ready to generate (or train more).')

    while True:
        cmd = input('\n[train/generate/save/load/exit] > ').strip().lower()

        if cmd == 'train':
            try:
                epochs = int(input('Number of epochs: '))
            except ValueError:
                print('Invalid number.')
                continue
            files = glob.glob(data_path)
            if not files:
                print(f'No MIDI files found at {data_path}')
                continue
            print(f'Found {len(files)} training files.')
            model.train(files, batch_size=100, num_epochs=epochs)

        elif cmd == 'generate':
            filename = input('Output filename (without .mid): ').strip() + '.mid'
            model.generate(filename)

        elif cmd == 'save':
            filename = input(f'Filename [{DEFAULT_WEIGHTS}]: ').strip()
            model.save_weights(filename or DEFAULT_WEIGHTS)

        elif cmd == 'load':
            filename = input(f'Filename [{DEFAULT_WEIGHTS}]: ').strip()
            model.load_weights(filename or DEFAULT_WEIGHTS)

        elif cmd == 'exit':
            break
