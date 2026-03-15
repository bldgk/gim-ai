#!/bin/bash
# Download Nottingham Music Database (MIDI files for training)
# Source: https://abc.sourceforge.net/NMD/

mkdir -p data/Nottingham/train
cd data/Nottingham/train

echo "Downloading Nottingham Music Database..."
curl -L "https://github.com/jukedeck/nottingham-dataset/archive/refs/heads/master.zip" -o nmd.zip
unzip -j nmd.zip "*/MIDI/melody/*" -d . 2>/dev/null
unzip -j nmd.zip "*/MIDI/train/*" -d . 2>/dev/null
rm -f nmd.zip

COUNT=$(ls *.mid 2>/dev/null | wc -l)
echo "Downloaded $COUNT MIDI files to data/Nottingham/train/"
