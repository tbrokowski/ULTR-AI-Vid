#!/usr/bin/env bash
set -e

# Base data folder (change later to your RCP scratch path if needed)
mkdir -p ~/ultra-data
cd ~/ultra-data

git clone https://github.com/NinaWie/COVID-BLUES.git
git clone https://github.com/nrc-cnrc/COVID-US.git
git clone https://github.com/cossiomanuel/covid19_pocus_ultrasound.git
