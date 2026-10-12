#!/bin/sh
# Download the TinyStories V2 (GPT-4) validation file and the first 150 MB of its training file
# (enough for the 20,000 training stories used here) into tools/words/work/dl/.
set -e
cd "$(dirname "$0")"
mkdir -p work/dl
B="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
curl -L -o work/dl/valid.txt "$B/TinyStoriesV2-GPT4-valid.txt"
curl -L -r 0-157286399 -o work/dl/train_head.txt "$B/TinyStoriesV2-GPT4-train.txt"
