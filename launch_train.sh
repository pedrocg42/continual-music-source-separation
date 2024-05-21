export DATASETS_PATH="/home/pcordoba/data/audio/"

madre-experiment experiments/bs-roformer/bs-roformer_vocals.yaml
madre-experiment experiments/bs-mel-roformer/bs-mel-roformer_vocals.yaml

madre-experiment experiments/bs-roformer/bs-roformer_bass.yaml
madre-experiment experiments/bs-mel-roformer/bs-mel-roformer_bass.yaml

madre-experiment experiments/bs-roformer/bs-roformer_drums.yaml
madre-experiment experiments/bs-mel-roformer/bs-mel-roformer_drums.yaml

madre-experiment experiments/bs-roformer/bs-roformer_others.yaml
madre-experiment experiments/bs-mel-roformer/bs-mel-roformer_others.yaml

sudo shutdown
