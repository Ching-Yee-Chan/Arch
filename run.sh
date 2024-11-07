export PYTHONPATH="/mnt/users/hccl.local/jkzhao/projects/WavTokenizer":$PYTHONPATH

# python evaluate_hf_models.py --model facebook/wav2vec2-base --device cuda \
#     --max_epochs 200 --verbose --tsv_logging_file results/hf_models.tsv \
#     --n_iters 1 --data_config_file configs/data_config.json \
#     --enabled_datasets ravdess slurp emovo audio_mnist fma_small \
#     magna_tag_a_tune irmas medleydb esc50 us8k fsd50k vivae

python evaluate_wavtokenizer.py --model wavtokenizer --device cuda \
    --max_epochs 10 --verbose --tsv_logging_file results/wavtokneizer.tsv \
    --n_iters 1 --data_config_file configs/data_config.json \
    --enabled_datasets  ravdess slurp emovo audio_mnist fma_small \
    magna_tag_a_tune irmas medleydb esc50 us8k fsd50k vivae