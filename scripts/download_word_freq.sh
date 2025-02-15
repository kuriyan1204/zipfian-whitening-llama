# download json via wget and save it to data folder
# create data folder first if it doesn't exist data/llama2 folder first if it doesn't exist
mkdir -p data/word_freq/llama2
wget -O data/word_freq/llama2/dolma.json https://raw.githubusercontent.com/kuriyan1204/unigram-prob-llama/refs/heads/main/llama2_token_id_to_frequency_dolma.json