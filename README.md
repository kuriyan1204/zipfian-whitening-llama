# zipfian-whitening-llama

Minimalistic implementation of [Zipfian Whitening](https://openreview.net/forum?id=pASJxzMJb7) for static embeddings of large language models. 

## Usage

### Installation
```bash
uv sync
```

### Use pre-trained model
[WIP] As PoC, we have five pre-trained models available now on huggingface hub:
- Baseline: `hkurita/llama-2-70b-embedding_mean`
  - Original token embedding of llama2-70b model without any modification + mean pooling.
- Uniform Centered: `hkurita/llama-2-70b-embedding_mean-uniform-centered`
  - Uniform centering + mean pooling.
- Uniform Whitened: `hkurita/llama-2-70b-embedding_mean-uniform-whitened`
  - Uniform whitening + mean pooling.
- Zipfian Centered: `hkurita/llama-2-70b-embedding_mean-zipfian-centered`
  - Zipfian centering + mean pooling.
- **Zipfian Whitened**: `hkurita/llama-2-70b-embedding_mean-zipfian-whitened`
  - Zipfian whitening + mean pooling. Supposed to be the best performing model.

For the Zipfian Centered/Whitened models, we have used the unigram frequency of [dolma](https://github.com/allenai/dolma) [4] which is calculated from [Infini-gram](https://infini-gram.io)[5].

```python
from sentence_transformers import SentenceTransformer

model_name = "hkurita/llama-2-70b-embedding_mean-zipfian-whitened"
model = SentenceTransformer(model_name)

# encode texts
texts = ["Hello, my dog is cute", "Hello, my cat is cute"]
model.encode(texts)
```

### Evaluation
[TBW]

### Add your own model
[TBW]

## Citation
This work is built upon following amazing works:  
[1] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERT networks. In *Proc. of EMNLP-IJCNLP*.  
[2] Niklas Muennighoff, Nouamane Tazi, Loic Magne, and
Nils Reimers. 2023. MTEB: Massive text embedding
benchmark. In *Proc of EACL*.  
[3] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov,
Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen,
Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj
Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez,
Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril,
Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor
Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan
Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams,
Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan
Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and
fine-tuned chat models. *arXiv preprint arXiv:2307.09288*  
[4] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Evan Walsh, Luke Zettlemoyer, Noah Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge and Kyle Lo. 2024. Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. In *Proc. of ACL*.  
[5] Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin Choi, and Hannaneh Hajishirzi. 2024. Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens. *arXiv preprint arXiv:2401.17377*

If you use this codebase or find this work helpful, please cite:
```
@inproceedings{
yokoi2024zipfian,
    title={Zipfian Whitening},
    author={Sho Yokoi and Han Bao and Hiroto Kurita and Hidetoshi Shimodaira},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=pASJxzMJb7}
}
```

```
@misc{
  kurita2025minimalistic-zipf,
  author = {Hiroto Kurita},
  title = {Zipfian Whitening Llama: Minimalistic implementation of Zipfian Whitening for static embeddings of large language models}, 
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kuriyan1204/zipfian-whitening-llama}}
}
```