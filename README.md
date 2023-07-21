# BT4222 sentiment analysis

This code provide 3 different kinds of embedding methods+ pure CNN baseline (Acc 55.03), pure LSTM (Acc 56.33), pure MLP (Acc 52.11). We also provide how to increase the model accuracy by playing with model structure. We provide CNN+LSTM (Acc 56.58), voting rule with CNN and LSTM (Acc 56.42) and CNN+LSTM+FC+Attention layer (Acc 55.04 May be the problem of overfitting).

All models structure and design methods are from state-of-art papers. We provide the paper name and link withion code.

# What to do next 

I will further add more detailed comments/figures.


# Package
Package                  Version
------------------------ ----------
blis                     0.7.9
catalogue                2.0.8
certifi                  2023.5.7
charset-normalizer       3.2.0
click                    8.1.4
cmake                    3.26.4
confection               0.1.0
contourpy                1.1.0
cycler                   0.11.0
cymem                    2.0.7
en-core-web-sm           3.6.0
filelock                 3.12.2
fonttools                4.40.0
gensim                   4.3.1
idna                     3.4
Jinja2                   3.1.2
joblib                   1.3.1
kiwisolver               1.4.4
langcodes                3.3.0
lit                      16.0.6
MarkupSafe               2.1.3
matplotlib               3.7.2
mpmath                   1.3.0
murmurhash               1.0.9
networkx                 3.1
nltk                     3.8.1
numpy                    1.25.1
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-cupti-cu11   11.7.101
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cudnn-cu11        8.5.0.96
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.2.10.91
nvidia-cusolver-cu11     11.4.0.1
nvidia-cusparse-cu11     11.7.4.91
nvidia-nccl-cu11         2.14.3
nvidia-nvtx-cu11         11.7.91
packaging                23.1
pandas                   2.0.3
pathy                    0.10.2
Pillow                   10.0.0
pip                      22.0.2
preshed                  3.0.8
pydantic                 1.10.11
pyparsing                3.0.9
python-dateutil          2.8.2
pytz                     2023.3
regex                    2023.6.3
requests                 2.31.0
scikit-learn             1.3.0
scipy                    1.11.1
seaborn                  0.12.2
setuptools               59.6.0
six                      1.16.0
sklearn                  0.0.post5
smart-open               6.3.0
spacy                    3.6.0
spacy-legacy             3.0.12
spacy-loggers            1.0.4
srsly                    2.4.6
sympy                    1.12
thinc                    8.1.10
threadpoolctl            3.1.0
torch                    2.0.1
torchdata                0.6.1
torchtext                0.15.2
torchvision              0.15.2
tqdm                     4.65.0
triton                   2.0.0
typer                    0.9.0
typing_extensions        4.7.1
tzdata                   2023.3
urllib3                  2.0.3
wasabi                   1.1.2
wheel                    0.37.1
