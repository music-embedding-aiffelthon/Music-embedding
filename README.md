# Music embedding with Semi-supervised model

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)


Jax/Flax implementation of BYOL, SimCLR, Convolutional Autoencoder.

* With this Jax/Flax implemented Semi-supervised model, you can generate general representation of music.
* We achieve decent result on the genre classification for kakao arena dataset with Semi-supervised model.
* You can use accuarcy for 3 pretrain method with this repository.

## Quick start
**Step 1.**

Prepare [kakao arena dataset](https://arena.kakao.com/c/8) and make dataset directory like this.
```
$ ls dataset
arena_mel   song_meta.json
```

**Step 2.**

Clone repository and install requirements.

```
git clone https://github.com/music-embedding-aiffelthon/Music-embedding && cd Music-embedding
pip install -r requirements.txt
```

**Step 3.**

Setup [config.yaml](https://github.com/music-embedding-aiffelthon/Music-embedding/blob/master/config/config.yaml) file to specify dataset folder and model usage.


**Step 4.**

Run main.py file.
```
python3 main.py
```

## Dataset
* We pretrained Semi-supervised model with [kakao arena dataset](https://arena.kakao.com/c/8), which contains 700,000 mel-spetrograms. It has 48 bins, 1876 time length. 
* We adjusted 30 to 14 distinct acoustic distinctions for labels to evaluate pretrain model.

## Convolutional Autoencoder 
![image](https://user-images.githubusercontent.com/85149409/191931651-159657ab-b941-4335-9d1e-931c520ad219.png)


* As a first method, we use latent vector of Autoencoder to create general representation.
* You can use Autoencoder model to change [config.yaml](https://github.com/music-embedding-aiffelthon/Music-embedding/blob/master/config/config.yaml)
* **Add result**

## SimCLR 
![1](https://user-images.githubusercontent.com/85149409/191939638-e73076d9-3423-4a20-8a8c-c6cf2d6ba693.png)

* As a second method, we use SimCLR to create general representation.
* For the augmentation method, we use [BYOL-A](https://github.com/nttcslab/byol-a) augmentation method to create positive pair. 
* We pretrained with batch size 128, TPU v3-8 15 hours.

## BYOL
![191939270-f56defba-3060-4009-bca9-38707b007db0](https://user-images.githubusercontent.com/85149409/191939763-052f361f-55fd-4f12-ae79-b0c623577593.png)

* As a third method, we use BYOL to create general representation.
* We pretrained with batch size 64, TPU v3-8 15 hours.

## Result
| Encoder / Model | Batch-size / pretrain epochs / linear evaluation epochs |  ROC-AUC |
|-------------|-------------|-------------|-------------|
| SampleCNN (supervised) | 64 / - / 200 | (Add result later) |
| Autoencoder (semi-supervised) | 48 / 10 / 200 | (Add result later) |
| SimCLR | 128 / 10 / 200 | 49 |
| BYOL | 64 / 10 / 200 | 50 (Add result later) |
