# A Neural Algorithm of Artistic Style

|Original|Style Transfer|
|---|---|
|![광화문1](/assets/gwang1_original.jpg)|![광화문1+gogh](/assets/gwang1_style.jpg)|
|![광화문2](/assets/gwang2_original.jpg)|![광화문2+gogh](/assets/gwang2_style.jpg)|
|![eiffel](/assets/eiffel_original.jpg)|![eiffel+oriental](/assets/eiffel_style.jpg)|

This repository implements the paper: **[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)**.

## Features

* Employed ***TensorFlow 2*** with performance optimization
* Simple structure
* Easy to reproduce

## Model Structure

As mentioned in paper, this approache make use of the VGG network. I used VGG19 structure and weight from built-in tensorflow library. (`tf.keras.applications.VGG19`)

![nn](/assets/vgg19.jpg)

I used `block4_conv2` as a content layer and `['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']` as style layers. And average pooling is used instead of max pooling as noted in the paper.

## Requirements

Install packages through `requirements.txt`.

Also, you need trained `VGG19` network unless you already have it. First call `VGG19` will automatically download the network. If you want to download model weight without running whole script, you can do as follow:

```bash
$ python download_vgg.py
```

### GPU Settings

Tensorflow 2 code is optimized for GPU running.

***Default running environment is assumed to be CPU-ONLY. If you want to run this repo on GPU machine, just replace `tensorflow` to `tensorflow-gpu` in package lists.***

## How to install

### `virtualenv`

```bash
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### `venv`

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## How to run

```bash
$ python main.py --help
usage: main.py [-h] [--content_url CONTENT_URL] [--style_url STYLE_URL]
               [--quick] [--train_epochs TRAIN_EPOCHS]
               [--log_interval LOG_INTERVAL]

A Neural Algorithm of Artistic Style

optional arguments:
  -h, --help            show this help message and exit
  --content_url CONTENT_URL
                        Content image url
  --style_url STYLE_URL
                        Style image url
  --quick               Set input image as the content image
  --train_epochs TRAIN_EPOCHS
  --log_interval LOG_INTERVAL
```

You can give custom image url to `content_url` and `style_url` arguemtns.

If you set `quick`, style transfer will start from your content image instead of white noise. This will give you the result much faster than starting from white noise.

## Test

`pytest` is used for testing.

```bash
============================= test session starts ==============================
platform linux -- Python 3.6.9, pytest-5.0.1, py-1.8.0, pluggy-0.13.1
rootdir: /home/jihoon/Documents/A-Neural-Algorithm-of-Artistic-Style
collected 3 items                                                              

test/test_styletf.py ...                                                 [100%]

=============================== warnings summary ===============================
venv/lib/python3.6/site-packages/tensorflow_core/python/pywrap_tensorflow_internal.py:15
  /home/jihoon/Documents/A-Neural-Algorithm-of-Artistic-Style/venv/lib/python3.6/site-packages/tensorflow_core/python/pywrap_tensorflow_internal.py:15: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
    import imp

-- Docs: https://docs.pytest.org/en/latest/warnings.html
==================== 3 passed, 1 warnings in 24.53 seconds =====================
```

## BibTeX

```
@article{DBLP:journals/corr/GatysEB15a,
  author    = {Leon A. Gatys and
               Alexander S. Ecker and
               Matthias Bethge},
  title     = {A Neural Algorithm of Artistic Style},
  journal   = {CoRR},
  volume    = {abs/1508.06576},
  year      = {2015},
  url       = {http://arxiv.org/abs/1508.06576},
  archivePrefix = {arXiv},
  eprint    = {1508.06576},
  timestamp = {Mon, 13 Aug 2018 16:48:03 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/GatysEB15a},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Author
Jihoon Kim ([@jihoonerd](https://github.com/jihoonerd))
