# Official Implementation of "Idempotent Learned Image Compression with Right-Inverse"

## preparation

* install CompressAI https://github.com/InterDigitalInc/CompressAI
* install GeoTorch https://github.com/lezcano/geotorch

## training

1. edit the config file (`workdir/psingle.q3.py` for example)
    * change `dataset` to your training dataset dir
    * change `q` to your preferred qaulity level (1-8 supported)

2. run `python LIC_train.py workdir/psingle.q3.py`

## evaluation
 * run `python LIC_eval.py -d kodak24/ -c psingle -p workdir/psingle.q3/checkpoint_best.pt`