#!/bin/bash
curl https://s3.amazonaws.com/fast-ai-imageclas/imagewang-160.tgz -o data/imagewang
tar -xf data/imagewang -C data/
rm data/imagewang

