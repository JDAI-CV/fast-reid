# FastReID Demo

We provide a command line tool to run a simple demo of builtin models.

You can run this command to get rank visualization results by cosine similarites between different images.

```shell script
python3 demo/visualize_result.py --config-file logs/dukemtmc/mgn_R50-ibn/config.yaml \
--parallel --vis-label --dataset-name 'DukeMTMC' --output logs/mgn_duke_vis \
--opts MODEL.WEIGHTS logs/dukemtmc/mgn_R50-ibn/model_final.pth
```

You can also run this command to extract image features.

```shell script
python3 demo/demo.py --config-file logs/dukemtmc/sbs_R50/config.yaml \
--parallel --input tools/deploy/test_data/*.jpg --output sbs_R50_feat \
--opts MODEL.WEIGHTS logs/dukemtmc/sbs_R50/model_final.pth
```