# FastReID Demo

We provide a command line tool to run a simple demo of builtin models.

You can run this command to get cosine similarites between different images 

```bash
cd demo/
sh run_demo.sh
```

What is more, you can use this command to make thing more interesting
```bash
export CUDA_VISIBLE_DEVICES=0
python3 demo/visualize_result.py --config-file ./configs/VeRi/sbs_R50-ibn.yml --actmap --dataset-name 'VeRi' --output logs/veri/sbs_R50-ibn/eval --opts MODEL.WEIGHTS logs/veri/sbs_R50-ibn/model_best.pth
```
![4](https://user-images.githubusercontent.com/77771760/123026335-90dd8780-d40e-11eb-8a8d-1683dc19a05a.jpg)
where `--actmap` is used to add activation map upon the original image.
