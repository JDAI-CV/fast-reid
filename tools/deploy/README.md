# Deployment

This directory contains:

1. A script that converts a fastreid model to Caffe format.

2. An exmpale that loads a R50 baseline model in Caffe and run inference.

## Tutorial

This is a tiny example for converting fastreid-baseline in `meta_arch` to Caffe model, if you want to convert more complex architecture, you need to customize more things.

1. Change `preprocess_image` in `fastreid/modeling/meta_arch/baseline.py` as below

    ```python
    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        # images = [x["images"] for x in batched_inputs]
        # images = batched_inputs["images"]
        images = batched_inputs
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images
    ```

2. Run `caffe_export.py` to get the converted Caffe model,

    ```bash
    python caffe_export.py --config-file "/export/home/lxy/fast-reid/logs/market1501/bagtricks_R50/config.yaml" --name "baseline_R50" --output "logs/caffe_model" --opts MODEL.WEIGHTS "/export/home/lxy/fast-reid/logs/market1501/bagtricks_R50/model_final.pth"
    ```

    then you can check the Caffe model and prototxt in `logs/caffe_model`.

3. Change `prototxt` following next three steps:

   1) Edit `max_pooling` in `baseline_R50.prototxt` like this

        ```prototxt
        layer {
            name: "max_pool1"
            type: "Pooling"
            bottom: "relu_blob1"
            top: "max_pool_blob1"
            pooling_param {
                pool: MAX
                kernel_size: 3
                stride: 2
                pad: 0 # 1
                # ceil_mode: false
            }
        }
        ```

   2) Add `avg_pooling` right place in `baseline_R50.prototxt`

        ```prototxt
        layer {
            name: "avgpool1"
            type: "Pooling"
            bottom: "relu_blob49"
            top: "avgpool_blob1"
            pooling_param {
                pool: AVE
                global_pooling: true
            }
        }
        ```

    3) Change the last layer `top` name to `output`

        ```prototxt
        layer {
            name: "bn_scale54"
            type: "Scale"
            bottom: "batch_norm_blob54"
            top: "output" # bn_norm_blob54
            scale_param {
                bias_term: true
            }
        }
        ```

4. (optional) You can open [Netscope](https://ethereon.github.io/netscope/quickstart.html), then enter you network `prototxt` to visualize the network.

5. Run `caffe_inference.py` to save Caffe model features with input images

   ```bash
    python caffe_inference.py --model-def "logs/caffe_model/baseline_R50.prototxt" \
    --model-weights "logs/caffe_model/baseline_R50.caffemodel" \
    --input \
    '/export/home/DATA/Market-1501-v15.09.15/bounding_box_test/1182_c5s3_015240_04.jpg' \
    '/export/home/DATA/Market-1501-v15.09.15/bounding_box_test/1182_c6s3_038217_01.jpg' \
    '/export/home/DATA/Market-1501-v15.09.15/bounding_box_test/1183_c5s3_006943_05.jpg' \
    --output "caffe_R34_output"
   ```

6. Run `demo/demo.py` to get fastreid model features with the same input images, then compute the cosine similarity of difference model features to verify if you convert Caffe model successfully.

## Acknowledgements

Thank to [CPFLAME](https://github.com/CPFLAME), [gcong18](https://github.com/gcong18), [YuxiangJohn](https://github.com/YuxiangJohn) and [wiggin66](https://github.com/wiggin66) at JDAI Model Acceleration Group for help in PyTorch to Caffe model converting.
