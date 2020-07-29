python demo.py --config-file ../projects/bjzProject/logs/bjz/R50_512dim_circle_bjz_0618_8x3/config.yaml \
--input \
"/home/liaoxingyu2/lxy/logs/badcase/seed.jpg" \
"/home/liaoxingyu2/lxy/logs/badcase/leaf_0.4883.jpg" \
"/home/liaoxingyu2/lxy/logs/badcase/leaf_0.467.jpg" \
--output logs/R50_256x128_pytorch_feat_output \
--opts MODEL.WEIGHTS ../projects/bjzProject/logs/bjz/R50_512dim_circle_bjz_0618_8x32/model_final.pth \
MODEL.DEVICE "cuda: 2"

