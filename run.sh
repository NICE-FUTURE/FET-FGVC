### training

python ./train.py --name pet-swin-base-224 --config configs/cub/gpu.yaml --arch swin-base --sample_classes 2 --sample_images 10 --img_size 224 --distill --gpus 0 --lr 0.00001 && \


### fine-tuning

python ./train.py --name pet-swin-base-448 --config configs/cub/gpu.yaml --arch swin-base --sample_classes 2 --sample_images 10 --img_size 448 --gpus 0 --lr 0.00001 --finetune --weights_dir ./middle/models/xxx --epochs 40 && \

echo "done."