python ./main/train_lesion.py --workers 24 \
--data_dir data/volumns \
--train_anno_file labels/train.txt --val_anno_file labels/valid.txt \
--batch-size=16  --model uniformer_small_IL --lr 1e-4 --warmup-epochs 5 \
--epochs 300 --checkpoint-hist 5 --output ./ckpts/outputs/ \
--num-classes=6 --num-hier-classes 29 --report-metrics f1 recall \
--train_transform_list random_crop z_flip x_flip y_flip rotation channel_cutout --cutcnum 2 --cutcprob 0.5 --cutcmode zeros
