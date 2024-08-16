### M_lesion
python validate.py \
--data_dir ../data/volumns/ --val_anno_file ../labels/test.txt \
--model uniformer_small_IL -j 0 -b 12 --num-classes 6 \
--checkpoint ./ckpts/outputs/best_checkpoint.pth.tar
