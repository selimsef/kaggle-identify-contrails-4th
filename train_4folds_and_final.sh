DATA_DIR=$1

echo "**********Train Bootstrap UnetEffnetV2L**********"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u   train_fold.py --gpu 0 --workers 4 --fold 0 \
--config configs/bootstrap_v2l.json  --test_every 1  --data-dir $DATA --prefix bootstrap_ --fp16   > logs/0 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -u   train_fold.py --gpu 1 --workers 4 --fold 1 \
--config configs/bootstrap_v2l.json  --test_every 1 --data-dir $DATA --prefix bootstrap_  --fp16  > logs/1 &
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python -u   train_fold.py --gpu 2 --workers 4 --fold 2 \
--config configs/bootstrap_v2l.json  --test_every 1  --data-dir $DATA --prefix bootstrap_ --fp16   > logs/2 &
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python -u   train_fold.py --gpu 3 --workers 4 --fold 3 \
--config configs/bootstrap_v2l.json  --test_every 1  --data-dir $DATA --prefix bootstrap_ --fp16    > logs/3 &
wait
echo "**********Train Bootstrap UnetEffnetV2L finished**********"

echo "**********Train Bootstrap UnetEffnetL2**********"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u   train_fold.py --gpu 0 --workers 4 --fold 0 \
--config configs/bootstrap_l2.json  --test_every 1  --data-dir $DATA --prefix bootstrap_ --fp16   > logs/0 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -u   train_fold.py --gpu 1 --workers 4 --fold 1 \
--config configs/bootstrap_l2.json  --test_every 1 --data-dir $DATA --prefix bootstrap_  --fp16  > logs/1 &
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python -u   train_fold.py --gpu 2 --workers 4 --fold 2 \
--config configs/bootstrap_l2.json  --test_every 1  --data-dir $DATA --prefix bootstrap_ --fp16   > logs/2 &
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python -u   train_fold.py --gpu 3 --workers 4 --fold 3 \
--config configs/bootstrap_l2.json  --test_every 1  --data-dir $DATA --prefix bootstrap_ --fp16    > logs/3 &
wait
echo "**********Train Bootstrap UnetEffnetL2 finished**********"

echo "**********Train Bootstrap UnetMaxvit**********"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u   train_fold.py --gpu 0 --workers 4 --fold 0 \
--config configs/bootstrap_maxvit.json  --test_every 1  --data-dir $DATA --prefix bootstrap_ --fp16   > logs/0 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -u   train_fold.py --gpu 1 --workers 4 --fold 1 \
--config configs/bootstrap_maxvit.json  --test_every 1 --data-dir $DATA --prefix bootstrap_  --fp16  > logs/1 &
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python -u   train_fold.py --gpu 2 --workers 4 --fold 2 \
--config configs/bootstrap_maxvit.json  --test_every 1  --data-dir $DATA --prefix bootstrap_ --fp16   > logs/2 &
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python -u   train_fold.py --gpu 3 --workers 4 --fold 3 \
--config configs/bootstrap_maxvit.json  --test_every 1  --data-dir $DATA --prefix bootstrap_ --fp16    > logs/3 &
wait
echo "**********Train Bootstrap UnetMaxvit finished**********"




echo "**********Predicting pseudo masks**********"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python predict_pseudo_simple.py --gpu 0 --config bootstrap_l2,bootstrap_maxvit,bootstrap_v2l --weights-path weights --data-dir $DATA_DIR --out-dir $DATA_DIR/seg_preds \
--checkpoint bootstrap_TimmUnetPure_tf_efficientnet_l2_ns_0_dice,bootstrap_TimmUnetPure_maxvit_base_tf_512.in21k_ft_in1k_0_dice,bootstrap_TimmUnetPure_tf_efficientnetv2_l_in21k_0_dice --fold 0 > logs/0 &

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python predict_pseudo_simple.py --gpu 1 --config bootstrap_l2,bootstrap_maxvit,bootstrap_v2l --weights-path weights --data-dir $DATA_DIR --out-dir $DATA_DIR/seg_preds \
--checkpoint bootstrap_TimmUnetPure_tf_efficientnet_l2_ns_1_dice,bootstrap_TimmUnetPure_maxvit_base_tf_512.in21k_ft_in1k_1_dice,bootstrap_TimmUnetPure_tf_efficientnetv2_l_in21k_1_dice --fold 1 > logs/1  &

CUDA_VISIBLE_DEVICES=2  PYTHONPATH=. python predict_pseudo_simple.py --gpu 2 --config bootstrap_l2,bootstrap_maxvit,bootstrap_v2l --weights-path weights --data-dir $DATA_DIR --out-dir $DATA_DIR/seg_preds \
--checkpoint bootstrap_TimmUnetPure_tf_efficientnet_l2_ns_2_dice,bootstrap_TimmUnetPure_maxvit_base_tf_512.in21k_ft_in1k_2_dice,bootstrap_TimmUnetPure_tf_efficientnetv2_l_in21k_2_dice --fold 2 > logs/2  &

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python predict_pseudo_simple.py --gpu 3 --config bootstrap_l2,bootstrap_maxvit,bootstrap_v2l --weights-path weights --data-dir $DATA_DIR --out-dir $DATA_DIR/seg_preds \
--checkpoint bootstrap_TimmUnetPure_tf_efficientnet_l2_ns_3_dice,bootstrap_TimmUnetPure_maxvit_base_tf_512.in21k_ft_in1k_3_dice,bootstrap_TimmUnetPure_tf_efficientnetv2_l_in21k_3_dice --fold 3 > logs/3 &

wait
echo "**********Finished predicting**********"

echo "**********Train Round UnetEffnetV2L**********"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u   train_fold.py --gpu 0 --workers 4 --fold 0 \
--config configs/round_v2l.json  --test_every 1  --data-dir $DATA --prefix round_ --fp16   > logs/0 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -u   train_fold.py --gpu 1 --workers 4 --fold 1 \
--config configs/round_v2l.json  --test_every 1 --data-dir $DATA --prefix round_  --fp16  > logs/1 &
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python -u   train_fold.py --gpu 2 --workers 4 --fold 2 \
--config configs/round_v2l.json  --test_every 1  --data-dir $DATA --prefix round_ --fp16   > logs/2 &
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python -u   train_fold.py --gpu 3 --workers 4 --fold 3 \
--config configs/round_v2l.json  --test_every 1  --data-dir $DATA --prefix round_ --fp16    > logs/3 &
wait
echo "**********Train Round UnetEffnetV2L finished**********"

echo "**********Train Round UnetEffnetL2**********"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u   train_fold.py --gpu 0 --workers 4 --fold 0 \
--config configs/round_l2.json  --test_every 1  --data-dir $DATA --prefix round_ --fp16   > logs/0 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -u   train_fold.py --gpu 1 --workers 4 --fold 1 \
--config configs/round_l2.json  --test_every 1 --data-dir $DATA --prefix round_  --fp16  > logs/1 &
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python -u   train_fold.py --gpu 2 --workers 4 --fold 2 \
--config configs/round_l2.json  --test_every 1  --data-dir $DATA --prefix round_ --fp16   > logs/2 &
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python -u   train_fold.py --gpu 3 --workers 4 --fold 3 \
--config configs/round_l2.json  --test_every 1  --data-dir $DATA --prefix round_ --fp16    > logs/3 &
wait
echo "**********Train Round UnetEffnetL2 finished**********"

echo "**********Train Round UnetMaxvit**********"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u   train_fold.py --gpu 0 --workers 4 --fold 0 \
--config configs/round_maxvit.json  --test_every 1  --data-dir $DATA --prefix round_ --fp16   > logs/0 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -u   train_fold.py --gpu 1 --workers 4 --fold 1 \
--config configs/round_maxvit.json  --test_every 1 --data-dir $DATA --prefix round_  --fp16  > logs/1 &
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python -u   train_fold.py --gpu 2 --workers 4 --fold 2 \
--config configs/round_maxvit.json  --test_every 1  --data-dir $DATA --prefix round_ --fp16   > logs/2 &
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python -u   train_fold.py --gpu 3 --workers 4 --fold 3 \
--config configs/round_maxvit.json  --test_every 1  --data-dir $DATA --prefix round_ --fp16    > logs/3 &
wait
echo "**********Train Round UnetMaxvit finished**********"

echo "**********Predicting pseudo masks**********"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python predict_pseudo_simple.py --gpu 0 --config round_l2,round_maxvit,round_v2l --weights-path weights --data-dir $DATA_DIR --out-dir $DATA_DIR/seg_preds \
--checkpoint round_TimmUnetPure_tf_efficientnet_l2_ns_0_dice,round_TimmUnetPure_maxvit_base_tf_512.in21k_ft_in1k_0_dice,round_TimmUnetPure_tf_efficientnetv2_l_in21k_0_dice --fold 0 > logs/0 &

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python predict_pseudo_simple.py --gpu 1 --config round_l2,round_maxvit,round_v2l --weights-path weights --data-dir $DATA_DIR --out-dir $DATA_DIR/seg_preds \
--checkpoint round_TimmUnetPure_tf_efficientnet_l2_ns_1_dice,round_TimmUnetPure_maxvit_base_tf_512.in21k_ft_in1k_1_dice,round_TimmUnetPure_tf_efficientnetv2_l_in21k_1_dice --fold 1 > logs/1  &

CUDA_VISIBLE_DEVICES=2  PYTHONPATH=. python predict_pseudo_simple.py --gpu 2 --config round_l2,round_maxvit,round_v2l --weights-path weights --data-dir $DATA_DIR --out-dir $DATA_DIR/seg_preds \
--checkpoint round_TimmUnetPure_tf_efficientnet_l2_ns_2_dice,round_TimmUnetPure_maxvit_base_tf_512.in21k_ft_in1k_2_dice,round_TimmUnetPure_tf_efficientnetv2_l_in21k_2_dice --fold 2 > logs/2  &

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python predict_pseudo_simple.py --gpu 3 --config round_l2,round_maxvit,round_v2l --weights-path weights --data-dir $DATA_DIR --out-dir $DATA_DIR/seg_preds \
--checkpoint round_TimmUnetPure_tf_efficientnet_l2_ns_3_dice,round_TimmUnetPure_maxvit_base_tf_512.in21k_ft_in1k_3_dice,round_TimmUnetPure_tf_efficientnetv2_l_in21k_3_dice --fold 3 > logs/3 &

wait
echo "**********Finished predicting**********"

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -u -m torch.distributed.launch  --nproc_per_node=4 \
  --master_port 9972   train_all.py --world-size 4 --distributed --workers 16 --fold 0  \
  --config configs/all_l2.json --test_every 1   --fp16   --prefix final_ > logs/l2 &
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -u -m torch.distributed.launch  --nproc_per_node=4 \
  --master_port 9972   train_all.py --world-size 4 --distributed --workers 16 --fold 0  \
  --config configs/all_v2l.json --test_every 1   --fp16    --prefix final_ > logs/v2l &
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -u -m torch.distributed.launch  --nproc_per_node=4 \
  --master_port 9972   train_all.py --world-size 4 --distributed --workers 16 --fold 0  \
  --config configs/all_v2xl.json --test_every 1   --fp16    --prefix final_ > logs/v2xl &
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -u -m torch.distributed.launch  --nproc_per_node=4 \
  --master_port 9972   train_all.py --world-size 4 --distributed --workers 16 --fold 0  \
  --config configs/all_maxvit.json --test_every 1   --fp16    --prefix final_ > logs/maxvit &
wait

python unet_swa.py --path weights --exp_name final_TimmUnetPure_tf_efficientnetv2_xl_in21k_0   --num_best 5 --min_epoch 25  --maximize --metric_name dice
python unet_swa.py --path weights --exp_name final_TimmUnetPure_tf_efficientnetv2_l_in21k_0   --num_best 5 --min_epoch 25  --maximize --metric_name dice
python unet_swa.py --path weights --exp_name final_TimmUnetPure_tf_efficientnet_l2_ns_0   --num_best 5 --min_epoch 25  --maximize --metric_name dice
python unet_swa.py --path weights --exp_name final_TimmUnetPure_maxvit_base_tf_512.in21k_ft_in1k_0  --num_best 5 --min_epoch 25  --maximize --metric_name dice