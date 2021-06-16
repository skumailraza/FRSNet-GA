## # Testing model experiments
###Train on Slurm
python -u train.py --batchSize=8 \
               --crop_height=384 \
               --crop_width=576 \
               --val_height=576 \
               --val_width=960 \
               --max_disp=12 \
               --thread=16 \
               --data_path='/netscratch/kraza/SceneFlow_Mixed/' \
               --training_list='lists/sceneflow_train.list' \
               --val_list='lists/sceneflow_test_select.list' \
               --save_path='./checkpoint/GANetMS_hourglass' \
               --cont=0 \
               --visualize=1 \
               --threshold=1.0 \
               --model='GANet_ms' \
               --lr=0.001 \
               --wandb=1 \
               --wbRunName="GANetMS_hourglass" \
               --model='GANet_ms' \
               --nEpochs=30 2>&1 |tee logs/train_GANetMS_hourglass.txt

# Kitti2015
python -u train.py --batchSize=8 \
               --crop_height=192 \
               --crop_width=1152 \
               --val_height=384 \
                --val_width=1344 \
                --max_disp=12 \
               --thread=16 \
               --data_path='/ds-av/public_datasets/kitti2015/raw/training/' \
               --training_list='lists/kitti2015_train.list' \
               --val_list='lists/kitti2015_val.list' \
               --save_path='./checkpoint/GANetMS_hourglass_ft' \
               --resume='./checkpoint/GANetMS_hourglass_best.pth' \
               --kitti2015=1 \
               --visualize=1 \
               --threshold=3.0 \
               --shift=3 \
               --wandb=1 \
               --wbRunName='GANetMS_hourglass_ft' \
               --model='GANet_ms' \
               --lr=0.001 \
               --nEpochs=1000 2>&1 |tee logs/train_GANetMS_hourglass_ft.txt

sh evaluation.sh
#srun -K --ntasks=1 --gpus-per-task=2 --cpus-per-gpu=3 -p A100 --mem=40000\
#  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/refinement/gaflow:/netscratch/kraza/refinement/gaflow \
#  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh  \
#  --container-workdir=/netscratch/kraza/refinement/gaflow \
#  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
#  python -u train.py --batchSize=8 \
#               --crop_height=192 \
#               --crop_width=1152 \
#               --max_disp=12 \
#               --thread=16 \
#               --data_path='/ds-av/public_datasets/kitti2015/raw/training/' \
#               --training_list='lists/kitti2015_all.list' \
#               --val_list='lists/kitti2015_train.list' \
#               --save_path='./checkpoint/3DConvtoSGA+1_refine_ft2' \
#               --resume='./checkpoint/3DConvtoSGA+1_refine_ft_best.pth' \
#               --kitti2015=1 \
#               --visualize=1 \
#               --threshold=3.0 \
#               --wandb=1 \
#               --wbRunName='3DConv-1_SGA+11_refine_ft_phase2' \
#               --shift=3 \
#               --model='GANet_ms' \
#               --lr=0.0001 \
#               --nEpochs=20 2>&1 |tee logs/train_3DConvtoSGA+1_refine_ft2.txt

#kitti2012
#srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=2 -p RTX3090 --mem=40000\
#  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/refinement/gaflow:/netscratch/kraza/refinement/gaflow \
#  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh  \
#  --container-workdir=/netscratch/kraza/refinement/gaflow \
#  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
#  python -u train.py --batchSize=16 \
#               --crop_height=192 \
#               --crop_width=576 \
#               --val_height=1344 \
#	             --val_width=384 \
#	             --max_disp=12 \
#               --thread=16 \
#               --data_path='./data/kitti/training/' \
#               --training_list='lists/kitti2012_train170.list' \
#               --val_list='lists/kitti2012_val24.list' \
#               --save_path='./checkpoint/3DConvtoSGA+1_ft_kitti2012' \
#               --resume='./checkpoint/3DConvtoSGA+1_best.pth' \
#               --kitti=1 \
#               --visualize=1 \
#               --shift=3 \
#               --wandb=1 \
#               --wbRunName='3DConv-1_SGA+1_ft_kitti2012_Slurm' \
#               --model='GANet_ms' \
#               --lr=0.001 \
#               --nEpochs=800 2>&1 |tee logs/train_3DConvtoSGA+1_ft_kitti2012.txt
#
#
#srun -K --ntasks=1 --gpus-per-task=2 --cpus-per-gpu=2 -p RTX3090 --mem=40000 \
#  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/refinement/gaflow:/netscratch/kraza/refinement/gaflow \
#  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh  \
#  --container-workdir=/netscratch/kraza/refinement/gaflow \
#  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
#  python -u train.py --batchSize=8 \
#               --crop_height=384 \
#               --crop_width=1344 \
#               --max_disp=12 \
#               --thread=16 \
#               --data_path='./data/kitti/training/' \
#               --training_list='lists/kitti2012_train170.list' \
#               --val_list='lists/kitti2012_val24.list' \
#               --save_path='./checkpoint/3DConvtoSGA+1_ft2_kitti2012' \
#               --resume='./checkpoint/3DConvtoSGA+1_ft_kitti2012_best.pth' \
#               --kitti=1 \
#               --visualize=1 \
#               --wandb=1 \
#               --wbRunName='3DConv-1_SGA+1_ft_phase2_kitti2012_Slurm' \
#               --shift=3 \
#               --model='GANet_ms' \
#               --lr=0.0001 \
#               --nEpochs=11 2>&1 |tee logs/train_3DConvtoSGA+1_ft2_kitti2012.txt


#CUDA_VISIBLE_DEVICES=0,1 python -u train.py --batchSize=4 \
#               --crop_height=192 \
#               --crop_width=576 \
#               --max_disp=12 \
#               --thread=16 \
#               --data_path='/mnt/serv-2101/serv-2105/data/schuster/BMW_SceneFlow/DATA/Freiburg/FlyingThings3D/' \
#               --training_list='lists/sceneflow_train_1.list' \
#               --save_path='./checkpoint/3DConvtoSGA' \
#               --resume='./checkpoint/train_nd_mg_warpFixed+_epoch_4.pth' \
#               --cont=0\
#               --visualize=1 \
#               --model='GANet_ms' \
#               --lr=0.001 \
#               --nEpochs=11 2>&1 |tee logs/train_3DConvtoSGA.txt

# CUDA_VISIBLE_DEVICES=0 python -u train.py --batchSize=4 \
#                --crop_height=192 \
#                --crop_width=576 \
#                --max_disp=4 \
#                --thread=16 \
#                --data_path='./data/' \
#                --training_list='lists/sceneflow_train_1.list' \
#                --save_path='./checkpoint/GANetMS_md_4_sc' \
#                --visualize=1 \
#                --model='GANet_ms' \
#                --lr=0.001 \
#                --nEpochs=11 2>&1 |tee logs/train_GANetMS_md_4_sc.txt

# exit

# CUDA_VISIBLE_DEVICES=1,2 python -u train.py --batchSize=2 \
#                 --crop_height=240 \
#                 --crop_width=720 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --kitti_data_path='./datasets/kitti/training/' \
#                 --eth3d_data_path='./datasets/eth3d/' \
#                 --kitti2015_data_path='./datasets/kitti2015/training/' \
#                 --middlebury_data_path='./datasets/data.mb/unzip/vision.middlebury.edu/stereo/data/' \
#                 --kitti_training_list='lists/kitti2012_train.list' \
#                 --eth3d_training_list='lists/eth3d.list' \
#                 --kitti2015_training_list='lists/kitti2015_train.list' \
#                 --middlebury_training_list='lists/middlebury_eval_new.list' \
#                 --save_path='./checkpoint/multi_training_all' \
#                 --val_list='./lists/middlebury_q.list' \
#                 --kitti2015=1 \
#                 --eth3d=1 \
#                 --kitti=1 \
#                 --middlebury=1 \
#                 --shift=3 \
#                 --visualize=0 \
#                 --multi=1 \
#                 --resume='./checkpoint/sceneflow_epoch_10.pth' \
#                 --nEpochs=600 2>&1 |tee logs/log_multi_train_all.txt

#CUDA_VISIBLE_DEVICES=0 python -u train.py --batchSize=4 \
#                --crop_height=192 \
#                --crop_width=576 \
#                --max_disp=12 \
#                --thread=16 \
#                --kitti_data_path='/mnt/serv-2101/public_datasets/kitti2015/raw/training/' \
#                --eth3d_data_path='./datasets/eth3d/' \
#                --kitti2015_data_path='./datasets/kitti2015/training/' \
#                --middlebury_data_path='./datasets/data.mb/unzip/vision.middlebury.edu/stereo/data/' \
#                --kitti_training_list='lists/kitti2012_train.list' \
#                --eth3d_training_list='lists/eth3d.list' \
#                --kitti2015_training_list='lists/kitti2015_train.list' \
#                --middlebury_training_list='lists/middlebury_eval_new.list' \
#                --val_list='./lists/kitti2015_train.list' \
#                --kitti2015=1 \
#                --eth3d=1 \
#                --kitti=1 \
#                --middlebury=1 \
#                --shift=3 \
#                --visualize=1 \
#                --multi=1 \
#                --resume='./checkpoint/train_nd_mg_warpFixed+_epoch_4.pth' \
#                --model='GANet_ms' \
#                --nEpochs=600 2>&1 |tee logs/log_GANetMSmulti_train_all.txt


# CUDA_VISIBLE_DEVICES=0 python -u train.py --batchSize=1 \
#                --crop_height=240 \
#                --crop_width=624 \
#                --max_disp=192 \
#                --thread=16 \
#                --data_path='./data/' \
#                --training_list='lists/sceneflow_train.list' \
#                --save_path='./checkpoint/sceneflow_retrain' \
#                --resume='' \
#                --visualize=0 \
#                --model='GANet_deep' \
#                --nEpochs=11 2>&1 |tee logs/log_train_sceneflow_retrain.txt

# exit



#Fine tuning for kitti 2015
# CUDA_VISIBLE_DEVICES=1,2 python -u train.py --batchSize=2 \
#                 --crop_height=288 \
#                 --crop_width=480 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --data_path='./datasets/kitti2015/training/' \
#                 --training_list='lists/kitti2015_train.list' \
#                 --save_path='./checkpoint/finetune_kitti2015_retrain' \
#                 --kitti2015=1 \
#                 --shift=3 \
#                 --visualize=0 \
#                 --resume='./checkpoint/sceneflow_epoch_10.pth' \
#                 --model='GANet_ms' \
#                 --nEpochs=10 2>&1 |tee logs/log_exp_finetune_kitti_ms.txt
# # exit
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=8 \
#                 --crop_height=240 \
#                 --crop_width=1248 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --data_path='/media/feihu/Storage/stereo/data_scene_flow/training/' \
#                 --training_list='lists/kitti2015_train.list' \
#                 --save_path='./checkpoint/finetune2_kitti2015' \
#                 --kitti2015=1 \
#                 --shift=3 \
#                 --lr=0.0001 \
#                 --resume='./checkpoint/finetune_kitti2015_epoch_800.pth' \
#                 --nEpochs=8 2>&1 |tee logs/log_finetune_kitti2015.txt

# #Fine tuning for kitti 2012

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 \
#                 --crop_height=240 \
#                 --crop_width=528 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --data_path='/media/feihu/Storage/stereo/kitti/training/' \
#                 --training_list='lists/kitti2012_train.list' \
#                 --save_path='./checkpoint/finetune_kitti' \
#                 --kitti=1 \
#                 --shift=3 \
#                 --resume='./checkpoint/sceneflow_epoch_10.pth' \
#                 --nEpochs=800 2>&1 |tee logs/log_finetune2_kitti.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=8 \
#                 --crop_height=240 \
#                 --crop_width=1248 \
#                 --max_disp=192 \
#                 --thread=16 \
#                 --data_path='/media/feihu/Storage/stereo/kitti/training/' \
#                 --training_list='lists/kitti2012_train.list' \
#                 --save_path='./checkpoint/finetune2_kitti' \
#                 --kitti=1 \
#                 --shift=3 \
#                 --lr=0.0001 \
#                 --resume='./checkpoint/finetune_kitti_epoch_800.pth' \
#                 --nEpochs=8 2>&1 |tee logs/log_finetune2_kitti.txt




