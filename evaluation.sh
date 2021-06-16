  python -u evaluation.py --crop_height=576 \
                  --crop_width=960 \
                  --max_disp=12 \
                  --data_path='/ds-av/public_datasets/freiburg_sceneflow_flyingthings3d/raw/' \
                  --test_list='lists/sceneflow_test_select.list' \
                  --save_path='./result/refined/sceneflow/' \
                  --resume='checkpoint/GANetMS_hourglass_best.pth' \
                  --threshold=1.0 2>&1 |tee logs/multi_scale/GANetMS_hourglass_sceneflow.txt


   python -u evaluation.py --crop_height=384 \
                   --crop_width=1152 \
                   --max_disp=12 \
                   --data_path='/ds-av/public_datasets/kitti2015/raw/training/' \
                   --test_list='lists/kitti2015_val.list' \
                   --save_path='./result/refined/kitti2015/' \
		             --resume='checkpoint/GANetMS_hourglass_ft_best.pth' \
                 --threshold=3.0 \
                 --benchmark=0\
		            --kitti2015=1 2>&1 |tee logs/multi_scale/GANetMS_hourglass_kitti2015.txt

#    srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=1 -p RTX3090 --mem=32000\
#  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/gaflow:/netscratch/kraza/gaflow \
#  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh  \
#  --container-workdir=/netscratch/kraza/gaflow \
#  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
#   python -u evaluation.py --crop_height=384 \
#                   --crop_width=768 \
#                   --max_disp=12 \
#                   --data_path='/ds-av/public_datasets/eth3d_low_res_two_view/raw/training/' \
#                   --test_list='lists/eth3d.list' \
#                   --save_path='./result/GANetMS/eth3d/' \
#		             --resume='checkpoint/GANetMS_best_refine_ft_best.pth' \
#                 --threshold=3.0 \
#                 --benchmark=0\
#		            --eth3d=1 2>&1 |tee logs/multi_scale/GANetMS_ft_refine_eth3d_best.txt

##    srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=1 -p RTX3090 --mem=32000\
##  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/gaflow:/netscratch/kraza/gaflow \
##  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh  \
##  --container-workdir=/netscratch/kraza/gaflow \
##  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
#   python -u evaluation.py --crop_height=576 \
#                   --crop_width=768 \
#                   --max_disp=12 \
#                   --data_path='/ds-av/public_datasets/middlebury_stereo_2014/raw/trainingH/' \
#                   --test_list='lists/middlebury_q.list' \
#                   --save_path='./result/GANetMS/middlebury/' \
#		              --resume='checkpoint/GANetMS_SDRNet_refine_ms_best.pth' \
#                 --threshold=3.0 \
#                 --benchmark=0\
#		             --middlebury=1 2>&1 |tee logs/multi_scale/GANetMS_ft_refine_middlebury_best.txt
#
##exit
##    srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=1 -p RTX3090 --mem=32000\
##  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/gaflow:/netscratch/kraza/gaflow \
##  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh  \
##  --container-workdir=/netscratch/kraza/gaflow \
##  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
#   python -u evaluation.py --crop_height=384 \
#                   --crop_width=1152 \
#                   --max_disp=12 \
#                   --data_path='/ds-av/public_datasets/kitti2012/raw/training/' \
#                   --test_list='lists/kitti2012_val24.list' \
#                   --save_path='./result/GANetMS/kitti2012/' \
#		             --resume='checkpoint/GANetMS_best_refine_ft_best.pth' \
#                 --threshold=3.0 \
#                 --benchmark=0\
#		            --kitti=1 2>&1 |tee logs/multi_scale/GANetMS_ft_refine_kitti2012_best.txt
# CUDA_VISIBLE_DEVICES=5 python -u evaluation.py --crop_height=576 \
#                   --crop_width=960 \
#                   --max_disp=192 \
# 		  --test_list='lists/middlebury_q.list' \
#  	          --save_path='./result/sceneflow_retrain_finetune/middlebury/' \
#   	          --resume='./checkpoint/multi_training_all_epoch_600.pth' \
#     	          --data_path='../gaflow/datasets/data.mb/unzip/vision.middlebury.edu/stereo/data/' \
#                   --middlebury=1 \
#  	          --threshold=1.0 2>&1 | tee logs/multi/log_eval_all_middlebury_600.txt
																	                    
# CUDA_VISIBLE_DEVICES=5 python -u evaluation.py --crop_height=576 \
#                   --crop_width=960 \
#                   --max_disp=192 \
#                   --test_list='lists/eth3d.list' \
#                   --save_path='./result/sceneflow_retrain_finetune/eth3d' \
#                   --resume='./checkpoint/multi_training_all_epoch_600.pth' \
#                   --data_path='../gaflow/datasets/eth3d/' \
#                   --eth3d=1 \
#                   --threshold=1.0 2>&1 |tee logs/multi/log_eval_all_eth3d_600.txt
# exit


# CUDA_VISIBLE_DEVICES=3 python -u evaluation.py --crop_height=384 \
#                    --crop_width=1344 \
#                    --max_disp=192 \
#                    --data_path='../gaflow/datasets/kitti2015/training/' \
#                    --test_list='lists/kitti2015_test.list' \
#                    --save_path='./result/multi_scale/kitti2015/' \
# 		        --resume='./checkpoint/finetune_kitti2015_retrain_epoch_50.pth' \
#                    --threshold=3.0 \
#                    --kitti2015=1 2>&1 |tee logs/multi_scale/log_eval_kitti_50.txt

# CUDA_VISIBLE_DEVICES=4 python -u evaluation.py --crop_height=576 \
#                   --crop_width=960 \
#                   --max_disp=192 \
# 		  --test_list='lists/middlebury_q.list' \
#  	          --save_path='./result/multi_scale/middlebury/' \
#   	          --resume='./checkpoint/finetune_kitti2015_retrain_epoch_50.pth' \
#     	          --data_path='../gaflow/datasets/data.mb/unzip/vision.middlebury.edu/stereo/data/' \
#                   --middlebury=1 \
#  	          --threshold=1.0 2>&1 | tee logs/multi_scale/log_eval_middlebury_50.txt
																	                    
# CUDA_VISIBLE_DEVICES=4 python -u evaluation.py --crop_height=576 \
#                   --crop_width=960 \
#                   --max_disp=192 \
#                   --test_list='lists/eth3d.list' \
#                   --save_path='./result/multi_scale/eth3d/' \
#                   --resume='./checkpoint/finetune_kitti2015_retrain_epoch_50.pth' \
#                   --data_path='../gaflow/datasets/eth3d/' \
#                   --eth3d=1 \
#                   --threshold=1.0 2>&1 |tee logs/multi_scale/log_eval_eth3d_50.txt
# exit


