# CUDA_VISIBLE_DEVICES=3 python profiler.py --crop_height=576 \
#                   --crop_width=960 \
#                   --max_disp=12 \
#                   --data_path='/data/schuster/BMW_SceneFlow/DATA/Freiburg/FlyingThings3D/' \
#                   --test_list='lists/sceneflow_train_1.list' \
#                   --save_path='./result/profiler/sceneflow_retrain_ms_d12/' \
# 		        --model='GANet_ms' \
#                   --resume='./checkpoint/sceneflow_retrain_ms_d12_epoch_5.pth' 2>&1 |tee logs/profiler/profile_GANet_ms_max_disp12.txt
# exit

# CUDA_VISIBLE_DEVICES=3 python profiler.py --crop_height=576 \
#                   --crop_width=960 \
#                   --max_disp=12 \
#                   --data_path='/data/schuster/BMW_SceneFlow/DATA/Freiburg/FlyingThings3D/' \
#                   --test_list='lists/sceneflow_train_1.list' \
#                   --save_path='./result/profiler/sceneflow_retrain_ms_d12/' \
# 		        --model='GANet_deep' \ 2>&1 |tee logs/profiler/profile_GANet_deep.txt
# exit


#CUDA_VISIBLE_DEVICES=0 python -u profiler.py \
#               --crop_height=384 \
#               --crop_width=1344 \
#	       --kitti2015=1 \
#               --max_disp=12  \
#               --data_path='./datasets/kitti2015/training/' \
#               --test_list='lists/kitti2015_train.list' \
#               --model='GANet_ms' 2>&1 |tee logs/profiler/profile_GANet_ms_max_disp12.txt

CUDA_VISIBLE_DEVICES=0 python -u profiler.py \
               --crop_height=192 \
               --crop_width=576 \
               --kitti2015=0 \
               --max_disp=12  \
               --data_path='./data/' \
               --test_list='lists/sceneflow_test_select.list' \
               --model='GANet_ms' 2>&1 |tee logs/profiler/profile_GANet_ms_sf_max_disp12.txt
