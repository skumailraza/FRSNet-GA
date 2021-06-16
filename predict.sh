#

srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=3 -p RTX3090  \
  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/FCE/gaflow:/netscratch/kraza/FCE/gaflow \
  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh  \
  --container-workdir=/netscratch/kraza/FCE/gaflow \
  --container-mount-home \
  python predict.py --crop_height=576 \
                  --crop_width=960 \
                  --max_disp=12 \
                  --data_path='demo/' \
                  --test_list='lists/demo.list' \
                  --save_path='result/hands/' \
                  --resume='checkpoint/GANetMS_hourglass_best.pth'

#CUDA_VISIBLE_DEVICES=7 python predict.py --crop_height=576 \
#                  --crop_width=960 \
#                  --max_disp=12 \
#                  --data_path='./datasets/data.mb/unzip/vision.middlebury.edu/stereo/data/' \
#                  --test_list='lists/middlebury_test_q.list' \
#                  --save_path='./result/middlebury_test_retrain/' \
#                  --kitti2015=0 \
#		        --middlebury=1 \
#		        --model='GANet_deep' \
#                  --resume='./checkpoint/kitti2015_final.pth'
#exit
#
#python predict.py --crop_height=384 \
#                  --crop_width=1248 \
#                  --max_disp=192 \
#                  --data_path='./datasets/data.mb/unzip/vision.middlebury.edu/stereo/data/' \
#                  --test_list='lists/middlebury_test_q.list' \
#                  --save_path='./result/middlebury_test/' \
#                  --kitti=1 \
#                  --resume='./checkpoint/kitti2012_final.pth'



