srun -K --ntasks=1 --gpus-per-task=4 --cpus-per-gpu=3 -p RTXA6000 --mem=64000 \
  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/FCE/gaflow:/netscratch/kraza/FCE/gaflow \
  --container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh  \
  --container-workdir=/netscratch/kraza/FCE/gaflow \
  --container-mount-home \
  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
  sh train_cluster.sh

#  srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=3 -p RTX6000  \
#  --container-mounts=/netscratch:/netscratch,/ds-av:/ds-av,/netscratch/kraza/FCE/gaflow:/netscratch/kraza/FCE/gaflow \
#  --container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh  \
#  --container-workdir=/netscratch/kraza/FCE/gaflow \
#  --container-mount-home \
#  --export="WANDB_API_KEY=406ca7642d853cdfbad965c078cf5c240a43a99e" \
#  sh torch_prof.sh