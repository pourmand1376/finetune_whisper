NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" torchrun  --nproc_per_node 5  traing_without_preprocessing.py 

# accelerate launch traing_without_preprocessing.py 