import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0" 
os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from processing import Processing

if __name__ == "__main__":
    process_obj = Processing()
