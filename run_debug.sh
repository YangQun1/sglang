# rm .hlog_debug -r
# rm compile_debug.log

# export LOG_FILE_SIZE=1048576000
# export HABANA_LOGS=.hlog_debug
# export LOG_LEVEL_ALL=1

export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_ALLREDUCE_GRAPH_SPLIT=0
export PT_HPU_ENABLE_WAITTENSOR_GRAPH_SPLIT=0
# export PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH=0 # to WA concat incorrect cache hit issue.

# export TORCH_LOGS="+dynamo"

# python test_extend.py \
#     --device hpu \
#     --model-path meta-llama/Meta-Llama-3.1-8B-Instruct

# python -m pytest test/srt/test_torch_native_attention_backend.py::TestTorchNativeAttnBackend::test_mmlu -s

## llama3.1-8b
# python examples/runtime/engine/offline_batch_inference.py \
#     --device hpu \
#     --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --attention-backend torch_native

# deepseek v2 lite
# export HABANA_LOGS=hlog
# export LOG_LEVEL_ALL=1
# export LOG_FILE_SIZE=1048576000
# export SGLANG_SAVE_FIRST_N_LAYERS=2
# export SGLANG_TENSOR_DATA_DIR=hpu_tensor_data
# python examples/runtime/engine/offline_batch_inference.py \
#     --device hpu \
#     --model-path deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
#     --trust-remote-code \
#     --attention-backend torch_native
    # --disable-mla

# export SGLANG_TENSOR_DATA_DIR=cpu_tensor_data
# python examples/runtime/engine/offline_batch_inference.py \
#     --device cpu \
#     --model-path deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
#     --trust-remote-code \
#     --attention-backend torch_native \
#     --disable-mla

# deepseek r1
# export HABANA_LOGS=hlog
# export LOG_LEVEL_ALL=1
# export LOG_FILE_SIZE=1048576000
# python examples/runtime/engine/offline_batch_inference.py \
#     --device hpu \
#     --tp 1 \
#     --model-path /software/data/DeepSeek-R1 \
#     --trust-remote-code \
#     --disable-mla \
#     --load-format dummy \
#     --watchdog-timeout 3000

# python3 -m sglang.bench_one_batch \
#     --batch-size 1 \
#     --input 1024 \
#     --output 8 \
#     --model /software/data/DeepSeek-R1 \
#     --trust-remote-code \
#     --device hpu \
#     --tp 8 \
#     --load-format dummy \
#     --disable-mla

export SGLANG_NUM_DECODER_LAYERS=61
export SGLANG_SAVE_FIRST_N_LAYERS=61
# export SGLANG_TENSOR_DATA_DIR=tensor_data/hpu/
# export SGLANG_TORCH_PROFILER_DIR=~/upstream/sglang/profile/
python examples/runtime/engine/offline_batch_inference.py \
    --device hpu \
    --tp 8 \
    --model-path /software/data/DeepSeek-R1 \
    --trust-remote-code \
    --attention-backend torch_native \
    --disable-mla \
    --watchdog-timeout 3000 > disable_mla.log 2>&1

python examples/runtime/engine/offline_batch_inference.py \
    --device hpu \
    --tp 8 \
    --model-path /software/data/DeepSeek-R1 \
    --trust-remote-code \
    --attention-backend torch_native \
    --watchdog-timeout 3000 > enable_mla.log 2>&1

# export PT_HPU_ENABLE_RECORD_STREAM=1
# export PT_HPU_ENABLE_RECORD_STREAM_NOHOLDER=1
# export PT_HPU_USE_LAUNCH_RECORD_STREAM=1
# export HABANA_PROFILE=1
# export SGLANG_TORCH_PROFILER_DIR=~/upstream/sglang/profile/
# python3 -m sglang.bench_one_batch \
#     --batch-size 2 \
#     --input 256 \
#     --output 8 \
#     --model /software/data/DeepSeek-R1 \
#     --trust-remote-code \
#     --device hpu \
#     --tp 8 \
#     --disable-mla \
#     --watchdog-timeout 3000 \
#     --profile \
#     --profile-filename-prefix profile_$(date +%Y%m%d_%H%M%S)
    # > dynamo.log 2>&1

# deepseek v3
# python examples/runtime/engine/offline_batch_inference.py \
#     --device hpu \
#     --tp 8 \
#     --model-path /mnt/weka/DeepSeek-V3 \
#     --trust-remote-code
    # --disable-mla

# python examples/runtime/engine/offline_batch_inference.py \
#     --device hpu \
#     --tp-size 2 \
#     --model-path meta-llama/Meta-Llama-3.1-8B-Instruct


# --enable-torch-compile \
# > compile_debug.log 2>&1

# python examples/runtime/engine/offline_batch_inference.py --device cuda --model-path /software/data/pytorch/llama3/Meta-Llama-3-8B-Instruct --attention-backend triton --disable-cuda-graph

# python python/sglang/bench_offline_throughput.py \
#     --device hpu \
#     --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --num-prompts 20 \
#     --dataset-path /home/quyang/qnpu/pt/src/vllm-fork/ShareGPT_V3_unfiltered_cleaned_split.json
#     # --enable-torch-compile


# python python/sglang/bench_offline_throughput.py \
#     --device cuda \
#     --model-path /software/data/pytorch/llama3/Meta-Llama-3-8B-Instruct \
#     --num-prompts 1000 \
#     --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json

# TRITON_INTERPRET=1 python -m pytest test/srt/test_triton_attention_kernels.py::TestTritonAttention::test_decode_attention

# TRITON_INTERPRET=1 python -m pytest test/srt/test_triton_attention_kernels.py::TestTritonAttention::test_grouped_decode_attention

# python python/sglang/bench_offline_throughput.py --device cuda --model-path /software/data/pytorch/llama3/Meta-Llama-3-8B-Instruct --num-prompts 1000 --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --attention-backend triton


# launch server
# python3 -m sglang.launch_server \
#     --model deepseek-ai/DeepSeek-V2-Lite \
#     --trust-remote-code \
#     --device hpu \
#     --disable-mla \
#     --port 31234 \
#     --log-requests