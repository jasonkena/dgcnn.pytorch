default:
    just --list
train:
    python main_ribseg.py --exp_name=ribseg_2048_40_32 --num_points=2048 --k=40 --batch_size=32 --dataset_path="../ribseg_benchmark"

train_binary:
    python main_ribseg.py --exp_name=ribseg_2048_40_32_binary --num_points=2048 --k=40 --batch_size=32 --binary --dataset_path="../ribseg_benchmark"

# note that second stage doesn't run with binary flag since it predicts over 25 classes
train_binary_second_stage:
    python main_ribseg.py --exp_name=ribseg_2048_40_32_binary_second_stage --num_points=2048 --k=40 --batch_size=32 --dataset_path="/data/adhinart/ribseg/outputs/dgcnn" --binary_dataset_path="/data/adhinart/ribseg/outputs/dgcnn_binary"

test_batch_size := "800"
inference:
    python inference_ribseg.py --exp_name=ribseg_2048_40_32 --num_points=2048 --k=40 --test_batch_size={{test_batch_size}} --model_path="/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32/models/model.t7" --dataset_path="../ribseg_benchmark"

# need to run summarize_outputs.py to generate 
inference_binary:
    python inference_ribseg.py --exp_name=ribseg_2048_40_32_binary --num_points=2048 --k=40 --test_batch_size={{test_batch_size}} --model_path="/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32_binary/models/model.t7" --binary --dataset_path="../ribseg_benchmark"

# see note on train_binary_second_stage
inference_binary_second_stage:
    python inference_ribseg.py --exp_name=ribseg_2048_40_32_binary_second_stage --num_points=2048 --k=40 --test_batch_size={{test_batch_size}} --model_path="/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32_binary_second_stage/models/model.t7" --dataset_path="/data/adhinart/ribseg/outputs/dgcnn" --binary_dataset_path="/data/adhinart/ribseg/outputs/dgcnn_binary"

time_num_samples := "10"
time_batch_size := "16"
time:
    python inference_ribseg.py --exp_name=ribseg_2048_40_32 --num_points=2048 --k=40 --test_batch_size={{time_batch_size}} --model_path="/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32/models/model.t7" --dataset_path="../ribseg_benchmark" --dry_run={{time_num_samples}}

# need to run summarize_outputs.py to generate 
time_binary:
    python inference_ribseg.py --exp_name=ribseg_2048_40_32_binary --num_points=2048 --k=40 --test_batch_size={{time_batch_size}} --model_path="/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32_binary/models/model.t7" --binary --dataset_path="../ribseg_benchmark" --dry_run={{time_num_samples}}

# see note on train_binary_second_stage
time_binary_second_stage:
    python inference_ribseg.py --exp_name=ribseg_2048_40_32_binary_second_stage --num_points=2048 --k=40 --test_batch_size={{time_batch_size}} --model_path="/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32_binary_second_stage/models/model.t7" --dataset_path="/data/adhinart/ribseg/outputs/dgcnn" --binary_dataset_path="/data/adhinart/ribseg/outputs/dgcnn_binary" --dry_run={{time_num_samples}}
