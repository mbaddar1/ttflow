# Desc 
Dataset : gaussian mixture 2D, num_comp =  4 
vt Model : NN
trajectory : global

# Model and OPT details
Network is on: cuda:0
2026-10-11 13:33:20.322 | INFO     | __main__:<module>:240 - Created optimizer : Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.001
)
2026-10-11 13:33:20.322 | INFO     | __main__:<module>:242 - Training model: VelocityFieldRegressionNet(
  (model): Sequential(
    (0): Linear(in_features=3, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=2, bias=True)
  )
) | Optimizer: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.001
) | Train shape: torch.Size([500000, 3]) | Test shape: torch.Size([200000, 3]) | Input dim: 3 | Hidden dim: 128 | Output dim: 2 | Depth: 2 | Device: cuda | Data type: torch.float32 | Batch size: 2048 | Epochs: 100 | Patience: 10 | Min rel loss delta: 5e-05

# Training Log 
```log
/home/mbaddar/Documents/mbaddar/ttflow/.venv/bin/python /home/mbaddar/Documents/mbaddar/ttflow/ttflow/training/nn/vt_trainer.py --data-snapshot-filepath data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_4_data_dim_2_n_train_test_eval_500000_200000_200000.pkl --vt-model nn --vt-opt adam --trajectory-model global 
2026-10-11 13:33:19.292 | INFO     | __main__:<module>:156 - args = Namespace(data_snapshot_filepath='data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_4_data_dim_2_n_train_test_eval_500000_200000_200000.pkl', use_gpu=True, vt_model='nn', vt_opt='adam', trajectory_model='global', rectification_linearity='linear', nn_total_num_param_updates=10001, nn_hidden_dim=128, batch_size=2048, nn_depth=2, lr=0.001, nn_reg_lambda=0.001, patience=10, min_rel_loss_delta=5e-05, dry_run=False, logs_dir='logs', output_model_dir='models/experimental')
2026-10-11 13:33:19.317 | INFO     | __main__:<module>:161 - is GPU/CUDA available = True
2026-10-11 13:33:19.317 | INFO     | __main__:<module>:164 - device = cuda
2026-10-11 13:33:19.317 | INFO     | __main__:<module>:168 - Using data snapshot: data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_4_data_dim_2_n_train_test_eval_500000_200000_200000.pkl
2026-10-11 13:33:19.317 | INFO     | ttflow.utils:check_system_memory_and_swap:56 - ram = 30.96, swap = 40.0 GB
2026-10-11 13:33:19.353 | INFO     | __main__:<module>:197 - data pkl size = 0.03 GB
2026-10-11 13:33:19.463 | INFO     | __main__:<module>:213 - Generating training data for the global trajectory training process..
2026-10-11 13:33:19.489 | INFO     | __main__:<module>:221 - Generating training data with xt_train_aug.shape = torch.Size([500000, 3]) x_test_aug.shape = torch.Size([200000, 3]) y_train.shape = torch.Size([500000, 2]) y_test.shape = torch.Size([200000, 2])
2026-10-11 13:33:19.491 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-11 13:33:20.322 | INFO     | __main__:<module>:240 - Created optimizer : Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.001
)
2026-10-11 13:33:20.322 | INFO     | __main__:<module>:242 - Training model: VelocityFieldRegressionNet(
  (model): Sequential(
    (0): Linear(in_features=3, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=2, bias=True)
  )
) | Optimizer: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0.001
) | Train shape: torch.Size([500000, 3]) | Test shape: torch.Size([200000, 3]) | Input dim: 3 | Hidden dim: 128 | Output dim: 2 | Depth: 2 | Device: cuda | Data type: torch.float32 | Batch size: 2048 | Epochs: 100 | Patience: 10 | Min rel loss delta: 5e-05
2026-10-11 13:33:21.214 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:58 - @epoch = 0 (before training), loss_fn = MSELoss(), loss_value = 50.253784
2026-10-11 13:33:21.219 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:65 - MSELoss(test) before training:50.2537841796875
Epoch:   0%|          | 0/100 [00:00<?, ?it/s]2026-10-11 13:33:24.451 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 1, smoothed-train-loss = 14.358298 | smoothed-test-loss = 9.680092 | rel_loss_delta = inf | loss_stagnation_count = 0
Epoch:   1%|          | 1/100 [00:03<05:19,  3.23s/it]2026-10-11 13:33:27.782 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 2, smoothed-train-loss = 10.035008 | smoothed-test-loss = 9.318899 | rel_loss_delta = inf | loss_stagnation_count = 0
Epoch:   2%|▏         | 2/100 [00:06<05:22,  3.29s/it]2026-10-11 13:33:30.969 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 3, smoothed-train-loss = 9.359559 | smoothed-test-loss = 9.084005 | rel_loss_delta = -0.0673092749043315 | loss_stagnation_count = 0
Epoch:   3%|▎         | 3/100 [00:09<05:14,  3.24s/it]2026-10-11 13:33:33.977 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 4, smoothed-train-loss = 9.077794 | smoothed-test-loss = 8.855647 | rel_loss_delta = -0.03010444499161429 | loss_stagnation_count = 0
Epoch:   4%|▍         | 4/100 [00:12<05:02,  3.15s/it]2026-10-11 13:33:36.983 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 5, smoothed-train-loss = 8.848770 | smoothed-test-loss = 8.652303 | rel_loss_delta = -0.025229021174253255 | loss_stagnation_count = 0
Epoch:   5%|▌         | 5/100 [00:15<04:54,  3.10s/it]2026-10-11 13:33:39.930 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 6, smoothed-train-loss = 8.649569 | smoothed-test-loss = 8.488568 | rel_loss_delta = -0.022511812965235124 | loss_stagnation_count = 0
Epoch:   6%|▌         | 6/100 [00:18<04:46,  3.05s/it]2026-10-11 13:33:43.118 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 7, smoothed-train-loss = 8.509797 | smoothed-test-loss = 8.375711 | rel_loss_delta = -0.01615933255126147 | loss_stagnation_count = 0
Epoch:   7%|▋         | 7/100 [00:21<04:47,  3.09s/it]2026-10-11 13:33:46.361 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 8, smoothed-train-loss = 8.412407 | smoothed-test-loss = 8.286896 | rel_loss_delta = -0.011444492873035403 | loss_stagnation_count = 0
Epoch:   8%|▊         | 8/100 [00:25<04:48,  3.14s/it]2026-10-11 13:33:50.038 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 9, smoothed-train-loss = 8.341014 | smoothed-test-loss = 8.265092 | rel_loss_delta = -0.00848661579580942 | loss_stagnation_count = 0
Epoch:   9%|▉         | 9/100 [00:28<05:01,  3.31s/it]2026-10-11 13:33:53.741 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 10, smoothed-train-loss = 8.297317 | smoothed-test-loss = 8.238584 | rel_loss_delta = -0.005238812508125898 | loss_stagnation_count = 0
Epoch:  10%|█         | 10/100 [00:32<05:08,  3.43s/it]2026-10-11 13:33:57.282 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 11, smoothed-train-loss = 8.261703 | smoothed-test-loss = 8.174577 | rel_loss_delta = -0.004292179479581141 | loss_stagnation_count = 0
Epoch:  11%|█         | 11/100 [00:36<05:08,  3.46s/it]2026-10-11 13:34:00.427 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 12, smoothed-train-loss = 8.235781 | smoothed-test-loss = 8.177643 | rel_loss_delta = -0.0031376898016966313 | loss_stagnation_count = 0
Epoch:  12%|█▏        | 12/100 [00:39<04:56,  3.37s/it]2026-10-11 13:34:04.013 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 13, smoothed-train-loss = 8.218784 | smoothed-test-loss = 8.183732 | rel_loss_delta = -0.002063718509773068 | loss_stagnation_count = 0
Epoch:  13%|█▎        | 13/100 [00:42<04:58,  3.43s/it]2026-10-11 13:34:07.229 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 14, smoothed-train-loss = 8.207598 | smoothed-test-loss = 8.145095 | rel_loss_delta = -0.0013611481463539398 | loss_stagnation_count = 0
Epoch:  14%|█▍        | 14/100 [00:46<04:49,  3.37s/it]2026-10-11 13:34:10.497 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 15, smoothed-train-loss = 8.194286 | smoothed-test-loss = 8.128459 | rel_loss_delta = -0.001621845442718142 | loss_stagnation_count = 0
Epoch:  15%|█▌        | 15/100 [00:49<04:43,  3.34s/it]2026-10-11 13:34:13.656 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 16, smoothed-train-loss = 8.188501 | smoothed-test-loss = 8.119764 | rel_loss_delta = -0.0007059347055753115 | loss_stagnation_count = 0
Epoch:  16%|█▌        | 16/100 [00:52<04:35,  3.28s/it]2026-10-11 13:34:16.730 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 17, smoothed-train-loss = 8.179481 | smoothed-test-loss = 8.109865 | rel_loss_delta = -0.0011015666433811665 | loss_stagnation_count = 0
Epoch:  17%|█▋        | 17/100 [00:55<04:27,  3.22s/it]2026-10-11 13:34:19.697 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 18, smoothed-train-loss = 8.175688 | smoothed-test-loss = 8.103550 | rel_loss_delta = -0.0004637292094502847 | loss_stagnation_count = 0
Epoch:  18%|█▊        | 18/100 [00:58<04:17,  3.14s/it]2026-10-11 13:34:22.885 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 19, smoothed-train-loss = 8.171845 | smoothed-test-loss = 8.100465 | rel_loss_delta = -0.0004700374186636981 | loss_stagnation_count = 0
Epoch:  19%|█▉        | 19/100 [01:01<04:15,  3.16s/it]2026-10-11 13:34:26.079 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 20, smoothed-train-loss = 8.167731 | smoothed-test-loss = 8.122649 | rel_loss_delta = -0.0005035172947789731 | loss_stagnation_count = 0
Epoch:  20%|██        | 20/100 [01:04<04:13,  3.17s/it]2026-10-11 13:34:28.979 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 21, smoothed-train-loss = 8.164350 | smoothed-test-loss = 8.103265 | rel_loss_delta = -0.0004139402741515764 | loss_stagnation_count = 0
Epoch:  21%|██        | 21/100 [01:07<04:03,  3.09s/it]2026-10-11 13:34:32.141 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 22, smoothed-train-loss = 8.160337 | smoothed-test-loss = 8.091356 | rel_loss_delta = -0.0004914920721132228 | loss_stagnation_count = 0
Epoch:  22%|██▏       | 22/100 [01:10<04:02,  3.11s/it]2026-10-11 13:34:35.380 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 23, smoothed-train-loss = 8.160305 | smoothed-test-loss = 8.083999 | rel_loss_delta = -3.884030661342325e-06 | loss_stagnation_count = 1
Epoch:  23%|██▎       | 23/100 [01:14<04:02,  3.15s/it]2026-10-11 13:34:38.527 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 24, smoothed-train-loss = 8.156765 | smoothed-test-loss = 8.108321 | rel_loss_delta = -0.000433799963812711 | loss_stagnation_count = 0
Epoch:  24%|██▍       | 24/100 [01:17<03:59,  3.15s/it]2026-10-11 13:34:41.890 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 25, smoothed-train-loss = 8.156862 | smoothed-test-loss = 8.080216 | rel_loss_delta = 1.1840160792676029e-05 | loss_stagnation_count = 1
Epoch:  25%|██▌       | 25/100 [01:20<04:00,  3.21s/it]2026-10-11 13:34:44.979 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 26, smoothed-train-loss = 8.152901 | smoothed-test-loss = 8.091149 | rel_loss_delta = -0.00048557915926661913 | loss_stagnation_count = 0
Epoch:  26%|██▌       | 26/100 [01:23<03:54,  3.18s/it]2026-10-11 13:34:48.530 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 27, smoothed-train-loss = 8.148511 | smoothed-test-loss = 8.102849 | rel_loss_delta = -0.0005384783737449171 | loss_stagnation_count = 0
Epoch:  27%|██▋       | 27/100 [01:27<04:00,  3.29s/it]2026-10-11 13:34:52.192 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 28, smoothed-train-loss = 8.147268 | smoothed-test-loss = 8.100636 | rel_loss_delta = -0.00015254899623908354 | loss_stagnation_count = 0
Epoch:  28%|██▊       | 28/100 [01:30<04:04,  3.40s/it]2026-10-11 13:34:56.056 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 29, smoothed-train-loss = 8.149750 | smoothed-test-loss = 8.077330 | rel_loss_delta = 0.0003046158484763008 | loss_stagnation_count = 0
Epoch:  29%|██▉       | 29/100 [01:34<04:11,  3.54s/it]2026-10-11 13:34:59.019 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 30, smoothed-train-loss = 8.145075 | smoothed-test-loss = 8.105619 | rel_loss_delta = -0.0005736010176702517 | loss_stagnation_count = 0
Epoch:  30%|███       | 30/100 [01:37<03:55,  3.37s/it]2026-10-11 13:35:01.995 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 31, smoothed-train-loss = 8.149481 | smoothed-test-loss = 8.093882 | rel_loss_delta = 0.0005409827591462058 | loss_stagnation_count = 0
Epoch:  31%|███       | 31/100 [01:40<03:44,  3.25s/it]2026-10-11 13:35:04.873 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 32, smoothed-train-loss = 8.145909 | smoothed-test-loss = 8.078323 | rel_loss_delta = -0.00043837401584734324 | loss_stagnation_count = 0
Epoch:  32%|███▏      | 32/100 [01:43<03:33,  3.14s/it]2026-10-11 13:35:08.187 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 33, smoothed-train-loss = 8.141411 | smoothed-test-loss = 8.063302 | rel_loss_delta = -0.0005521079383584347 | loss_stagnation_count = 0
Epoch:  33%|███▎      | 33/100 [01:46<03:33,  3.19s/it]2026-10-11 13:35:11.513 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 34, smoothed-train-loss = 8.140209 | smoothed-test-loss = 8.075310 | rel_loss_delta = -0.0001476957960560191 | loss_stagnation_count = 0
Epoch:  34%|███▍      | 34/100 [01:50<03:33,  3.23s/it]2026-10-11 13:35:14.758 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 35, smoothed-train-loss = 8.141029 | smoothed-test-loss = 8.078387 | rel_loss_delta = 0.0001007311346996766 | loss_stagnation_count = 0
Epoch:  35%|███▌      | 35/100 [01:53<03:30,  3.24s/it]2026-10-11 13:35:17.920 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 36, smoothed-train-loss = 8.138015 | smoothed-test-loss = 8.085803 | rel_loss_delta = -0.0003702678558535139 | loss_stagnation_count = 0
Epoch:  36%|███▌      | 36/100 [01:56<03:25,  3.21s/it]2026-10-11 13:35:21.547 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 37, smoothed-train-loss = 8.139193 | smoothed-test-loss = 8.081338 | rel_loss_delta = 0.00014485824853696552 | loss_stagnation_count = 0
Epoch:  37%|███▋      | 37/100 [02:00<03:30,  3.34s/it]2026-10-11 13:35:24.674 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 38, smoothed-train-loss = 8.136330 | smoothed-test-loss = 8.058299 | rel_loss_delta = -0.0003517521256822064 | loss_stagnation_count = 0
Epoch:  38%|███▊      | 38/100 [02:03<03:23,  3.27s/it]2026-10-11 13:35:28.344 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 39, smoothed-train-loss = 8.136787 | smoothed-test-loss = 8.076739 | rel_loss_delta = 5.60982095063516e-05 | loss_stagnation_count = 0
Epoch:  39%|███▉      | 39/100 [02:07<03:26,  3.39s/it]2026-10-11 13:35:31.760 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 40, smoothed-train-loss = 8.136498 | smoothed-test-loss = 8.077462 | rel_loss_delta = -3.549461437836323e-05 | loss_stagnation_count = 1
Epoch:  40%|████      | 40/100 [02:10<03:23,  3.40s/it]2026-10-11 13:35:35.268 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 41, smoothed-train-loss = 8.132174 | smoothed-test-loss = 8.084605 | rel_loss_delta = -0.0005314348005151358 | loss_stagnation_count = 0
Epoch:  41%|████      | 41/100 [02:14<03:22,  3.43s/it]2026-10-11 13:35:38.658 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 42, smoothed-train-loss = 8.132534 | smoothed-test-loss = 8.080176 | rel_loss_delta = 4.4212371263343944e-05 | loss_stagnation_count = 1
Epoch:  42%|████▏     | 42/100 [02:17<03:18,  3.42s/it]2026-10-11 13:35:41.981 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 43, smoothed-train-loss = 8.135832 | smoothed-test-loss = 8.092945 | rel_loss_delta = 0.00040555369437209147 | loss_stagnation_count = 0
Epoch:  43%|████▎     | 43/100 [02:20<03:13,  3.39s/it]2026-10-11 13:35:45.556 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 44, smoothed-train-loss = 8.130439 | smoothed-test-loss = 8.092005 | rel_loss_delta = -0.0006628964224638388 | loss_stagnation_count = 0
Epoch:  44%|████▍     | 44/100 [02:24<03:12,  3.45s/it]2026-10-11 13:35:48.876 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 45, smoothed-train-loss = 8.131805 | smoothed-test-loss = 8.087098 | rel_loss_delta = 0.00016802773371836393 | loss_stagnation_count = 0
Epoch:  45%|████▌     | 45/100 [02:27<03:07,  3.41s/it]2026-10-11 13:35:51.933 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 46, smoothed-train-loss = 8.134749 | smoothed-test-loss = 8.081540 | rel_loss_delta = 0.0003620166065218027 | loss_stagnation_count = 0
Epoch:  46%|████▌     | 46/100 [02:30<02:58,  3.30s/it]2026-10-11 13:35:55.335 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 47, smoothed-train-loss = 8.132894 | smoothed-test-loss = 8.086529 | rel_loss_delta = -0.00022797149655141196 | loss_stagnation_count = 0
Epoch:  47%|████▋     | 47/100 [02:34<02:56,  3.33s/it]2026-10-11 13:35:58.758 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 48, smoothed-train-loss = 8.128916 | smoothed-test-loss = 8.077761 | rel_loss_delta = -0.0004891479090797384 | loss_stagnation_count = 0
Epoch:  48%|████▊     | 48/100 [02:37<02:54,  3.36s/it]2026-10-11 13:36:02.261 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 49, smoothed-train-loss = 8.129202 | smoothed-test-loss = 8.064225 | rel_loss_delta = 3.522758559312411e-05 | loss_stagnation_count = 1
Epoch:  49%|████▉     | 49/100 [02:41<02:53,  3.40s/it]2026-10-11 13:36:05.829 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 50, smoothed-train-loss = 8.130307 | smoothed-test-loss = 8.073361 | rel_loss_delta = 0.00013587994467253848 | loss_stagnation_count = 0
Epoch:  50%|█████     | 50/100 [02:44<02:52,  3.45s/it]2026-10-11 13:36:09.454 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 51, smoothed-train-loss = 8.129334 | smoothed-test-loss = 8.082182 | rel_loss_delta = -0.00011960966530217183 | loss_stagnation_count = 0
Epoch:  51%|█████     | 51/100 [02:48<02:51,  3.50s/it]2026-10-11 13:36:12.577 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 52, smoothed-train-loss = 8.130750 | smoothed-test-loss = 8.081652 | rel_loss_delta = 0.00017410988213310924 | loss_stagnation_count = 0
Epoch:  52%|█████▏    | 52/100 [02:51<02:42,  3.39s/it]2026-10-11 13:36:15.951 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 53, smoothed-train-loss = 8.130820 | smoothed-test-loss = 8.057777 | rel_loss_delta = 8.666612417173764e-06 | loss_stagnation_count = 1
Epoch:  53%|█████▎    | 53/100 [02:54<02:39,  3.39s/it]2026-10-11 13:36:19.069 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 54, smoothed-train-loss = 8.124652 | smoothed-test-loss = 8.060470 | rel_loss_delta = -0.0007586662199795941 | loss_stagnation_count = 0
Epoch:  54%|█████▍    | 54/100 [02:57<02:32,  3.30s/it]2026-10-11 13:36:22.143 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 55, smoothed-train-loss = 8.125154 | smoothed-test-loss = 8.060925 | rel_loss_delta = 6.178362508036311e-05 | loss_stagnation_count = 0
Epoch:  55%|█████▌    | 55/100 [03:00<02:25,  3.24s/it]2026-10-11 13:36:24.973 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 56, smoothed-train-loss = 8.127468 | smoothed-test-loss = 8.108473 | rel_loss_delta = 0.0002848501507977248 | loss_stagnation_count = 0
Epoch:  56%|█████▌    | 56/100 [03:03<02:17,  3.11s/it]2026-10-11 13:36:27.879 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 57, smoothed-train-loss = 8.126602 | smoothed-test-loss = 8.089046 | rel_loss_delta = -0.00010650252984154814 | loss_stagnation_count = 0
Epoch:  57%|█████▋    | 57/100 [03:06<02:11,  3.05s/it]2026-10-11 13:36:30.930 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 58, smoothed-train-loss = 8.127897 | smoothed-test-loss = 8.053663 | rel_loss_delta = 0.00015935207629323464 | loss_stagnation_count = 0
Epoch:  58%|█████▊    | 58/100 [03:09<02:08,  3.05s/it]2026-10-11 13:36:33.846 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 59, smoothed-train-loss = 8.122661 | smoothed-test-loss = 8.071838 | rel_loss_delta = -0.0006442215486376768 | loss_stagnation_count = 0
Epoch:  59%|█████▉    | 59/100 [03:12<02:03,  3.01s/it]2026-10-11 13:36:36.675 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 60, smoothed-train-loss = 8.129700 | smoothed-test-loss = 8.073368 | rel_loss_delta = 0.0008666026221245244 | loss_stagnation_count = 0
Epoch:  60%|██████    | 60/100 [03:15<01:58,  2.96s/it]2026-10-11 13:36:39.943 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 61, smoothed-train-loss = 8.126181 | smoothed-test-loss = 8.054390 | rel_loss_delta = -0.0004328849725285244 | loss_stagnation_count = 0
Epoch:  61%|██████    | 61/100 [03:18<01:58,  3.05s/it]2026-10-11 13:36:43.135 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 62, smoothed-train-loss = 8.127676 | smoothed-test-loss = 8.111421 | rel_loss_delta = 0.00018400980681948674 | loss_stagnation_count = 0
Epoch:  62%|██████▏   | 62/100 [03:21<01:57,  3.09s/it]2026-10-11 13:36:46.271 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 63, smoothed-train-loss = 8.133222 | smoothed-test-loss = 8.072127 | rel_loss_delta = 0.000682246799418037 | loss_stagnation_count = 0
Epoch:  63%|██████▎   | 63/100 [03:25<01:54,  3.11s/it]2026-10-11 13:36:49.263 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 64, smoothed-train-loss = 8.126836 | smoothed-test-loss = 8.062460 | rel_loss_delta = -0.0007850957510315441 | loss_stagnation_count = 0
Epoch:  64%|██████▍   | 64/100 [03:28<01:50,  3.07s/it]2026-10-11 13:36:52.446 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 65, smoothed-train-loss = 8.127657 | smoothed-test-loss = 8.059059 | rel_loss_delta = 0.00010096911177170418 | loss_stagnation_count = 0
Epoch:  65%|██████▌   | 65/100 [03:31<01:48,  3.10s/it]2026-10-11 13:36:55.659 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 66, smoothed-train-loss = 8.121577 | smoothed-test-loss = 8.070327 | rel_loss_delta = -0.0007480271004682524 | loss_stagnation_count = 0
Epoch:  66%|██████▌   | 66/100 [03:34<01:46,  3.14s/it]2026-10-11 13:36:59.043 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 67, smoothed-train-loss = 8.123464 | smoothed-test-loss = 8.055405 | rel_loss_delta = 0.00023232188685648548 | loss_stagnation_count = 0
Epoch:  67%|██████▋   | 67/100 [03:37<01:45,  3.21s/it]2026-10-11 13:37:02.597 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 68, smoothed-train-loss = 8.123818 | smoothed-test-loss = 8.072528 | rel_loss_delta = 4.362674397274998e-05 | loss_stagnation_count = 1
Epoch:  68%|██████▊   | 68/100 [03:41<01:46,  3.31s/it]2026-10-11 13:37:06.391 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 69, smoothed-train-loss = 8.123784 | smoothed-test-loss = 8.071067 | rel_loss_delta = -4.216611513319648e-06 | loss_stagnation_count = 2
Epoch:  69%|██████▉   | 69/100 [03:45<01:47,  3.46s/it]2026-10-11 13:37:09.897 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 70, smoothed-train-loss = 8.125103 | smoothed-test-loss = 8.067251 | rel_loss_delta = 0.00016239698908992697 | loss_stagnation_count = 0
Epoch:  70%|███████   | 70/100 [03:48<01:44,  3.47s/it]2026-10-11 13:37:13.352 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 71, smoothed-train-loss = 8.121423 | smoothed-test-loss = 8.065484 | rel_loss_delta = -0.0004529147268288647 | loss_stagnation_count = 0
Epoch:  71%|███████   | 71/100 [03:52<01:40,  3.47s/it]2026-10-11 13:37:16.426 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 72, smoothed-train-loss = 8.124212 | smoothed-test-loss = 8.069000 | rel_loss_delta = 0.0003434351893154352 | loss_stagnation_count = 0
Epoch:  72%|███████▏  | 72/100 [03:55<01:33,  3.35s/it]2026-10-11 13:37:19.543 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 73, smoothed-train-loss = 8.127328 | smoothed-test-loss = 8.047557 | rel_loss_delta = 0.00038350301020400094 | loss_stagnation_count = 0
Epoch:  73%|███████▎  | 73/100 [03:58<01:28,  3.28s/it]2026-10-11 13:37:22.993 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 74, smoothed-train-loss = 8.125786 | smoothed-test-loss = 8.055447 | rel_loss_delta = -0.00018970007024757854 | loss_stagnation_count = 0
Epoch:  74%|███████▍  | 74/100 [04:01<01:26,  3.33s/it]2026-10-11 13:37:26.680 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 75, smoothed-train-loss = 8.120915 | smoothed-test-loss = 8.067705 | rel_loss_delta = -0.0005995495948463132 | loss_stagnation_count = 0
Epoch:  75%|███████▌  | 75/100 [04:05<01:25,  3.44s/it]2026-10-11 13:37:29.846 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 76, smoothed-train-loss = 8.123026 | smoothed-test-loss = 8.091869 | rel_loss_delta = 0.00026005636992660095 | loss_stagnation_count = 0
Epoch:  76%|███████▌  | 76/100 [04:08<01:20,  3.36s/it]2026-10-11 13:37:33.719 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 77, smoothed-train-loss = 8.123058 | smoothed-test-loss = 8.061013 | rel_loss_delta = 3.850213643176377e-06 | loss_stagnation_count = 1
Epoch:  77%|███████▋  | 77/100 [04:12<01:20,  3.51s/it]2026-10-11 13:37:36.943 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 78, smoothed-train-loss = 8.122510 | smoothed-test-loss = 8.082834 | rel_loss_delta = -6.747127674748753e-05 | loss_stagnation_count = 0
Epoch:  78%|███████▊  | 78/100 [04:15<01:15,  3.42s/it]2026-10-11 13:37:40.901 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 79, smoothed-train-loss = 8.125769 | smoothed-test-loss = 8.092966 | rel_loss_delta = 0.0004012308285587713 | loss_stagnation_count = 0
Epoch:  79%|███████▉  | 79/100 [04:19<01:15,  3.59s/it]2026-10-11 13:37:44.286 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 80, smoothed-train-loss = 8.126526 | smoothed-test-loss = 8.071681 | rel_loss_delta = 9.324184920837527e-05 | loss_stagnation_count = 0
Epoch:  80%|████████  | 80/100 [04:23<01:10,  3.53s/it]2026-10-11 13:37:47.655 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 81, smoothed-train-loss = 8.122933 | smoothed-test-loss = 8.062039 | rel_loss_delta = -0.00044212756513148236 | loss_stagnation_count = 0
Epoch:  81%|████████  | 81/100 [04:26<01:06,  3.48s/it]2026-10-11 13:37:50.833 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 82, smoothed-train-loss = 8.121243 | smoothed-test-loss = 8.051545 | rel_loss_delta = -0.00020806692636665032 | loss_stagnation_count = 0
Epoch:  82%|████████▏ | 82/100 [04:29<01:00,  3.39s/it]2026-10-11 13:37:53.920 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 83, smoothed-train-loss = 8.125158 | smoothed-test-loss = 8.059523 | rel_loss_delta = 0.0004820624028077752 | loss_stagnation_count = 0
Epoch:  83%|████████▎ | 83/100 [04:32<00:56,  3.30s/it]2026-10-11 13:37:57.416 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 84, smoothed-train-loss = 8.121944 | smoothed-test-loss = 8.051971 | rel_loss_delta = -0.00039557609597600364 | loss_stagnation_count = 0
Epoch:  84%|████████▍ | 84/100 [04:36<00:53,  3.36s/it]2026-10-11 13:38:00.680 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 85, smoothed-train-loss = 8.120209 | smoothed-test-loss = 8.058199 | rel_loss_delta = -0.00021365552975322156 | loss_stagnation_count = 0
Epoch:  85%|████████▌ | 85/100 [04:39<00:49,  3.33s/it]2026-10-11 13:38:03.696 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 86, smoothed-train-loss = 8.120157 | smoothed-test-loss = 8.062653 | rel_loss_delta = -6.387912621388432e-06 | loss_stagnation_count = 1
Epoch:  86%|████████▌ | 86/100 [04:42<00:45,  3.24s/it]2026-10-11 13:38:06.705 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 87, smoothed-train-loss = 8.118565 | smoothed-test-loss = 8.079077 | rel_loss_delta = -0.00019600436510123207 | loss_stagnation_count = 0
Epoch:  87%|████████▋ | 87/100 [04:45<00:41,  3.17s/it]2026-10-11 13:38:09.546 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 88, smoothed-train-loss = 8.121081 | smoothed-test-loss = 8.064122 | rel_loss_delta = 0.0003098586478900836 | loss_stagnation_count = 0
Epoch:  88%|████████▊ | 88/100 [04:48<00:36,  3.07s/it]2026-10-11 13:38:12.500 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 89, smoothed-train-loss = 8.121609 | smoothed-test-loss = 8.049875 | rel_loss_delta = 6.502830545876289e-05 | loss_stagnation_count = 0
Epoch:  89%|████████▉ | 89/100 [04:51<00:33,  3.03s/it]2026-10-11 13:38:15.508 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 90, smoothed-train-loss = 8.121077 | smoothed-test-loss = 8.069783 | rel_loss_delta = -6.5476967172493e-05 | loss_stagnation_count = 0
Epoch:  90%|█████████ | 90/100 [04:54<00:30,  3.03s/it]2026-10-11 13:38:18.575 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 91, smoothed-train-loss = 8.118669 | smoothed-test-loss = 8.061581 | rel_loss_delta = -0.0002966004118567499 | loss_stagnation_count = 0
Epoch:  91%|█████████ | 91/100 [04:57<00:27,  3.04s/it]2026-10-11 13:38:21.570 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 92, smoothed-train-loss = 8.118199 | smoothed-test-loss = 8.056284 | rel_loss_delta = -5.78784962612269e-05 | loss_stagnation_count = 0
Epoch:  92%|█████████▏| 92/100 [05:00<00:24,  3.03s/it]2026-10-11 13:38:24.702 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 93, smoothed-train-loss = 8.118012 | smoothed-test-loss = 8.064719 | rel_loss_delta = -2.304690078540567e-05 | loss_stagnation_count = 1
Epoch:  93%|█████████▎| 93/100 [05:03<00:21,  3.06s/it]2026-10-11 13:38:27.795 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 94, smoothed-train-loss = 8.119815 | smoothed-test-loss = 8.058982 | rel_loss_delta = 0.00022215062473039912 | loss_stagnation_count = 0
Epoch:  94%|█████████▍| 94/100 [05:06<00:18,  3.07s/it]2026-10-11 13:38:30.745 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 95, smoothed-train-loss = 8.122148 | smoothed-test-loss = 8.069089 | rel_loss_delta = 0.0002873164933805977 | loss_stagnation_count = 0
Epoch:  95%|█████████▌| 95/100 [05:09<00:15,  3.03s/it]2026-10-11 13:38:33.605 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 96, smoothed-train-loss = 8.121837 | smoothed-test-loss = 8.063456 | rel_loss_delta = -3.8252926854889515e-05 | loss_stagnation_count = 1
Epoch:  96%|█████████▌| 96/100 [05:12<00:11,  2.98s/it]2026-10-11 13:38:36.734 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 97, smoothed-train-loss = 8.119003 | smoothed-test-loss = 8.073571 | rel_loss_delta = -0.0003490062630851629 | loss_stagnation_count = 0
Epoch:  97%|█████████▋| 97/100 [05:15<00:09,  3.03s/it]2026-10-11 13:38:39.832 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 98, smoothed-train-loss = 8.119154 | smoothed-test-loss = 8.062148 | rel_loss_delta = 1.8578512306310944e-05 | loss_stagnation_count = 1
Epoch:  98%|█████████▊| 98/100 [05:18<00:06,  3.05s/it]2026-10-11 13:38:42.945 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 99, smoothed-train-loss = 8.117593 | smoothed-test-loss = 8.063669 | rel_loss_delta = -0.00019217555691084786 | loss_stagnation_count = 0
Epoch:  99%|█████████▉| 99/100 [05:21<00:03,  3.07s/it]2026-10-11 13:38:46.039 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 100, smoothed-train-loss = 8.118062 | smoothed-test-loss = 8.053883 | rel_loss_delta = 5.769947765482802e-05 | loss_stagnation_count = 0
Epoch: 100%|██████████| 100/100 [05:24<00:00,  3.25s/it]
2026-10-11 13:38:46.044 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:8.052794456481934
2026-10-11 13:38:46.045 | INFO     | __main__:<module>:287 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_4_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-11T13:33:19.292751.pkl
2026-10-11 13:38:46.045 | INFO     | __main__:<module>:288 - vt training process finished successfully.

Process finished with exit code 0

```
## Inference
### summary 
```log
2026-10-11 13:52:54.709 | INFO     | __main__:<module>:163 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: 0.0326 | Std: 0.0084
2026-10-11 13:53:41.147 | INFO     | __main__:<module>:163 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: 0.0268 | Std: 0.0034
2026-10-11 13:54:27.450 | INFO     | __main__:<module>:163 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: 0.0304 | Std: 0.0104

```
