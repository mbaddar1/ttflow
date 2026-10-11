# Desc
* Model : nn
* traj : global
* data : gaussian mixture 2d num_comp = 16

# Model and Trajectory hyperparams
Model 
```log
2026-10-11 13:59:48.206 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-11 13:59:49.018 | INFO     | __main__:<module>:240 - Created optimizer : Adam (
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
2026-10-11 13:59:49.019 | INFO     | __main__:<module>:242 - Training model: VelocityFieldRegressionNet(
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
```
# Model Training Log
```log
/home/mbaddar/Documents/mbaddar/ttflow/.venv/bin/python /home/mbaddar/Documents/mbaddar/ttflow/ttflow/training/nn/vt_trainer.py --data-snapshot-filepath data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_data_dim_2_n_train_test_eval_500000_200000_200000.pkl --vt-model nn --vt-opt adam --trajectory-model global 
2026-10-11 13:59:48.006 | INFO     | __main__:<module>:156 - args = Namespace(data_snapshot_filepath='data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_data_dim_2_n_train_test_eval_500000_200000_200000.pkl', use_gpu=True, vt_model='nn', vt_opt='adam', trajectory_model='global', rectification_linearity='linear', nn_total_num_param_updates=10001, nn_hidden_dim=128, batch_size=2048, nn_depth=2, lr=0.001, nn_reg_lambda=0.001, patience=10, min_rel_loss_delta=5e-05, dry_run=False, logs_dir='logs', output_model_dir='models/experimental')
2026-10-11 13:59:48.023 | INFO     | __main__:<module>:161 - is GPU/CUDA available = True
2026-10-11 13:59:48.023 | INFO     | __main__:<module>:164 - device = cuda
2026-10-11 13:59:48.023 | INFO     | __main__:<module>:168 - Using data snapshot: data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_data_dim_2_n_train_test_eval_500000_200000_200000.pkl
2026-10-11 13:59:48.023 | INFO     | ttflow.utils:check_system_memory_and_swap:56 - ram = 30.96, swap = 40.0 GB
2026-10-11 13:59:48.073 | INFO     | __main__:<module>:197 - data pkl size = 0.03 GB
2026-10-11 13:59:48.176 | INFO     | __main__:<module>:213 - Generating training data for the global trajectory training process..
2026-10-11 13:59:48.205 | INFO     | __main__:<module>:221 - Generating training data with xt_train_aug.shape = torch.Size([500000, 3]) x_test_aug.shape = torch.Size([200000, 3]) y_train.shape = torch.Size([500000, 2]) y_test.shape = torch.Size([200000, 2])
2026-10-11 13:59:48.206 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-11 13:59:49.018 | INFO     | __main__:<module>:240 - Created optimizer : Adam (
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
2026-10-11 13:59:49.019 | INFO     | __main__:<module>:242 - Training model: VelocityFieldRegressionNet(
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
2026-10-11 13:59:49.932 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:58 - @epoch = 0 (before training), loss_fn = MSELoss(), loss_value = 48.836697
2026-10-11 13:59:49.937 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:65 - MSELoss(test) before training:48.836700439453125
Epoch:   0%|          | 0/100 [00:00<?, ?it/s]2026-10-11 13:59:53.265 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 1, smoothed-train-loss = 14.683224 | smoothed-test-loss = 10.929641 | rel_loss_delta = inf | loss_stagnation_count = 0
Epoch:   1%|          | 1/100 [00:03<05:29,  3.33s/it]2026-10-11 13:59:56.265 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 2, smoothed-train-loss = 11.190304 | smoothed-test-loss = 10.716331 | rel_loss_delta = inf | loss_stagnation_count = 0
Epoch:   2%|▏         | 2/100 [00:06<05:07,  3.13s/it]2026-10-11 13:59:59.226 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 3, smoothed-train-loss = 10.703407 | smoothed-test-loss = 10.549850 | rel_loss_delta = -0.04351063294774079 | loss_stagnation_count = 0
Epoch:   3%|▎         | 3/100 [00:09<04:56,  3.06s/it]2026-10-11 14:00:02.301 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 4, smoothed-train-loss = 10.491399 | smoothed-test-loss = 10.363925 | rel_loss_delta = -0.019807564482257376 | loss_stagnation_count = 0
Epoch:   4%|▍         | 4/100 [00:12<04:54,  3.06s/it]2026-10-11 14:00:05.251 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 5, smoothed-train-loss = 10.272123 | smoothed-test-loss = 10.148565 | rel_loss_delta = -0.020900472790755506 | loss_stagnation_count = 0
Epoch:   5%|▌         | 5/100 [00:15<04:47,  3.02s/it]2026-10-11 14:00:08.287 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 6, smoothed-train-loss = 10.076187 | smoothed-test-loss = 9.983916 | rel_loss_delta = -0.019074589510743093 | loss_stagnation_count = 0
Epoch:   6%|▌         | 6/100 [00:18<04:44,  3.03s/it]2026-10-11 14:00:11.439 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 7, smoothed-train-loss = 9.944678 | smoothed-test-loss = 9.878910 | rel_loss_delta = -0.01305145666116692 | loss_stagnation_count = 0
Epoch:   7%|▋         | 7/100 [00:21<04:45,  3.07s/it]2026-10-11 14:00:14.498 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 8, smoothed-train-loss = 9.854810 | smoothed-test-loss = 9.811733 | rel_loss_delta = -0.009036807075887896 | loss_stagnation_count = 0
Epoch:   8%|▊         | 8/100 [00:24<04:41,  3.06s/it]2026-10-11 14:00:17.470 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 9, smoothed-train-loss = 9.795206 | smoothed-test-loss = 9.750721 | rel_loss_delta = -0.0060481554994687515 | loss_stagnation_count = 0
Epoch:   9%|▉         | 9/100 [00:27<04:36,  3.04s/it]2026-10-11 14:00:20.527 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 10, smoothed-train-loss = 9.754826 | smoothed-test-loss = 9.750379 | rel_loss_delta = -0.00412248089514909 | loss_stagnation_count = 0
Epoch:  10%|█         | 10/100 [00:30<04:33,  3.04s/it]2026-10-11 14:00:23.593 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 11, smoothed-train-loss = 9.720367 | smoothed-test-loss = 9.674500 | rel_loss_delta = -0.0035324966831655584 | loss_stagnation_count = 0
Epoch:  11%|█         | 11/100 [00:33<04:31,  3.05s/it]2026-10-11 14:00:26.812 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 12, smoothed-train-loss = 9.699600 | smoothed-test-loss = 9.682854 | rel_loss_delta = -0.002136463821413035 | loss_stagnation_count = 0
Epoch:  12%|█▏        | 12/100 [00:36<04:32,  3.10s/it]2026-10-11 14:00:29.786 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 13, smoothed-train-loss = 9.680128 | smoothed-test-loss = 9.653563 | rel_loss_delta = -0.002007459375588913 | loss_stagnation_count = 0
Epoch:  13%|█▎        | 13/100 [00:39<04:26,  3.06s/it]2026-10-11 14:00:32.648 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 14, smoothed-train-loss = 9.670369 | smoothed-test-loss = 9.645999 | rel_loss_delta = -0.0010081768593154976 | loss_stagnation_count = 0
Epoch:  14%|█▍        | 14/100 [00:42<04:18,  3.00s/it]2026-10-11 14:00:35.747 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 15, smoothed-train-loss = 9.655832 | smoothed-test-loss = 9.630586 | rel_loss_delta = -0.0015032373385950569 | loss_stagnation_count = 0
Epoch:  15%|█▌        | 15/100 [00:45<04:17,  3.03s/it]2026-10-11 14:00:38.811 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 16, smoothed-train-loss = 9.648398 | smoothed-test-loss = 9.683143 | rel_loss_delta = -0.0007698904310331431 | loss_stagnation_count = 0
Epoch:  16%|█▌        | 16/100 [00:48<04:15,  3.04s/it]2026-10-11 14:00:41.747 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 17, smoothed-train-loss = 9.644450 | smoothed-test-loss = 9.642322 | rel_loss_delta = -0.00040918527385829065 | loss_stagnation_count = 0
Epoch:  17%|█▋        | 17/100 [00:51<04:09,  3.01s/it]2026-10-11 14:00:44.582 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 18, smoothed-train-loss = 9.640494 | smoothed-test-loss = 9.611412 | rel_loss_delta = -0.00041017428603671826 | loss_stagnation_count = 0
Epoch:  18%|█▊        | 18/100 [00:54<04:02,  2.96s/it]2026-10-11 14:00:47.530 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 19, smoothed-train-loss = 9.632753 | smoothed-test-loss = 9.605931 | rel_loss_delta = -0.000803015560525052 | loss_stagnation_count = 0
Epoch:  19%|█▉        | 19/100 [00:57<03:59,  2.95s/it]2026-10-11 14:00:50.617 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 20, smoothed-train-loss = 9.627654 | smoothed-test-loss = 9.629326 | rel_loss_delta = -0.0005293324438829781 | loss_stagnation_count = 0
Epoch:  20%|██        | 20/100 [01:00<03:59,  2.99s/it]2026-10-11 14:00:53.655 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 21, smoothed-train-loss = 9.623852 | smoothed-test-loss = 9.581736 | rel_loss_delta = -0.0003948594602243457 | loss_stagnation_count = 0
Epoch:  21%|██        | 21/100 [01:03<03:57,  3.01s/it]2026-10-11 14:00:56.560 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 22, smoothed-train-loss = 9.619615 | smoothed-test-loss = 9.608270 | rel_loss_delta = -0.0004402534220195879 | loss_stagnation_count = 0
Epoch:  22%|██▏       | 22/100 [01:06<03:52,  2.98s/it]2026-10-11 14:00:59.535 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 23, smoothed-train-loss = 9.617589 | smoothed-test-loss = 9.603485 | rel_loss_delta = -0.00021060422536138546 | loss_stagnation_count = 0
Epoch:  23%|██▎       | 23/100 [01:09<03:49,  2.98s/it]2026-10-11 14:01:02.607 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 24, smoothed-train-loss = 9.611851 | smoothed-test-loss = 9.590585 | rel_loss_delta = -0.0005966474978824903 | loss_stagnation_count = 0
Epoch:  24%|██▍       | 24/100 [01:12<03:48,  3.00s/it]2026-10-11 14:01:05.740 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 25, smoothed-train-loss = 9.610739 | smoothed-test-loss = 9.619276 | rel_loss_delta = -0.00011570122215189953 | loss_stagnation_count = 0
Epoch:  25%|██▌       | 25/100 [01:15<03:48,  3.04s/it]2026-10-11 14:01:08.704 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 26, smoothed-train-loss = 9.612204 | smoothed-test-loss = 9.603006 | rel_loss_delta = 0.00015248133582928354 | loss_stagnation_count = 0
Epoch:  26%|██▌       | 26/100 [01:18<03:43,  3.02s/it]2026-10-11 14:01:11.865 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 27, smoothed-train-loss = 9.610242 | smoothed-test-loss = 9.598545 | rel_loss_delta = -0.00020413454304604872 | loss_stagnation_count = 0
Epoch:  27%|██▋       | 27/100 [01:21<03:43,  3.06s/it]2026-10-11 14:01:14.837 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 28, smoothed-train-loss = 9.604356 | smoothed-test-loss = 9.577986 | rel_loss_delta = -0.0006124426617341144 | loss_stagnation_count = 0
Epoch:  28%|██▊       | 28/100 [01:24<03:38,  3.03s/it]2026-10-11 14:01:18.251 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 29, smoothed-train-loss = 9.602157 | smoothed-test-loss = 9.636178 | rel_loss_delta = -0.0002289746537078242 | loss_stagnation_count = 0
Epoch:  29%|██▉       | 29/100 [01:28<03:43,  3.15s/it]2026-10-11 14:01:21.168 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 30, smoothed-train-loss = 9.601675 | smoothed-test-loss = 9.666785 | rel_loss_delta = -5.025981437922221e-05 | loss_stagnation_count = 0
Epoch:  30%|███       | 30/100 [01:31<03:35,  3.08s/it]2026-10-11 14:01:24.302 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 31, smoothed-train-loss = 9.606509 | smoothed-test-loss = 9.700824 | rel_loss_delta = 0.0005034340047418438 | loss_stagnation_count = 0
Epoch:  31%|███       | 31/100 [01:34<03:33,  3.10s/it]2026-10-11 14:01:27.524 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 32, smoothed-train-loss = 9.603642 | smoothed-test-loss = 9.587399 | rel_loss_delta = -0.00029844222709308506 | loss_stagnation_count = 0
Epoch:  32%|███▏      | 32/100 [01:37<03:33,  3.13s/it]2026-10-11 14:01:30.587 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 33, smoothed-train-loss = 9.598603 | smoothed-test-loss = 9.565966 | rel_loss_delta = -0.0005246384318642564 | loss_stagnation_count = 0
Epoch:  33%|███▎      | 33/100 [01:40<03:28,  3.11s/it]2026-10-11 14:01:33.403 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 34, smoothed-train-loss = 9.596430 | smoothed-test-loss = 9.625500 | rel_loss_delta = -0.00022641627923114733 | loss_stagnation_count = 0
Epoch:  34%|███▍      | 34/100 [01:43<03:19,  3.02s/it]2026-10-11 14:01:36.522 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 35, smoothed-train-loss = 9.596866 | smoothed-test-loss = 9.588155 | rel_loss_delta = 4.54553189662253e-05 | loss_stagnation_count = 1
Epoch:  35%|███▌      | 35/100 [01:46<03:18,  3.05s/it]2026-10-11 14:01:39.868 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 36, smoothed-train-loss = 9.598042 | smoothed-test-loss = 9.579359 | rel_loss_delta = 0.00012251775576216752 | loss_stagnation_count = 0
Epoch:  36%|███▌      | 36/100 [01:49<03:20,  3.14s/it]2026-10-11 14:01:43.086 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 37, smoothed-train-loss = 9.594454 | smoothed-test-loss = 9.571165 | rel_loss_delta = -0.00037381316437294285 | loss_stagnation_count = 0
Epoch:  37%|███▋      | 37/100 [01:53<03:19,  3.16s/it]2026-10-11 14:01:46.077 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 38, smoothed-train-loss = 9.593055 | smoothed-test-loss = 9.590103 | rel_loss_delta = -0.00014581955518039677 | loss_stagnation_count = 0
Epoch:  38%|███▊      | 38/100 [01:56<03:12,  3.11s/it]2026-10-11 14:01:49.083 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 39, smoothed-train-loss = 9.594072 | smoothed-test-loss = 9.603438 | rel_loss_delta = 0.0001060759572439554 | loss_stagnation_count = 0
Epoch:  39%|███▉      | 39/100 [01:59<03:07,  3.08s/it]2026-10-11 14:01:52.192 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 40, smoothed-train-loss = 9.594635 | smoothed-test-loss = 9.576229 | rel_loss_delta = 5.8669718453237905e-05 | loss_stagnation_count = 0
Epoch:  40%|████      | 40/100 [02:02<03:05,  3.09s/it]2026-10-11 14:01:55.334 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 41, smoothed-train-loss = 9.595617 | smoothed-test-loss = 9.569984 | rel_loss_delta = 0.00010229224443907612 | loss_stagnation_count = 0
Epoch:  41%|████      | 41/100 [02:05<03:03,  3.10s/it]2026-10-11 14:01:58.429 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 42, smoothed-train-loss = 9.592816 | smoothed-test-loss = 9.610853 | rel_loss_delta = -0.00029193261713076886 | loss_stagnation_count = 0
Epoch:  42%|████▏     | 42/100 [02:08<02:59,  3.10s/it]2026-10-11 14:02:01.478 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 43, smoothed-train-loss = 9.588241 | smoothed-test-loss = 9.576161 | rel_loss_delta = -0.00047684814384534946 | loss_stagnation_count = 0
Epoch:  43%|████▎     | 43/100 [02:11<02:55,  3.09s/it]2026-10-11 14:02:04.440 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 44, smoothed-train-loss = 9.588174 | smoothed-test-loss = 9.568557 | rel_loss_delta = -6.984127339125914e-06 | loss_stagnation_count = 1
Epoch:  44%|████▍     | 44/100 [02:14<02:50,  3.05s/it]2026-10-11 14:02:07.439 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 45, smoothed-train-loss = 9.588758 | smoothed-test-loss = 9.564421 | rel_loss_delta = 6.0828943227575595e-05 | loss_stagnation_count = 0
Epoch:  45%|████▌     | 45/100 [02:17<02:46,  3.03s/it]2026-10-11 14:02:10.329 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 46, smoothed-train-loss = 9.589339 | smoothed-test-loss = 9.572374 | rel_loss_delta = 6.066320128996421e-05 | loss_stagnation_count = 0
Epoch:  46%|████▌     | 46/100 [02:20<02:41,  2.99s/it]2026-10-11 14:02:13.420 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 47, smoothed-train-loss = 9.587554 | smoothed-test-loss = 9.601560 | rel_loss_delta = -0.0001861959345491647 | loss_stagnation_count = 0
Epoch:  47%|████▋     | 47/100 [02:23<02:40,  3.02s/it]2026-10-11 14:02:16.500 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 48, smoothed-train-loss = 9.589799 | smoothed-test-loss = 9.562414 | rel_loss_delta = 0.00023422036701635488 | loss_stagnation_count = 0
Epoch:  48%|████▊     | 48/100 [02:26<02:38,  3.04s/it]2026-10-11 14:02:19.486 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 49, smoothed-train-loss = 9.584139 | smoothed-test-loss = 9.558457 | rel_loss_delta = -0.0005902004719998727 | loss_stagnation_count = 0
Epoch:  49%|████▉     | 49/100 [02:29<02:34,  3.02s/it]2026-10-11 14:02:22.338 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 50, smoothed-train-loss = 9.588525 | smoothed-test-loss = 9.566340 | rel_loss_delta = 0.00045760687321750665 | loss_stagnation_count = 0
Epoch:  50%|█████     | 50/100 [02:32<02:28,  2.97s/it]2026-10-11 14:02:25.369 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 51, smoothed-train-loss = 9.582723 | smoothed-test-loss = 9.561862 | rel_loss_delta = -0.0006051123652123263 | loss_stagnation_count = 0
Epoch:  51%|█████     | 51/100 [02:35<02:26,  2.99s/it]2026-10-11 14:02:28.569 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 52, smoothed-train-loss = 9.587598 | smoothed-test-loss = 9.555493 | rel_loss_delta = 0.0005087616679498773 | loss_stagnation_count = 0
Epoch:  52%|█████▏    | 52/100 [02:38<02:26,  3.05s/it]2026-10-11 14:02:31.547 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 53, smoothed-train-loss = 9.582020 | smoothed-test-loss = 9.604594 | rel_loss_delta = -0.0005818471160852674 | loss_stagnation_count = 0
Epoch:  53%|█████▎    | 53/100 [02:41<02:22,  3.03s/it]2026-10-11 14:02:34.404 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 54, smoothed-train-loss = 9.584779 | smoothed-test-loss = 9.560636 | rel_loss_delta = 0.00028800016357637825 | loss_stagnation_count = 0
Epoch:  54%|█████▍    | 54/100 [02:44<02:16,  2.98s/it]2026-10-11 14:02:37.346 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 55, smoothed-train-loss = 9.584295 | smoothed-test-loss = 9.565886 | rel_loss_delta = -5.0557174197615104e-05 | loss_stagnation_count = 0
Epoch:  55%|█████▌    | 55/100 [02:47<02:13,  2.97s/it]2026-10-11 14:02:40.379 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 56, smoothed-train-loss = 9.581875 | smoothed-test-loss = 9.603345 | rel_loss_delta = -0.00025252804364467486 | loss_stagnation_count = 0
Epoch:  56%|█████▌    | 56/100 [02:50<02:11,  2.99s/it]2026-10-11 14:02:43.574 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 57, smoothed-train-loss = 9.587708 | smoothed-test-loss = 9.579153 | rel_loss_delta = 0.0006088248063558667 | loss_stagnation_count = 0
Epoch:  57%|█████▋    | 57/100 [02:53<02:11,  3.05s/it]2026-10-11 14:02:46.463 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 58, smoothed-train-loss = 9.579420 | smoothed-test-loss = 9.561027 | rel_loss_delta = -0.0008644523295433785 | loss_stagnation_count = 0
Epoch:  58%|█████▊    | 58/100 [02:56<02:06,  3.00s/it]2026-10-11 14:02:49.482 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 59, smoothed-train-loss = 9.581966 | smoothed-test-loss = 9.568267 | rel_loss_delta = 0.0002657650933842832 | loss_stagnation_count = 0
Epoch:  59%|█████▉    | 59/100 [02:59<02:03,  3.01s/it]2026-10-11 14:02:52.544 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 60, smoothed-train-loss = 9.585398 | smoothed-test-loss = 9.596007 | rel_loss_delta = 0.00035811922594797664 | loss_stagnation_count = 0
Epoch:  60%|██████    | 60/100 [03:02<02:00,  3.02s/it]2026-10-11 14:02:55.532 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 61, smoothed-train-loss = 9.580739 | smoothed-test-loss = 9.567807 | rel_loss_delta = -0.00048596675244871716 | loss_stagnation_count = 0
Epoch:  61%|██████    | 61/100 [03:05<01:57,  3.01s/it]2026-10-11 14:02:58.425 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 62, smoothed-train-loss = 9.580260 | smoothed-test-loss = 9.562384 | rel_loss_delta = -4.999151579649352e-05 | loss_stagnation_count = 1
Epoch:  62%|██████▏   | 62/100 [03:08<01:53,  2.98s/it]2026-10-11 14:03:01.524 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 63, smoothed-train-loss = 9.580370 | smoothed-test-loss = 9.610020 | rel_loss_delta = 1.1453138150438016e-05 | loss_stagnation_count = 2
Epoch:  63%|██████▎   | 63/100 [03:11<01:51,  3.01s/it]2026-10-11 14:03:04.596 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 64, smoothed-train-loss = 9.582445 | smoothed-test-loss = 9.562185 | rel_loss_delta = 0.00021658622620274968 | loss_stagnation_count = 0
Epoch:  64%|██████▍   | 64/100 [03:14<01:49,  3.03s/it]2026-10-11 14:03:07.621 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 65, smoothed-train-loss = 9.579777 | smoothed-test-loss = 9.564895 | rel_loss_delta = -0.0002784362304291507 | loss_stagnation_count = 0
Epoch:  65%|██████▌   | 65/100 [03:17<01:46,  3.03s/it]2026-10-11 14:03:10.621 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 66, smoothed-train-loss = 9.581654 | smoothed-test-loss = 9.556328 | rel_loss_delta = 0.00019591185466282443 | loss_stagnation_count = 0
Epoch:  66%|██████▌   | 66/100 [03:20<01:42,  3.02s/it]2026-10-11 14:03:13.692 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 67, smoothed-train-loss = 9.581927 | smoothed-test-loss = 9.579748 | rel_loss_delta = 2.8538966395863497e-05 | loss_stagnation_count = 1
Epoch:  67%|██████▋   | 67/100 [03:23<01:40,  3.04s/it]2026-10-11 14:03:16.696 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 68, smoothed-train-loss = 9.581474 | smoothed-test-loss = 9.591774 | rel_loss_delta = -4.729893009712973e-05 | loss_stagnation_count = 2
Epoch:  68%|██████▊   | 68/100 [03:26<01:36,  3.03s/it]2026-10-11 14:03:19.789 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 69, smoothed-train-loss = 9.579356 | smoothed-test-loss = 9.557883 | rel_loss_delta = -0.00022100964716926928 | loss_stagnation_count = 0
Epoch:  69%|██████▉   | 69/100 [03:29<01:34,  3.05s/it]2026-10-11 14:03:22.815 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 70, smoothed-train-loss = 9.576930 | smoothed-test-loss = 9.542922 | rel_loss_delta = -0.00025324400880814336 | loss_stagnation_count = 0
Epoch:  70%|███████   | 70/100 [03:32<01:31,  3.04s/it]2026-10-11 14:03:25.983 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 71, smoothed-train-loss = 9.578125 | smoothed-test-loss = 9.567653 | rel_loss_delta = 0.00012477214903163192 | loss_stagnation_count = 0
Epoch:  71%|███████   | 71/100 [03:36<01:29,  3.08s/it]2026-10-11 14:03:29.047 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 72, smoothed-train-loss = 9.576985 | smoothed-test-loss = 9.567185 | rel_loss_delta = -0.00011906360748625456 | loss_stagnation_count = 0
Epoch:  72%|███████▏  | 72/100 [03:39<01:26,  3.07s/it]2026-10-11 14:03:32.252 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 73, smoothed-train-loss = 9.574716 | smoothed-test-loss = 9.559297 | rel_loss_delta = -0.0002369542751510654 | loss_stagnation_count = 0
Epoch:  73%|███████▎  | 73/100 [03:42<01:24,  3.11s/it]2026-10-11 14:03:35.246 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 74, smoothed-train-loss = 9.576143 | smoothed-test-loss = 9.578068 | rel_loss_delta = 0.00014904655161482064 | loss_stagnation_count = 0
Epoch:  74%|███████▍  | 74/100 [03:45<01:20,  3.08s/it]2026-10-11 14:03:38.390 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 75, smoothed-train-loss = 9.577776 | smoothed-test-loss = 9.578694 | rel_loss_delta = 0.0001705812352785628 | loss_stagnation_count = 0
Epoch:  75%|███████▌  | 75/100 [03:48<01:17,  3.10s/it]2026-10-11 14:03:41.580 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 76, smoothed-train-loss = 9.574630 | smoothed-test-loss = 9.556119 | rel_loss_delta = -0.00032850378323852824 | loss_stagnation_count = 0
Epoch:  76%|███████▌  | 76/100 [03:51<01:15,  3.13s/it]2026-10-11 14:03:44.708 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 77, smoothed-train-loss = 9.580299 | smoothed-test-loss = 9.562258 | rel_loss_delta = 0.000592128672056093 | loss_stagnation_count = 0
Epoch:  77%|███████▋  | 77/100 [03:54<01:11,  3.13s/it]2026-10-11 14:03:47.628 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 78, smoothed-train-loss = 9.584835 | smoothed-test-loss = 9.551354 | rel_loss_delta = 0.0004734632185614088 | loss_stagnation_count = 0
Epoch:  78%|███████▊  | 78/100 [03:57<01:07,  3.06s/it]2026-10-11 14:03:50.766 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 79, smoothed-train-loss = 9.575444 | smoothed-test-loss = 9.554588 | rel_loss_delta = -0.0009797695368712802 | loss_stagnation_count = 0
Epoch:  79%|███████▉  | 79/100 [04:00<01:04,  3.09s/it]2026-10-11 14:03:53.864 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 80, smoothed-train-loss = 9.577952 | smoothed-test-loss = 9.558425 | rel_loss_delta = 0.00026187993828635217 | loss_stagnation_count = 0
Epoch:  80%|████████  | 80/100 [04:03<01:01,  3.09s/it]2026-10-11 14:03:57.122 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 81, smoothed-train-loss = 9.576151 | smoothed-test-loss = 9.567463 | rel_loss_delta = -0.0001880365538361372 | loss_stagnation_count = 0
Epoch:  81%|████████  | 81/100 [04:07<00:59,  3.14s/it]2026-10-11 14:04:00.012 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 82, smoothed-train-loss = 9.575894 | smoothed-test-loss = 9.559223 | rel_loss_delta = -2.686442246956184e-05 | loss_stagnation_count = 1
Epoch:  82%|████████▏ | 82/100 [04:10<00:55,  3.07s/it]2026-10-11 14:04:03.088 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 83, smoothed-train-loss = 9.574705 | smoothed-test-loss = 9.552400 | rel_loss_delta = -0.00012418370498321483 | loss_stagnation_count = 0
Epoch:  83%|████████▎ | 83/100 [04:13<00:52,  3.07s/it]2026-10-11 14:04:06.199 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 84, smoothed-train-loss = 9.572193 | smoothed-test-loss = 9.559906 | rel_loss_delta = -0.000262360503183131 | loss_stagnation_count = 0
Epoch:  84%|████████▍ | 84/100 [04:16<00:49,  3.08s/it]2026-10-11 14:04:09.331 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 85, smoothed-train-loss = 9.575121 | smoothed-test-loss = 9.590154 | rel_loss_delta = 0.0003059433280927055 | loss_stagnation_count = 0
Epoch:  85%|████████▌ | 85/100 [04:19<00:46,  3.10s/it]2026-10-11 14:04:12.360 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 86, smoothed-train-loss = 9.579801 | smoothed-test-loss = 9.556004 | rel_loss_delta = 0.0004887469064425172 | loss_stagnation_count = 0
Epoch:  86%|████████▌ | 86/100 [04:22<00:43,  3.08s/it]2026-10-11 14:04:15.345 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 87, smoothed-train-loss = 9.573019 | smoothed-test-loss = 9.559062 | rel_loss_delta = -0.0007078954716648411 | loss_stagnation_count = 0
Epoch:  87%|████████▋ | 87/100 [04:25<00:39,  3.05s/it]2026-10-11 14:04:18.471 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 88, smoothed-train-loss = 9.574055 | smoothed-test-loss = 9.542335 | rel_loss_delta = 0.0001081707181782813 | loss_stagnation_count = 0
Epoch:  88%|████████▊ | 88/100 [04:28<00:36,  3.07s/it]2026-10-11 14:04:21.468 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 89, smoothed-train-loss = 9.573338 | smoothed-test-loss = 9.562825 | rel_loss_delta = -7.484766444298098e-05 | loss_stagnation_count = 0
Epoch:  89%|████████▉ | 89/100 [04:31<00:33,  3.05s/it]2026-10-11 14:04:24.318 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 90, smoothed-train-loss = 9.572268 | smoothed-test-loss = 9.566947 | rel_loss_delta = -0.00011184622421253557 | loss_stagnation_count = 0
Epoch:  90%|█████████ | 90/100 [04:34<00:29,  2.99s/it]2026-10-11 14:04:27.268 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 91, smoothed-train-loss = 9.576112 | smoothed-test-loss = 9.573280 | rel_loss_delta = 0.00040160316434943895 | loss_stagnation_count = 0
Epoch:  91%|█████████ | 91/100 [04:37<00:26,  2.98s/it]2026-10-11 14:04:30.264 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 92, smoothed-train-loss = 9.573964 | smoothed-test-loss = 9.568037 | rel_loss_delta = -0.000224254743538496 | loss_stagnation_count = 0
Epoch:  92%|█████████▏| 92/100 [04:40<00:23,  2.98s/it]2026-10-11 14:04:33.496 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 93, smoothed-train-loss = 9.574789 | smoothed-test-loss = 9.540068 | rel_loss_delta = 8.611409886352714e-05 | loss_stagnation_count = 0
Epoch:  93%|█████████▎| 93/100 [04:43<00:21,  3.06s/it]2026-10-11 14:04:36.597 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 94, smoothed-train-loss = 9.576178 | smoothed-test-loss = 9.542361 | rel_loss_delta = 0.00014514051011825572 | loss_stagnation_count = 0
Epoch:  94%|█████████▍| 94/100 [04:46<00:18,  3.07s/it]2026-10-11 14:04:39.577 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 95, smoothed-train-loss = 9.573560 | smoothed-test-loss = 9.566743 | rel_loss_delta = -0.000273393056245084 | loss_stagnation_count = 0
Epoch:  95%|█████████▌| 95/100 [04:49<00:15,  3.04s/it]2026-10-11 14:04:42.545 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 96, smoothed-train-loss = 9.571541 | smoothed-test-loss = 9.606242 | rel_loss_delta = -0.00021091428798357195 | loss_stagnation_count = 0
Epoch:  96%|█████████▌| 96/100 [04:52<00:12,  3.02s/it]2026-10-11 14:04:45.498 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 97, smoothed-train-loss = 9.573930 | smoothed-test-loss = 9.563299 | rel_loss_delta = 0.0002495850476232807 | loss_stagnation_count = 0
Epoch:  97%|█████████▋| 97/100 [04:55<00:09,  3.00s/it]2026-10-11 14:04:48.329 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 98, smoothed-train-loss = 9.573496 | smoothed-test-loss = 9.560768 | rel_loss_delta = -4.533095785237431e-05 | loss_stagnation_count = 1
Epoch:  98%|█████████▊| 98/100 [04:58<00:05,  2.95s/it]2026-10-11 14:04:51.368 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 99, smoothed-train-loss = 9.570970 | smoothed-test-loss = 9.558055 | rel_loss_delta = -0.0002638968124645947 | loss_stagnation_count = 0
Epoch:  99%|█████████▉| 99/100 [05:01<00:02,  2.98s/it]2026-10-11 14:04:54.454 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 100, smoothed-train-loss = 9.572561 | smoothed-test-loss = 9.544162 | rel_loss_delta = 0.0001662577892403776 | loss_stagnation_count = 0
Epoch: 100%|██████████| 100/100 [05:04<00:00,  3.05s/it]
2026-10-11 14:04:54.459 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:9.542616844177246
2026-10-11 14:04:54.460 | INFO     | __main__:<module>:287 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-11T13:59:48.006131.pkl
2026-10-11 14:04:54.460 | INFO     | __main__:<module>:288 - vt training process finished successfully.

Process finished with exit code 0

```

# Inference
```log
2026-10-11 14:36:34.963 | INFO     | __main__:<module>:163 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: 0.1178 | Std: 0.0149 |Inference time for total number of inference samples: 22.246977 seconds
2026-10-11 14:37:18.544 | INFO     | __main__:<module>:163 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: 0.1162 | Std: 0.0136 |Inference time for total number of inference samples: 22.266585 seconds
2026-10-11 14:38:03.978 | INFO     | __main__:<module>:163 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: 0.1123 | Std: 0.0140 |Inference time for total number of inference samples: 22.539382 seconds

```