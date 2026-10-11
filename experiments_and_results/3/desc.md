# Desc
* Model vt
* traj  global
* data skewed gaussian 2d
# Model and Trajectory Hyperparameters
```log
Network is on: cuda:0
2026-10-11 14:45:45.137 | INFO     | __main__:<module>:240 - Created optimizer : Adam (
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
2026-10-11 14:45:45.137 | INFO     | __main__:<module>:242 - Training model: VelocityFieldRegressionNet(
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
/home/mbaddar/Documents/mbaddar/ttflow/.venv/bin/python /home/mbaddar/Documents/mbaddar/ttflow/ttflow/training/nn/vt_trainer.py --data-snapshot-filepath data-snapshots/experimental/data_snapshot_skewed_gaussian_2d_mean_[0.0, 0.0]_cov_[[1.0, 0.0], [0.0, 1.0]]_shape_[5.0, 5.0]_n_train_500000_n_test_200000_n_eval_200000.pkl --vt-model nn --vt-opt adam --trajectory-model global 
2026-10-11 14:45:44.172 | INFO     | __main__:<module>:156 - args = Namespace(data_snapshot_filepath='data-snapshots/experimental/data_snapshot_skewed_gaussian_2d_mean_[0.0, 0.0]_cov_[[1.0, 0.0], [0.0, 1.0]]_shape_[5.0, 5.0]_n_train_500000_n_test_200000_n_eval_200000.pkl', use_gpu=True, vt_model='nn', vt_opt='adam', trajectory_model='global', rectification_linearity='linear', nn_total_num_param_updates=10001, nn_hidden_dim=128, batch_size=2048, nn_depth=2, lr=0.001, nn_reg_lambda=0.001, patience=10, min_rel_loss_delta=5e-05, dry_run=False, logs_dir='logs', output_model_dir='models/experimental')
2026-10-11 14:45:44.193 | INFO     | __main__:<module>:161 - is GPU/CUDA available = True
2026-10-11 14:45:44.193 | INFO     | __main__:<module>:164 - device = cuda
2026-10-11 14:45:44.193 | INFO     | __main__:<module>:168 - Using data snapshot: data-snapshots/experimental/data_snapshot_skewed_gaussian_2d_mean_[0.0, 0.0]_cov_[[1.0, 0.0], [0.0, 1.0]]_shape_[5.0, 5.0]_n_train_500000_n_test_200000_n_eval_200000.pkl
2026-10-11 14:45:44.194 | INFO     | ttflow.utils:check_system_memory_and_swap:56 - ram = 30.96, swap = 40.0 GB
2026-10-11 14:45:44.238 | INFO     | __main__:<module>:197 - data pkl size = 0.03 GB
2026-10-11 14:45:44.359 | INFO     | __main__:<module>:213 - Generating training data for the global trajectory training process..
2026-10-11 14:45:44.394 | INFO     | __main__:<module>:221 - Generating training data with xt_train_aug.shape = torch.Size([500000, 3]) x_test_aug.shape = torch.Size([200000, 3]) y_train.shape = torch.Size([500000, 2]) y_test.shape = torch.Size([200000, 2])
2026-10-11 14:45:44.395 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-11 14:45:45.137 | INFO     | __main__:<module>:240 - Created optimizer : Adam (
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
2026-10-11 14:45:45.137 | INFO     | __main__:<module>:242 - Training model: VelocityFieldRegressionNet(
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
2026-10-11 14:45:45.946 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:58 - @epoch = 0 (before training), loss_fn = MSELoss(), loss_value = 2.031158
2026-10-11 14:45:45.950 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:65 - MSELoss(test) before training:2.031157970428467
Epoch:   0%|          | 0/100 [00:00<?, ?it/s]2026-10-11 14:45:49.141 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 1, smoothed-train-loss = 1.340929 | smoothed-test-loss = 1.261398 | rel_loss_delta = inf | loss_stagnation_count = 0
Epoch:   1%|          | 1/100 [00:03<05:15,  3.19s/it]2026-10-11 14:45:52.310 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 2, smoothed-train-loss = 1.270107 | smoothed-test-loss = 1.257780 | rel_loss_delta = inf | loss_stagnation_count = 0
Epoch:   2%|▏         | 2/100 [00:06<05:11,  3.18s/it]2026-10-11 14:45:55.405 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 3, smoothed-train-loss = 1.260785 | smoothed-test-loss = 1.255644 | rel_loss_delta = -0.007339562438654612 | loss_stagnation_count = 0
Epoch:   3%|▎         | 3/100 [00:09<05:04,  3.14s/it]2026-10-11 14:45:58.302 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 4, smoothed-train-loss = 1.259013 | smoothed-test-loss = 1.257272 | rel_loss_delta = -0.0014055177282230986 | loss_stagnation_count = 0
Epoch:   4%|▍         | 4/100 [00:12<04:52,  3.04s/it]2026-10-11 14:46:01.256 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 5, smoothed-train-loss = 1.258813 | smoothed-test-loss = 1.256757 | rel_loss_delta = -0.00015928476829610245 | loss_stagnation_count = 0
Epoch:   5%|▌         | 5/100 [00:15<04:46,  3.01s/it]2026-10-11 14:46:04.308 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 6, smoothed-train-loss = 1.258098 | smoothed-test-loss = 1.255851 | rel_loss_delta = -0.0005677850622470984 | loss_stagnation_count = 0
Epoch:   6%|▌         | 6/100 [00:18<04:44,  3.03s/it]2026-10-11 14:46:07.245 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 7, smoothed-train-loss = 1.257848 | smoothed-test-loss = 1.255129 | rel_loss_delta = -0.00019843733004927232 | loss_stagnation_count = 0
Epoch:   7%|▋         | 7/100 [00:21<04:38,  3.00s/it]2026-10-11 14:46:10.198 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 8, smoothed-train-loss = 1.257890 | smoothed-test-loss = 1.255252 | rel_loss_delta = 3.3232285805568925e-05 | loss_stagnation_count = 1
Epoch:   8%|▊         | 8/100 [00:24<04:34,  2.98s/it]2026-10-11 14:46:13.204 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 9, smoothed-train-loss = 1.257501 | smoothed-test-loss = 1.256694 | rel_loss_delta = -0.00030944513165287083 | loss_stagnation_count = 0
Epoch:   9%|▉         | 9/100 [00:27<04:32,  2.99s/it]2026-10-11 14:46:16.253 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 10, smoothed-train-loss = 1.257457 | smoothed-test-loss = 1.254818 | rel_loss_delta = -3.4576836193355906e-05 | loss_stagnation_count = 1
Epoch:  10%|█         | 10/100 [00:30<04:30,  3.01s/it]2026-10-11 14:46:19.294 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 11, smoothed-train-loss = 1.257337 | smoothed-test-loss = 1.254523 | rel_loss_delta = -9.582591483597017e-05 | loss_stagnation_count = 0
Epoch:  11%|█         | 11/100 [00:33<04:28,  3.02s/it]2026-10-11 14:46:22.263 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 12, smoothed-train-loss = 1.257223 | smoothed-test-loss = 1.253533 | rel_loss_delta = -9.03273652234984e-05 | loss_stagnation_count = 0
Epoch:  12%|█▏        | 12/100 [00:36<04:24,  3.00s/it]2026-10-11 14:46:25.358 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 13, smoothed-train-loss = 1.257175 | smoothed-test-loss = 1.254888 | rel_loss_delta = -3.871555703537949e-05 | loss_stagnation_count = 1
Epoch:  13%|█▎        | 13/100 [00:39<04:23,  3.03s/it]2026-10-11 14:46:28.415 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 14, smoothed-train-loss = 1.257356 | smoothed-test-loss = 1.254427 | rel_loss_delta = 0.00014423914090276675 | loss_stagnation_count = 0
Epoch:  14%|█▍        | 14/100 [00:42<04:21,  3.04s/it]2026-10-11 14:46:31.561 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 15, smoothed-train-loss = 1.257075 | smoothed-test-loss = 1.255295 | rel_loss_delta = -0.00022327273106634169 | loss_stagnation_count = 0
Epoch:  15%|█▌        | 15/100 [00:45<04:21,  3.07s/it]2026-10-11 14:46:34.487 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 16, smoothed-train-loss = 1.257151 | smoothed-test-loss = 1.254282 | rel_loss_delta = 5.9924625970420915e-05 | loss_stagnation_count = 0
Epoch:  16%|█▌        | 16/100 [00:48<04:14,  3.03s/it]2026-10-11 14:46:37.497 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 17, smoothed-train-loss = 1.257052 | smoothed-test-loss = 1.253634 | rel_loss_delta = -7.815428120510098e-05 | loss_stagnation_count = 0
Epoch:  17%|█▋        | 17/100 [00:51<04:10,  3.02s/it]2026-10-11 14:46:40.748 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 18, smoothed-train-loss = 1.256861 | smoothed-test-loss = 1.253962 | rel_loss_delta = -0.0001521020388677184 | loss_stagnation_count = 0
Epoch:  18%|█▊        | 18/100 [00:54<04:13,  3.09s/it]2026-10-11 14:46:43.832 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 19, smoothed-train-loss = 1.256806 | smoothed-test-loss = 1.253907 | rel_loss_delta = -4.355864268145726e-05 | loss_stagnation_count = 1
Epoch:  19%|█▉        | 19/100 [00:57<04:10,  3.09s/it]2026-10-11 14:46:46.728 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 20, smoothed-train-loss = 1.257050 | smoothed-test-loss = 1.253537 | rel_loss_delta = 0.00019366694805404426 | loss_stagnation_count = 0
Epoch:  20%|██        | 20/100 [01:00<04:02,  3.03s/it]2026-10-11 14:46:49.741 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 21, smoothed-train-loss = 1.256861 | smoothed-test-loss = 1.254611 | rel_loss_delta = -0.00015016767927110726 | loss_stagnation_count = 0
Epoch:  21%|██        | 21/100 [01:03<03:59,  3.03s/it]2026-10-11 14:46:52.731 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 22, smoothed-train-loss = 1.256787 | smoothed-test-loss = 1.253319 | rel_loss_delta = -5.864023303633602e-05 | loss_stagnation_count = 0
Epoch:  22%|██▏       | 22/100 [01:06<03:55,  3.01s/it]2026-10-11 14:46:55.672 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 23, smoothed-train-loss = 1.256761 | smoothed-test-loss = 1.253351 | rel_loss_delta = -2.0869585192113384e-05 | loss_stagnation_count = 1
Epoch:  23%|██▎       | 23/100 [01:09<03:50,  2.99s/it]2026-10-11 14:46:58.535 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 24, smoothed-train-loss = 1.256734 | smoothed-test-loss = 1.254270 | rel_loss_delta = -2.1759670631515706e-05 | loss_stagnation_count = 2
Epoch:  24%|██▍       | 24/100 [01:12<03:44,  2.95s/it]2026-10-11 14:47:01.510 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 25, smoothed-train-loss = 1.256705 | smoothed-test-loss = 1.253552 | rel_loss_delta = -2.2824031835143578e-05 | loss_stagnation_count = 3
Epoch:  25%|██▌       | 25/100 [01:15<03:42,  2.96s/it]2026-10-11 14:47:04.506 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 26, smoothed-train-loss = 1.256699 | smoothed-test-loss = 1.256133 | rel_loss_delta = -4.794998254074466e-06 | loss_stagnation_count = 4
Epoch:  26%|██▌       | 26/100 [01:18<03:39,  2.97s/it]2026-10-11 14:47:07.555 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 27, smoothed-train-loss = 1.256778 | smoothed-test-loss = 1.254206 | rel_loss_delta = 6.251963728755608e-05 | loss_stagnation_count = 0
Epoch:  27%|██▋       | 27/100 [01:21<03:38,  2.99s/it]2026-10-11 14:47:10.431 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 28, smoothed-train-loss = 1.256703 | smoothed-test-loss = 1.253409 | rel_loss_delta = -5.937155259482169e-05 | loss_stagnation_count = 0
Epoch:  28%|██▊       | 28/100 [01:24<03:33,  2.96s/it]2026-10-11 14:47:13.469 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 29, smoothed-train-loss = 1.256650 | smoothed-test-loss = 1.253546 | rel_loss_delta = -4.2116655371328754e-05 | loss_stagnation_count = 1
Epoch:  29%|██▉       | 29/100 [01:27<03:31,  2.98s/it]2026-10-11 14:47:16.388 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 30, smoothed-train-loss = 1.256442 | smoothed-test-loss = 1.253117 | rel_loss_delta = -0.00016572518290258777 | loss_stagnation_count = 0
Epoch:  30%|███       | 30/100 [01:30<03:27,  2.96s/it]2026-10-11 14:47:19.421 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 31, smoothed-train-loss = 1.256692 | smoothed-test-loss = 1.254177 | rel_loss_delta = 0.00019894895990302881 | loss_stagnation_count = 0
Epoch:  31%|███       | 31/100 [01:33<03:25,  2.98s/it]2026-10-11 14:47:22.385 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 32, smoothed-train-loss = 1.256742 | smoothed-test-loss = 1.253515 | rel_loss_delta = 3.962607408322636e-05 | loss_stagnation_count = 1
Epoch:  32%|███▏      | 32/100 [01:36<03:22,  2.98s/it]2026-10-11 14:47:25.334 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 33, smoothed-train-loss = 1.256636 | smoothed-test-loss = 1.253467 | rel_loss_delta = -8.377035102341737e-05 | loss_stagnation_count = 0
Epoch:  33%|███▎      | 33/100 [01:39<03:18,  2.97s/it]2026-10-11 14:47:28.421 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 34, smoothed-train-loss = 1.256705 | smoothed-test-loss = 1.253515 | rel_loss_delta = 5.42588159753088e-05 | loss_stagnation_count = 0
Epoch:  34%|███▍      | 34/100 [01:42<03:18,  3.00s/it]2026-10-11 14:47:31.424 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 35, smoothed-train-loss = 1.256816 | smoothed-test-loss = 1.253863 | rel_loss_delta = 8.85957008899919e-05 | loss_stagnation_count = 0
Epoch:  35%|███▌      | 35/100 [01:45<03:15,  3.00s/it]2026-10-11 14:47:34.347 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 36, smoothed-train-loss = 1.256478 | smoothed-test-loss = 1.253284 | rel_loss_delta = -0.000269013428038858 | loss_stagnation_count = 0
Epoch:  36%|███▌      | 36/100 [01:48<03:10,  2.98s/it]2026-10-11 14:47:37.437 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 37, smoothed-train-loss = 1.256501 | smoothed-test-loss = 1.253540 | rel_loss_delta = 1.840697087485642e-05 | loss_stagnation_count = 1
Epoch:  37%|███▋      | 37/100 [01:51<03:09,  3.01s/it]2026-10-11 14:47:40.472 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 38, smoothed-train-loss = 1.256372 | smoothed-test-loss = 1.253975 | rel_loss_delta = -0.0001028370769086508 | loss_stagnation_count = 0
Epoch:  38%|███▊      | 38/100 [01:54<03:07,  3.02s/it]2026-10-11 14:47:43.487 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 39, smoothed-train-loss = 1.256603 | smoothed-test-loss = 1.252879 | rel_loss_delta = 0.00018414561437962048 | loss_stagnation_count = 0
Epoch:  39%|███▉      | 39/100 [01:57<03:04,  3.02s/it]2026-10-11 14:47:46.367 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 40, smoothed-train-loss = 1.256590 | smoothed-test-loss = 1.252859 | rel_loss_delta = -1.0575734341870442e-05 | loss_stagnation_count = 1
Epoch:  40%|████      | 40/100 [02:00<02:58,  2.98s/it]2026-10-11 14:47:49.431 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 41, smoothed-train-loss = 1.256373 | smoothed-test-loss = 1.253723 | rel_loss_delta = -0.00017256775285650055 | loss_stagnation_count = 0
Epoch:  41%|████      | 41/100 [02:03<02:57,  3.00s/it]2026-10-11 14:47:52.490 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 42, smoothed-train-loss = 1.256246 | smoothed-test-loss = 1.253404 | rel_loss_delta = -0.00010110352511407188 | loss_stagnation_count = 0
Epoch:  42%|████▏     | 42/100 [02:06<02:55,  3.02s/it]2026-10-11 14:47:55.454 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 43, smoothed-train-loss = 1.256303 | smoothed-test-loss = 1.253315 | rel_loss_delta = 4.571326559912002e-05 | loss_stagnation_count = 1
Epoch:  43%|████▎     | 43/100 [02:09<02:51,  3.00s/it]2026-10-11 14:47:58.376 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 44, smoothed-train-loss = 1.256459 | smoothed-test-loss = 1.253980 | rel_loss_delta = 0.00012366470392386622 | loss_stagnation_count = 0
Epoch:  44%|████▍     | 44/100 [02:12<02:46,  2.98s/it]2026-10-11 14:48:01.338 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 45, smoothed-train-loss = 1.256550 | smoothed-test-loss = 1.253618 | rel_loss_delta = 7.240088201198451e-05 | loss_stagnation_count = 0
Epoch:  45%|████▌     | 45/100 [02:15<02:43,  2.97s/it]2026-10-11 14:48:04.351 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 46, smoothed-train-loss = 1.256304 | smoothed-test-loss = 1.253290 | rel_loss_delta = -0.00019570257284251945 | loss_stagnation_count = 0
Epoch:  46%|████▌     | 46/100 [02:18<02:41,  2.99s/it]2026-10-11 14:48:07.413 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 47, smoothed-train-loss = 1.256457 | smoothed-test-loss = 1.253176 | rel_loss_delta = 0.0001218934379328849 | loss_stagnation_count = 0
Epoch:  47%|████▋     | 47/100 [02:21<02:39,  3.01s/it]2026-10-11 14:48:10.473 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 48, smoothed-train-loss = 1.256371 | smoothed-test-loss = 1.253290 | rel_loss_delta = -6.836661156419579e-05 | loss_stagnation_count = 0
Epoch:  48%|████▊     | 48/100 [02:24<02:37,  3.02s/it]2026-10-11 14:48:13.459 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 49, smoothed-train-loss = 1.256358 | smoothed-test-loss = 1.253493 | rel_loss_delta = -1.0472340369291207e-05 | loss_stagnation_count = 1
Epoch:  49%|████▉     | 49/100 [02:27<02:33,  3.01s/it]2026-10-11 14:48:16.522 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 50, smoothed-train-loss = 1.256404 | smoothed-test-loss = 1.253075 | rel_loss_delta = 3.7069156617624814e-05 | loss_stagnation_count = 2
Epoch:  50%|█████     | 50/100 [02:30<02:31,  3.03s/it]2026-10-11 14:48:19.586 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 51, smoothed-train-loss = 1.256540 | smoothed-test-loss = 1.252709 | rel_loss_delta = 0.00010798699044626331 | loss_stagnation_count = 0
Epoch:  51%|█████     | 51/100 [02:33<02:28,  3.04s/it]2026-10-11 14:48:22.568 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 52, smoothed-train-loss = 1.256286 | smoothed-test-loss = 1.253506 | rel_loss_delta = -0.00020192973233245607 | loss_stagnation_count = 0
Epoch:  52%|█████▏    | 52/100 [02:36<02:25,  3.02s/it]2026-10-11 14:48:25.567 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 53, smoothed-train-loss = 1.256222 | smoothed-test-loss = 1.254160 | rel_loss_delta = -5.100125113832128e-05 | loss_stagnation_count = 0
Epoch:  53%|█████▎    | 53/100 [02:39<02:21,  3.01s/it]2026-10-11 14:48:28.550 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 54, smoothed-train-loss = 1.256258 | smoothed-test-loss = 1.253247 | rel_loss_delta = 2.8534088509640323e-05 | loss_stagnation_count = 1
Epoch:  54%|█████▍    | 54/100 [02:42<02:18,  3.01s/it]2026-10-11 14:48:31.565 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 55, smoothed-train-loss = 1.256236 | smoothed-test-loss = 1.253485 | rel_loss_delta = -1.7545091592810046e-05 | loss_stagnation_count = 2
Epoch:  55%|█████▌    | 55/100 [02:45<02:15,  3.01s/it]2026-10-11 14:48:34.507 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 56, smoothed-train-loss = 1.256146 | smoothed-test-loss = 1.252970 | rel_loss_delta = -7.144602027084099e-05 | loss_stagnation_count = 0
Epoch:  56%|█████▌    | 56/100 [02:48<02:11,  2.99s/it]2026-10-11 14:48:37.611 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 57, smoothed-train-loss = 1.256408 | smoothed-test-loss = 1.252799 | rel_loss_delta = 0.0002086624256325186 | loss_stagnation_count = 0
Epoch:  57%|█████▋    | 57/100 [02:51<02:09,  3.02s/it]2026-10-11 14:48:40.661 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 58, smoothed-train-loss = 1.256335 | smoothed-test-loss = 1.253036 | rel_loss_delta = -5.8462230749950076e-05 | loss_stagnation_count = 0
Epoch:  58%|█████▊    | 58/100 [02:54<02:07,  3.03s/it]2026-10-11 14:48:43.703 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 59, smoothed-train-loss = 1.256163 | smoothed-test-loss = 1.252936 | rel_loss_delta = -0.00013686807044563314 | loss_stagnation_count = 0
Epoch:  59%|█████▉    | 59/100 [02:57<02:04,  3.03s/it]2026-10-11 14:48:46.615 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 60, smoothed-train-loss = 1.256228 | smoothed-test-loss = 1.253312 | rel_loss_delta = 5.1445964150257793e-05 | loss_stagnation_count = 0
Epoch:  60%|██████    | 60/100 [03:00<01:59,  3.00s/it]2026-10-11 14:48:49.643 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 61, smoothed-train-loss = 1.256199 | smoothed-test-loss = 1.253001 | rel_loss_delta = -2.274792679491175e-05 | loss_stagnation_count = 1
Epoch:  61%|██████    | 61/100 [03:03<01:57,  3.01s/it]2026-10-11 14:48:52.631 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 62, smoothed-train-loss = 1.256110 | smoothed-test-loss = 1.253270 | rel_loss_delta = -7.074078855155736e-05 | loss_stagnation_count = 0
Epoch:  62%|██████▏   | 62/100 [03:06<01:54,  3.00s/it]2026-10-11 14:48:55.805 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 63, smoothed-train-loss = 1.256309 | smoothed-test-loss = 1.253010 | rel_loss_delta = 0.0001580589966416326 | loss_stagnation_count = 0
Epoch:  63%|██████▎   | 63/100 [03:09<01:52,  3.05s/it]2026-10-11 14:48:58.852 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 64, smoothed-train-loss = 1.256391 | smoothed-test-loss = 1.253341 | rel_loss_delta = 6.525093017776817e-05 | loss_stagnation_count = 0
Epoch:  64%|██████▍   | 64/100 [03:12<01:49,  3.05s/it]2026-10-11 14:49:01.840 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 65, smoothed-train-loss = 1.256180 | smoothed-test-loss = 1.254329 | rel_loss_delta = -0.00016765781359798332 | loss_stagnation_count = 0
Epoch:  65%|██████▌   | 65/100 [03:15<01:46,  3.03s/it]2026-10-11 14:49:04.811 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 66, smoothed-train-loss = 1.256296 | smoothed-test-loss = 1.253060 | rel_loss_delta = 9.218293117478702e-05 | loss_stagnation_count = 0
Epoch:  66%|██████▌   | 66/100 [03:18<01:42,  3.01s/it]2026-10-11 14:49:07.848 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 67, smoothed-train-loss = 1.256069 | smoothed-test-loss = 1.253127 | rel_loss_delta = -0.0001802123507641614 | loss_stagnation_count = 0
Epoch:  67%|██████▋   | 67/100 [03:21<01:39,  3.02s/it]2026-10-11 14:49:10.830 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 68, smoothed-train-loss = 1.256316 | smoothed-test-loss = 1.253286 | rel_loss_delta = 0.00019633578925365655 | loss_stagnation_count = 0
Epoch:  68%|██████▊   | 68/100 [03:24<01:36,  3.01s/it]2026-10-11 14:49:13.817 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 69, smoothed-train-loss = 1.256160 | smoothed-test-loss = 1.253226 | rel_loss_delta = -0.0001244580102595432 | loss_stagnation_count = 0
Epoch:  69%|██████▉   | 69/100 [03:27<01:33,  3.00s/it]2026-10-11 14:49:16.754 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 70, smoothed-train-loss = 1.256229 | smoothed-test-loss = 1.253140 | rel_loss_delta = 5.514657910188609e-05 | loss_stagnation_count = 0
Epoch:  70%|███████   | 70/100 [03:30<01:29,  2.98s/it]2026-10-11 14:49:19.941 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 71, smoothed-train-loss = 1.256144 | smoothed-test-loss = 1.252733 | rel_loss_delta = -6.792491948223689e-05 | loss_stagnation_count = 0
Epoch:  71%|███████   | 71/100 [03:33<01:28,  3.04s/it]2026-10-11 14:49:22.807 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 72, smoothed-train-loss = 1.256146 | smoothed-test-loss = 1.253383 | rel_loss_delta = 1.6832189264422834e-06 | loss_stagnation_count = 1
Epoch:  72%|███████▏  | 72/100 [03:36<01:23,  2.99s/it]2026-10-11 14:49:25.735 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 73, smoothed-train-loss = 1.256245 | smoothed-test-loss = 1.253244 | rel_loss_delta = 7.883071009540437e-05 | loss_stagnation_count = 0
Epoch:  73%|███████▎  | 73/100 [03:39<01:20,  2.97s/it]2026-10-11 14:49:28.793 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 74, smoothed-train-loss = 1.256084 | smoothed-test-loss = 1.253268 | rel_loss_delta = -0.0001279440775567805 | loss_stagnation_count = 0
Epoch:  74%|███████▍  | 74/100 [03:42<01:17,  3.00s/it]2026-10-11 14:49:31.760 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 75, smoothed-train-loss = 1.256060 | smoothed-test-loss = 1.253844 | rel_loss_delta = -1.8934808001440382e-05 | loss_stagnation_count = 1
Epoch:  75%|███████▌  | 75/100 [03:45<01:14,  2.99s/it]2026-10-11 14:49:34.695 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 76, smoothed-train-loss = 1.256099 | smoothed-test-loss = 1.253103 | rel_loss_delta = 3.060592196378298e-05 | loss_stagnation_count = 2
Epoch:  76%|███████▌  | 76/100 [03:48<01:11,  2.97s/it]2026-10-11 14:49:37.642 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 77, smoothed-train-loss = 1.256089 | smoothed-test-loss = 1.254015 | rel_loss_delta = -7.985824598238169e-06 | loss_stagnation_count = 3
Epoch:  77%|███████▋  | 77/100 [03:51<01:08,  2.96s/it]2026-10-11 14:49:40.588 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 78, smoothed-train-loss = 1.256238 | smoothed-test-loss = 1.253320 | rel_loss_delta = 0.00011857292924438129 | loss_stagnation_count = 0
Epoch:  78%|███████▊  | 78/100 [03:54<01:05,  2.96s/it]2026-10-11 14:49:43.656 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 79, smoothed-train-loss = 1.256321 | smoothed-test-loss = 1.253333 | rel_loss_delta = 6.619485576746146e-05 | loss_stagnation_count = 0
Epoch:  79%|███████▉  | 79/100 [03:57<01:02,  2.99s/it]2026-10-11 14:49:46.526 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 80, smoothed-train-loss = 1.256165 | smoothed-test-loss = 1.253204 | rel_loss_delta = -0.00012423733046588108 | loss_stagnation_count = 0
Epoch:  80%|████████  | 80/100 [04:00<00:59,  2.96s/it]2026-10-11 14:49:49.601 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 81, smoothed-train-loss = 1.256029 | smoothed-test-loss = 1.253060 | rel_loss_delta = -0.00010783595149026096 | loss_stagnation_count = 0
Epoch:  81%|████████  | 81/100 [04:03<00:56,  2.99s/it]2026-10-11 14:49:52.594 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 82, smoothed-train-loss = 1.255999 | smoothed-test-loss = 1.253357 | rel_loss_delta = -2.3859895307249927e-05 | loss_stagnation_count = 1
Epoch:  82%|████████▏ | 82/100 [04:06<00:53,  2.99s/it]2026-10-11 14:49:55.631 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 83, smoothed-train-loss = 1.255958 | smoothed-test-loss = 1.253149 | rel_loss_delta = -3.277147802649451e-05 | loss_stagnation_count = 2
Epoch:  83%|████████▎ | 83/100 [04:09<00:51,  3.01s/it]2026-10-11 14:49:58.630 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 84, smoothed-train-loss = 1.255941 | smoothed-test-loss = 1.253970 | rel_loss_delta = -1.3331212368510786e-05 | loss_stagnation_count = 3
Epoch:  84%|████████▍ | 84/100 [04:12<00:48,  3.00s/it]2026-10-11 14:50:01.568 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 85, smoothed-train-loss = 1.256048 | smoothed-test-loss = 1.252868 | rel_loss_delta = 8.45109979299345e-05 | loss_stagnation_count = 0
Epoch:  85%|████████▌ | 85/100 [04:15<00:44,  2.98s/it]2026-10-11 14:50:04.531 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 86, smoothed-train-loss = 1.256068 | smoothed-test-loss = 1.252522 | rel_loss_delta = 1.612286146476472e-05 | loss_stagnation_count = 1
Epoch:  86%|████████▌ | 86/100 [04:18<00:41,  2.98s/it]2026-10-11 14:50:07.556 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 87, smoothed-train-loss = 1.255948 | smoothed-test-loss = 1.253312 | rel_loss_delta = -9.571721199864343e-05 | loss_stagnation_count = 0
Epoch:  87%|████████▋ | 87/100 [04:21<00:38,  2.99s/it]2026-10-11 14:50:10.431 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 88, smoothed-train-loss = 1.256060 | smoothed-test-loss = 1.253074 | rel_loss_delta = 8.951759730595912e-05 | loss_stagnation_count = 0
Epoch:  88%|████████▊ | 88/100 [04:24<00:35,  2.96s/it]2026-10-11 14:50:13.490 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 89, smoothed-train-loss = 1.255977 | smoothed-test-loss = 1.252949 | rel_loss_delta = -6.590204672671383e-05 | loss_stagnation_count = 0
Epoch:  89%|████████▉ | 89/100 [04:27<00:32,  2.99s/it]2026-10-11 14:50:16.564 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 90, smoothed-train-loss = 1.256144 | smoothed-test-loss = 1.252820 | rel_loss_delta = 0.0001325171320957315 | loss_stagnation_count = 0
Epoch:  90%|█████████ | 90/100 [04:30<00:30,  3.01s/it]2026-10-11 14:50:19.608 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 91, smoothed-train-loss = 1.256009 | smoothed-test-loss = 1.253204 | rel_loss_delta = -0.00010732340806582183 | loss_stagnation_count = 0
Epoch:  91%|█████████ | 91/100 [04:33<00:27,  3.02s/it]2026-10-11 14:50:22.500 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 92, smoothed-train-loss = 1.256047 | smoothed-test-loss = 1.253655 | rel_loss_delta = 3.0080981977662348e-05 | loss_stagnation_count = 1
Epoch:  92%|█████████▏| 92/100 [04:36<00:23,  2.98s/it]2026-10-11 14:50:25.506 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 93, smoothed-train-loss = 1.256097 | smoothed-test-loss = 1.252885 | rel_loss_delta = 4.002802892593491e-05 | loss_stagnation_count = 2
Epoch:  93%|█████████▎| 93/100 [04:39<00:20,  2.99s/it]2026-10-11 14:50:28.523 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 94, smoothed-train-loss = 1.256105 | smoothed-test-loss = 1.253706 | rel_loss_delta = 6.328835734728102e-06 | loss_stagnation_count = 3
Epoch:  94%|█████████▍| 94/100 [04:42<00:17,  3.00s/it]2026-10-11 14:50:31.886 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 95, smoothed-train-loss = 1.256000 | smoothed-test-loss = 1.252955 | rel_loss_delta = -8.383033249439557e-05 | loss_stagnation_count = 0
Epoch:  95%|█████████▌| 95/100 [04:45<00:15,  3.11s/it]2026-10-11 14:50:34.868 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 96, smoothed-train-loss = 1.256027 | smoothed-test-loss = 1.253198 | rel_loss_delta = 2.223005392580094e-05 | loss_stagnation_count = 1
Epoch:  96%|█████████▌| 96/100 [04:48<00:12,  3.07s/it]2026-10-11 14:50:37.842 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 97, smoothed-train-loss = 1.255941 | smoothed-test-loss = 1.252809 | rel_loss_delta = -6.894117893458976e-05 | loss_stagnation_count = 0
Epoch:  97%|█████████▋| 97/100 [04:51<00:09,  3.04s/it]2026-10-11 14:50:40.775 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 98, smoothed-train-loss = 1.256080 | smoothed-test-loss = 1.252940 | rel_loss_delta = 0.00011044972264770726 | loss_stagnation_count = 0
Epoch:  98%|█████████▊| 98/100 [04:54<00:06,  3.01s/it]2026-10-11 14:50:43.707 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 99, smoothed-train-loss = 1.256011 | smoothed-test-loss = 1.253000 | rel_loss_delta = -5.477545779987875e-05 | loss_stagnation_count = 0
Epoch:  99%|█████████▉| 99/100 [04:57<00:02,  2.99s/it]2026-10-11 14:50:46.630 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:119 - @epoch = 100, smoothed-train-loss = 1.256063 | smoothed-test-loss = 1.252816 | rel_loss_delta = 4.179993259828593e-05 | loss_stagnation_count = 1
Epoch: 100%|██████████| 100/100 [05:00<00:00,  3.01s/it]
2026-10-11 14:50:46.634 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:1.252795934677124
2026-10-11 14:50:46.635 | INFO     | __main__:<module>:287 - Output model artifact saved to = models/experimental/vt_dataset_skewed_gaussian_2d_data_params_mean_[0.0, 0.0]_cov_[[1.0, 0.0], [0.0, 1.0]]_shape_[5.0, 5.0]_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-11T14:45:44.172796.pkl
2026-10-11 14:50:46.636 | INFO     | __main__:<module>:288 - vt training process finished successfully.

Process finished with exit code 0

```
# Inference
```log
2026-10-11 15:40:07.508 | INFO     | __main__:<module>:163 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: -0.0002 | Std: 0.0004 |Inference time for total number of inference samples: 21.818478 seconds
2026-10-11 15:42:56.966 | INFO     | __main__:<module>:163 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: -0.0003 | Std: 0.0006 |Inference time for total number of inference samples: 21.881968 seconds
2026-10-11 15:43:50.948 | INFO     | __main__:<module>:163 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: -0.0002 | Std: 0.0003 |Inference time for total number of inference samples: 22.331231 seconds
```