

# Model Paths
## 1-RecFlow

```log
models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_8_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-15T17:35:20.804524.pkl
```
## 2-Recflow (dopri8)
```log
models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_8_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-15T18:19:55.778345.pkl
```

# Model Training log
## 1 - RecFlow
```log
/home/mbaddar/Documents/mbaddar/ttflow/.venv/bin/python /home/mbaddar/Documents/mbaddar/ttflow/ttflow/training/nn/vt_trainer.py --data-snapshot-filepath data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_8_data_dim_2_n_train_test_eval_500000_200000_200000.pkl --vt-model nn --vt-opt adam --trajectory-model global --reflow-iterations 0 --nn-epochs 300 --reflow-iterations 0 --ode-solver-method dopri8 
2026-10-15 17:35:20.804 | INFO     | __main__:<module>:154 - args = Namespace(data_snapshot_filepath='data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_8_data_dim_2_n_train_test_eval_500000_200000_200000.pkl', use_gpu=True, vt_model='nn', vt_opt='adam', trajectory_model='global', ode_solver_method='dopri8', rectification_linearity='linear', reflow_iterations=0, nn_epochs=300, nn_hidden_dim=128, batch_size=2048, nn_depth=2, lr=0.001, nn_reg_lambda=0.001, patience=10, min_rel_loss_delta=5e-05, dry_run=False, logs_dir='logs', output_model_dir='models/experimental')
2026-10-15 17:35:20.804 | INFO     | __main__:<module>:166 - is GPU/CUDA available = True
2026-10-15 17:35:20.804 | INFO     | __main__:<module>:169 - device = cuda
2026-10-15 17:35:20.881 | INFO     | __main__:<module>:174 - Using data snapshot: data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_8_data_dim_2_n_train_test_eval_500000_200000_200000.pkl
2026-10-15 17:35:20.881 | INFO     | ttflow.utils:check_system_memory_and_swap:56 - ram = 30.96, swap = 40.0 GB
2026-10-15 17:35:20.899 | INFO     | __main__:<module>:203 - data pkl size = 0.01 GB
2026-10-15 17:35:20.901 | INFO     | __main__:<module>:214 - Generating training data for the global trajectory training process..
2026-10-15 17:35:21.092 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-15 17:35:21.711 | INFO     | __main__:<module>:241 - ***********************************************************Training Model Specs:
  [Model Architecture]
    - model =                   VelocityFieldRegressionNet(
  (model): Sequential(
    (0): Linear(in_features=3, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=2, bias=True)
  )
)
    - Input Dim:                3
    - Hidden Dim:               128
    - Output Dim:               2
    - Depth:                    2
    - Device:                   cuda
    - Data Type:                torch.float32
  [Optimizer Specs]
    - Optimizer :               Adam (
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
):
    - Learning Rate:            0.001
    - Weight Decay (lambda):    0.001
  [Training Params]
    - Epochs:                   300
    - Batch Size:               2048
    - Patience:                 10
    - Min Rel Loss Delta:       5e-05
    - ZT Mode:                  linear
    - Reflow Iterations:  0**********************************************************
reflow iterations:   0%|          | 0/1 [00:00<?, ?it/s]2026-10-15 17:35:21.711 | INFO     | __main__:<module>:266 - Reflow iteration: 0
2026-10-15 17:35:22.615 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:57 - @epoch = 0 (before training), loss_fn = MSELoss(), loss_value = 50.405908
2026-10-15 17:35:22.620 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:64 - MSELoss(test) before training:50.40591049194336

Epoch:   0%|          | 0/300 [00:00<?, ?it/s]2026-10-15 17:35:25.789 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 1, smoothed-train-loss = 14.950357 | smoothed-test-loss = 11.031263 | rel_loss_delta = inf | loss_stagnation_count = 0

Epoch:   0%|          | 1/300 [00:03<15:47,  3.17s/it]2026-10-15 17:35:29.176 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 2, smoothed-train-loss = 11.036730 | smoothed-test-loss = 10.632206 | rel_loss_delta = inf | loss_stagnation_count = 0

Epoch:   1%|          | 2/300 [00:06<16:22,  3.30s/it]2026-10-15 17:35:33.011 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 3, smoothed-train-loss = 10.438446 | smoothed-test-loss = 10.437106 | rel_loss_delta = -0.054208447118662816 | loss_stagnation_count = 0

Epoch:   1%|          | 3/300 [00:10<17:32,  3.54s/it]2026-10-15 17:35:36.645 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 4, smoothed-train-loss = 10.175403 | smoothed-test-loss = 10.189650 | rel_loss_delta = -0.02519941901272887 | loss_stagnation_count = 0

Epoch:   1%|▏         | 4/300 [00:14<17:39,  3.58s/it]2026-10-15 17:35:39.688 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 5, smoothed-train-loss = 9.914149 | smoothed-test-loss = 9.905503 | rel_loss_delta = -0.02567509310404747 | loss_stagnation_count = 0

Epoch:   2%|▏         | 5/300 [00:17<16:38,  3.39s/it]2026-10-15 17:35:43.296 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 6, smoothed-train-loss = 9.672909 | smoothed-test-loss = 9.688892 | rel_loss_delta = -0.024332885105119367 | loss_stagnation_count = 0

Epoch:   2%|▏         | 6/300 [00:20<16:57,  3.46s/it]2026-10-15 17:35:46.530 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 7, smoothed-train-loss = 9.504320 | smoothed-test-loss = 9.578090 | rel_loss_delta = -0.01742893966259542 | loss_stagnation_count = 0

Epoch:   2%|▏         | 7/300 [00:23<16:32,  3.39s/it]2026-10-15 17:35:49.697 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 8, smoothed-train-loss = 9.402054 | smoothed-test-loss = 9.474763 | rel_loss_delta = -0.010760011432644007 | loss_stagnation_count = 0

Epoch:   3%|▎         | 8/300 [00:27<16:08,  3.32s/it]2026-10-15 17:35:52.912 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 9, smoothed-train-loss = 9.333700 | smoothed-test-loss = 9.462056 | rel_loss_delta = -0.00727004607546756 | loss_stagnation_count = 0

Epoch:   3%|▎         | 9/300 [00:30<15:55,  3.29s/it]2026-10-15 17:35:56.106 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 10, smoothed-train-loss = 9.284914 | smoothed-test-loss = 9.415534 | rel_loss_delta = -0.005226857360882137 | loss_stagnation_count = 0

Epoch:   3%|▎         | 10/300 [00:33<15:44,  3.26s/it]2026-10-15 17:35:59.481 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 11, smoothed-train-loss = 9.249583 | smoothed-test-loss = 9.363748 | rel_loss_delta = -0.003805225477385885 | loss_stagnation_count = 0

Epoch:   4%|▎         | 11/300 [00:36<15:51,  3.29s/it]2026-10-15 17:36:02.421 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 12, smoothed-train-loss = 9.222049 | smoothed-test-loss = 9.362665 | rel_loss_delta = -0.002976819095131079 | loss_stagnation_count = 0

Epoch:   4%|▍         | 12/300 [00:39<15:17,  3.19s/it]2026-10-15 17:36:05.543 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 13, smoothed-train-loss = 9.204624 | smoothed-test-loss = 9.332703 | rel_loss_delta = -0.0018894749651680782 | loss_stagnation_count = 0

Epoch:   4%|▍         | 13/300 [00:42<15:08,  3.17s/it]2026-10-15 17:36:08.910 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 14, smoothed-train-loss = 9.187802 | smoothed-test-loss = 9.343920 | rel_loss_delta = -0.0018275302523561709 | loss_stagnation_count = 0

Epoch:   5%|▍         | 14/300 [00:46<15:22,  3.23s/it]2026-10-15 17:36:12.293 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 15, smoothed-train-loss = 9.174213 | smoothed-test-loss = 9.286700 | rel_loss_delta = -0.0014790429558409108 | loss_stagnation_count = 0

Epoch:   5%|▌         | 15/300 [00:49<15:33,  3.27s/it]2026-10-15 17:36:15.374 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 16, smoothed-train-loss = 9.161891 | smoothed-test-loss = 9.292130 | rel_loss_delta = -0.0013431623247945124 | loss_stagnation_count = 0

Epoch:   5%|▌         | 16/300 [00:52<15:13,  3.22s/it]2026-10-15 17:36:18.241 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 17, smoothed-train-loss = 9.152912 | smoothed-test-loss = 9.282327 | rel_loss_delta = -0.0009799634127804634 | loss_stagnation_count = 0

Epoch:   6%|▌         | 17/300 [00:55<14:40,  3.11s/it]2026-10-15 17:36:21.409 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 18, smoothed-train-loss = 9.147800 | smoothed-test-loss = 9.318528 | rel_loss_delta = -0.0005585945857916046 | loss_stagnation_count = 0

Epoch:   6%|▌         | 18/300 [00:58<14:42,  3.13s/it]2026-10-15 17:36:24.878 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 19, smoothed-train-loss = 9.146034 | smoothed-test-loss = 9.261903 | rel_loss_delta = -0.00019305262335230102 | loss_stagnation_count = 0

Epoch:   6%|▋         | 19/300 [01:02<15:07,  3.23s/it]2026-10-15 17:36:28.341 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 20, smoothed-train-loss = 9.136477 | smoothed-test-loss = 9.283896 | rel_loss_delta = -0.0010449053745149246 | loss_stagnation_count = 0

Epoch:   7%|▋         | 20/300 [01:05<15:24,  3.30s/it]2026-10-15 17:36:31.500 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 21, smoothed-train-loss = 9.132492 | smoothed-test-loss = 9.311873 | rel_loss_delta = -0.0004361991011493065 | loss_stagnation_count = 0

Epoch:   7%|▋         | 21/300 [01:08<15:08,  3.26s/it]2026-10-15 17:36:35.129 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 22, smoothed-train-loss = 9.129160 | smoothed-test-loss = 9.258932 | rel_loss_delta = -0.0003647827773920043 | loss_stagnation_count = 0

Epoch:   7%|▋         | 22/300 [01:12<15:36,  3.37s/it]2026-10-15 17:36:38.428 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 23, smoothed-train-loss = 9.126131 | smoothed-test-loss = 9.242019 | rel_loss_delta = -0.00033184892580975626 | loss_stagnation_count = 0

Epoch:   8%|▊         | 23/300 [01:15<15:27,  3.35s/it]2026-10-15 17:36:41.479 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 24, smoothed-train-loss = 9.121062 | smoothed-test-loss = 9.223919 | rel_loss_delta = -0.0005554606717826467 | loss_stagnation_count = 0

Epoch:   8%|▊         | 24/300 [01:18<14:59,  3.26s/it]2026-10-15 17:36:45.169 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 25, smoothed-train-loss = 9.119349 | smoothed-test-loss = 9.240385 | rel_loss_delta = -0.0001877865254302289 | loss_stagnation_count = 0

Epoch:   8%|▊         | 25/300 [01:22<15:31,  3.39s/it]2026-10-15 17:36:49.054 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 26, smoothed-train-loss = 9.114753 | smoothed-test-loss = 9.256073 | rel_loss_delta = -0.0005039619455389712 | loss_stagnation_count = 0

Epoch:   9%|▊         | 26/300 [01:26<16:09,  3.54s/it]2026-10-15 17:36:52.274 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 27, smoothed-train-loss = 9.115289 | smoothed-test-loss = 9.270660 | rel_loss_delta = 5.882933073616992e-05 | loss_stagnation_count = 0

Epoch:   9%|▉         | 27/300 [01:29<15:39,  3.44s/it]2026-10-15 17:36:55.416 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 28, smoothed-train-loss = 9.113052 | smoothed-test-loss = 9.277673 | rel_loss_delta = -0.00024539135440930423 | loss_stagnation_count = 0

Epoch:   9%|▉         | 28/300 [01:32<15:11,  3.35s/it]2026-10-15 17:36:58.395 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 29, smoothed-train-loss = 9.107642 | smoothed-test-loss = 9.232705 | rel_loss_delta = -0.0005936809060373544 | loss_stagnation_count = 0

Epoch:  10%|▉         | 29/300 [01:35<14:38,  3.24s/it]2026-10-15 17:37:01.338 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 30, smoothed-train-loss = 9.107787 | smoothed-test-loss = 9.219965 | rel_loss_delta = 1.5868333259534807e-05 | loss_stagnation_count = 1

Epoch:  10%|█         | 30/300 [01:38<14:10,  3.15s/it]2026-10-15 17:37:04.411 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 31, smoothed-train-loss = 9.103340 | smoothed-test-loss = 9.226763 | rel_loss_delta = -0.0004882610060041191 | loss_stagnation_count = 0

Epoch:  10%|█         | 31/300 [01:41<14:01,  3.13s/it]2026-10-15 17:37:07.333 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 32, smoothed-train-loss = 9.107009 | smoothed-test-loss = 9.218762 | rel_loss_delta = 0.0004030740360284387 | loss_stagnation_count = 0

Epoch:  11%|█         | 32/300 [01:44<13:41,  3.07s/it]2026-10-15 17:37:10.235 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 33, smoothed-train-loss = 9.101234 | smoothed-test-loss = 9.226615 | rel_loss_delta = -0.0006340905288666987 | loss_stagnation_count = 0

Epoch:  11%|█         | 33/300 [01:47<13:25,  3.02s/it]2026-10-15 17:37:13.495 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 34, smoothed-train-loss = 9.099893 | smoothed-test-loss = 9.223133 | rel_loss_delta = -0.00014737768018971965 | loss_stagnation_count = 0

Epoch:  11%|█▏        | 34/300 [01:50<13:41,  3.09s/it]2026-10-15 17:37:16.356 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 35, smoothed-train-loss = 9.098750 | smoothed-test-loss = 9.260500 | rel_loss_delta = -0.00012556115051831773 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 35/300 [01:53<13:20,  3.02s/it]2026-10-15 17:37:19.216 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 36, smoothed-train-loss = 9.100957 | smoothed-test-loss = 9.234377 | rel_loss_delta = 0.00024253679809180497 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 36/300 [01:56<13:04,  2.97s/it]2026-10-15 17:37:22.005 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 37, smoothed-train-loss = 9.098020 | smoothed-test-loss = 9.224662 | rel_loss_delta = -0.00032278128152026864 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 37/300 [01:59<12:47,  2.92s/it]2026-10-15 17:37:25.024 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 38, smoothed-train-loss = 9.091573 | smoothed-test-loss = 9.258835 | rel_loss_delta = -0.0007085494117253571 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 38/300 [02:02<12:52,  2.95s/it]2026-10-15 17:37:28.038 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 39, smoothed-train-loss = 9.092378 | smoothed-test-loss = 9.266071 | rel_loss_delta = 8.848656324305162e-05 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 39/300 [02:05<12:54,  2.97s/it]2026-10-15 17:37:31.185 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 40, smoothed-train-loss = 9.094594 | smoothed-test-loss = 9.229611 | rel_loss_delta = 0.00024381678634263684 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 40/300 [02:08<13:05,  3.02s/it]2026-10-15 17:37:34.520 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 41, smoothed-train-loss = 9.094534 | smoothed-test-loss = 9.232859 | rel_loss_delta = -6.665561791671687e-06 | loss_stagnation_count = 1

Epoch:  14%|█▎        | 41/300 [02:11<13:26,  3.12s/it]2026-10-15 17:37:37.888 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 42, smoothed-train-loss = 9.088764 | smoothed-test-loss = 9.255238 | rel_loss_delta = -0.0006344220410534441 | loss_stagnation_count = 0

Epoch:  14%|█▍        | 42/300 [02:15<13:43,  3.19s/it]2026-10-15 17:37:41.674 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 43, smoothed-train-loss = 9.092008 | smoothed-test-loss = 9.226296 | rel_loss_delta = 0.00035691530082867184 | loss_stagnation_count = 0

Epoch:  14%|█▍        | 43/300 [02:19<14:25,  3.37s/it]2026-10-15 17:37:45.111 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 44, smoothed-train-loss = 9.089892 | smoothed-test-loss = 9.228591 | rel_loss_delta = -0.00023268474882697868 | loss_stagnation_count = 0

Epoch:  15%|█▍        | 44/300 [02:22<14:27,  3.39s/it]2026-10-15 17:37:47.998 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 45, smoothed-train-loss = 9.089283 | smoothed-test-loss = 9.256475 | rel_loss_delta = -6.70199420986236e-05 | loss_stagnation_count = 0

Epoch:  15%|█▌        | 45/300 [02:25<13:45,  3.24s/it]2026-10-15 17:37:51.015 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 46, smoothed-train-loss = 9.084314 | smoothed-test-loss = 9.233717 | rel_loss_delta = -0.0005466994485305072 | loss_stagnation_count = 0

Epoch:  15%|█▌        | 46/300 [02:28<13:25,  3.17s/it]2026-10-15 17:37:54.095 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 47, smoothed-train-loss = 9.079686 | smoothed-test-loss = 9.224901 | rel_loss_delta = -0.0005094169924068667 | loss_stagnation_count = 0

Epoch:  16%|█▌        | 47/300 [02:31<13:15,  3.14s/it]2026-10-15 17:37:57.094 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 48, smoothed-train-loss = 9.078687 | smoothed-test-loss = 9.245390 | rel_loss_delta = -0.0001101040519166282 | loss_stagnation_count = 0

Epoch:  16%|█▌        | 48/300 [02:34<13:01,  3.10s/it]2026-10-15 17:37:59.910 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 49, smoothed-train-loss = 9.080053 | smoothed-test-loss = 9.228193 | rel_loss_delta = 0.00015053993737700814 | loss_stagnation_count = 0

Epoch:  16%|█▋        | 49/300 [02:37<12:36,  3.02s/it]2026-10-15 17:38:02.776 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 50, smoothed-train-loss = 9.083104 | smoothed-test-loss = 9.211377 | rel_loss_delta = 0.000335945901111556 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 50/300 [02:40<12:22,  2.97s/it]2026-10-15 17:38:05.717 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 51, smoothed-train-loss = 9.080912 | smoothed-test-loss = 9.222574 | rel_loss_delta = -0.00024127642965345356 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 51/300 [02:43<12:17,  2.96s/it]2026-10-15 17:38:08.595 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 52, smoothed-train-loss = 9.083232 | smoothed-test-loss = 9.237557 | rel_loss_delta = 0.00025547155139677623 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 52/300 [02:45<12:08,  2.94s/it]2026-10-15 17:38:11.357 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 53, smoothed-train-loss = 9.079539 | smoothed-test-loss = 9.209967 | rel_loss_delta = -0.0004065684816666864 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 53/300 [02:48<11:52,  2.88s/it]2026-10-15 17:38:14.363 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 54, smoothed-train-loss = 9.077291 | smoothed-test-loss = 9.204683 | rel_loss_delta = -0.0002475828585291049 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 54/300 [02:51<11:58,  2.92s/it]2026-10-15 17:38:17.292 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 55, smoothed-train-loss = 9.076709 | smoothed-test-loss = 9.239194 | rel_loss_delta = -6.420024772017948e-05 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 55/300 [02:54<11:56,  2.92s/it]2026-10-15 17:38:20.298 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 56, smoothed-train-loss = 9.078443 | smoothed-test-loss = 9.212704 | rel_loss_delta = 0.00019106937616276336 | loss_stagnation_count = 0

Epoch:  19%|█▊        | 56/300 [02:57<11:59,  2.95s/it]2026-10-15 17:38:23.454 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 57, smoothed-train-loss = 9.074786 | smoothed-test-loss = 9.238124 | rel_loss_delta = -0.00040282317431080783 | loss_stagnation_count = 0

Epoch:  19%|█▉        | 57/300 [03:00<12:11,  3.01s/it]2026-10-15 17:38:26.486 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 58, smoothed-train-loss = 9.073969 | smoothed-test-loss = 9.206497 | rel_loss_delta = -8.996891834138634e-05 | loss_stagnation_count = 0

Epoch:  19%|█▉        | 58/300 [03:03<12:10,  3.02s/it]2026-10-15 17:38:29.542 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 59, smoothed-train-loss = 9.072001 | smoothed-test-loss = 9.215301 | rel_loss_delta = -0.00021688415776514785 | loss_stagnation_count = 0

Epoch:  20%|█▉        | 59/300 [03:06<12:09,  3.03s/it]2026-10-15 17:38:32.489 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 60, smoothed-train-loss = 9.074782 | smoothed-test-loss = 9.229551 | rel_loss_delta = 0.00030654671225300087 | loss_stagnation_count = 0

Epoch:  20%|██        | 60/300 [03:09<12:01,  3.00s/it]2026-10-15 17:38:35.398 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 61, smoothed-train-loss = 9.072499 | smoothed-test-loss = 9.234943 | rel_loss_delta = -0.0002516547944254385 | loss_stagnation_count = 0

Epoch:  20%|██        | 61/300 [03:12<11:51,  2.98s/it]2026-10-15 17:38:38.465 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 62, smoothed-train-loss = 9.073167 | smoothed-test-loss = 9.197575 | rel_loss_delta = 7.366039700225889e-05 | loss_stagnation_count = 0

Epoch:  21%|██        | 62/300 [03:15<11:54,  3.00s/it]2026-10-15 17:38:41.489 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 63, smoothed-train-loss = 9.070613 | smoothed-test-loss = 9.215819 | rel_loss_delta = -0.00028143045198764105 | loss_stagnation_count = 0

Epoch:  21%|██        | 63/300 [03:18<11:53,  3.01s/it]2026-10-15 17:38:44.473 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 64, smoothed-train-loss = 9.075634 | smoothed-test-loss = 9.201208 | rel_loss_delta = 0.0005534409735916401 | loss_stagnation_count = 0

Epoch:  21%|██▏       | 64/300 [03:21<11:48,  3.00s/it]2026-10-15 17:38:47.390 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 65, smoothed-train-loss = 9.071593 | smoothed-test-loss = 9.266308 | rel_loss_delta = -0.00044520504481288274 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 65/300 [03:24<11:39,  2.98s/it]2026-10-15 17:38:50.334 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 66, smoothed-train-loss = 9.075760 | smoothed-test-loss = 9.231695 | rel_loss_delta = 0.00045934841258859105 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 66/300 [03:27<11:34,  2.97s/it]2026-10-15 17:38:53.512 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 67, smoothed-train-loss = 9.071636 | smoothed-test-loss = 9.220151 | rel_loss_delta = -0.00045436387031782074 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 67/300 [03:30<11:46,  3.03s/it]2026-10-15 17:38:56.576 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 68, smoothed-train-loss = 9.068849 | smoothed-test-loss = 9.239733 | rel_loss_delta = -0.000307208978184415 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 68/300 [03:33<11:45,  3.04s/it]2026-10-15 17:38:59.393 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 69, smoothed-train-loss = 9.070903 | smoothed-test-loss = 9.204001 | rel_loss_delta = 0.00022644514376693301 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 69/300 [03:36<11:26,  2.97s/it]2026-10-15 17:39:02.271 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 70, smoothed-train-loss = 9.071535 | smoothed-test-loss = 9.221489 | rel_loss_delta = 6.962469474041091e-05 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 70/300 [03:39<11:17,  2.94s/it]2026-10-15 17:39:05.213 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 71, smoothed-train-loss = 9.067805 | smoothed-test-loss = 9.189517 | rel_loss_delta = -0.0004111600158496734 | loss_stagnation_count = 0

Epoch:  24%|██▎       | 71/300 [03:42<11:14,  2.94s/it]2026-10-15 17:39:08.429 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 72, smoothed-train-loss = 9.069795 | smoothed-test-loss = 9.243049 | rel_loss_delta = 0.0002194336464565561 | loss_stagnation_count = 0

Epoch:  24%|██▍       | 72/300 [03:45<11:29,  3.03s/it]2026-10-15 17:39:11.305 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 73, smoothed-train-loss = 9.067931 | smoothed-test-loss = 9.211330 | rel_loss_delta = -0.00020551297381286222 | loss_stagnation_count = 0

Epoch:  24%|██▍       | 73/300 [03:48<11:16,  2.98s/it]2026-10-15 17:39:14.218 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 74, smoothed-train-loss = 9.066698 | smoothed-test-loss = 9.201116 | rel_loss_delta = -0.00013590089225111798 | loss_stagnation_count = 0

Epoch:  25%|██▍       | 74/300 [03:51<11:09,  2.96s/it]2026-10-15 17:39:17.193 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 75, smoothed-train-loss = 9.064331 | smoothed-test-loss = 9.215167 | rel_loss_delta = -0.00026108070639217887 | loss_stagnation_count = 0

Epoch:  25%|██▌       | 75/300 [03:54<11:07,  2.96s/it]2026-10-15 17:39:20.190 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 76, smoothed-train-loss = 9.065701 | smoothed-test-loss = 9.231450 | rel_loss_delta = 0.00015113517449354794 | loss_stagnation_count = 0

Epoch:  25%|██▌       | 76/300 [03:57<11:06,  2.97s/it]2026-10-15 17:39:23.457 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 77, smoothed-train-loss = 9.068244 | smoothed-test-loss = 9.196173 | rel_loss_delta = 0.0002804782900303223 | loss_stagnation_count = 0

Epoch:  26%|██▌       | 77/300 [04:00<11:22,  3.06s/it]2026-10-15 17:39:26.520 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 78, smoothed-train-loss = 9.065731 | smoothed-test-loss = 9.195558 | rel_loss_delta = -0.000277135695554483 | loss_stagnation_count = 0

Epoch:  26%|██▌       | 78/300 [04:03<11:19,  3.06s/it]2026-10-15 17:39:29.469 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 79, smoothed-train-loss = 9.066958 | smoothed-test-loss = 9.223287 | rel_loss_delta = 0.00013540016881491014 | loss_stagnation_count = 0

Epoch:  26%|██▋       | 79/300 [04:06<11:09,  3.03s/it]2026-10-15 17:39:32.324 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 80, smoothed-train-loss = 9.066685 | smoothed-test-loss = 9.228604 | rel_loss_delta = -3.00899056027427e-05 | loss_stagnation_count = 1

Epoch:  27%|██▋       | 80/300 [04:09<10:54,  2.98s/it]2026-10-15 17:39:35.158 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 81, smoothed-train-loss = 9.064049 | smoothed-test-loss = 9.241592 | rel_loss_delta = -0.0002907306808536597 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 81/300 [04:12<10:42,  2.93s/it]2026-10-15 17:39:38.269 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 82, smoothed-train-loss = 9.064369 | smoothed-test-loss = 9.219623 | rel_loss_delta = 3.5275095035354955e-05 | loss_stagnation_count = 1

Epoch:  27%|██▋       | 82/300 [04:15<10:51,  2.99s/it]2026-10-15 17:39:41.150 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 83, smoothed-train-loss = 9.071292 | smoothed-test-loss = 9.196091 | rel_loss_delta = 0.0007637969807600372 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 83/300 [04:18<10:41,  2.96s/it]2026-10-15 17:39:43.998 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 84, smoothed-train-loss = 9.064533 | smoothed-test-loss = 9.179379 | rel_loss_delta = -0.0007451946091994247 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 84/300 [04:21<10:31,  2.92s/it]2026-10-15 17:39:46.763 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 85, smoothed-train-loss = 9.065676 | smoothed-test-loss = 9.221618 | rel_loss_delta = 0.0001261085606725458 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 85/300 [04:24<10:18,  2.88s/it]2026-10-15 17:39:49.721 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 86, smoothed-train-loss = 9.062107 | smoothed-test-loss = 9.183931 | rel_loss_delta = -0.0003935899150431028 | loss_stagnation_count = 0

Epoch:  29%|██▊       | 86/300 [04:27<10:20,  2.90s/it]2026-10-15 17:39:52.798 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 87, smoothed-train-loss = 9.062238 | smoothed-test-loss = 9.234866 | rel_loss_delta = 1.4442447284839034e-05 | loss_stagnation_count = 1

Epoch:  29%|██▉       | 87/300 [04:30<10:29,  2.95s/it]2026-10-15 17:39:55.792 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 88, smoothed-train-loss = 9.063524 | smoothed-test-loss = 9.194645 | rel_loss_delta = 0.00014183840891718747 | loss_stagnation_count = 0

Epoch:  29%|██▉       | 88/300 [04:33<10:28,  2.97s/it]2026-10-15 17:39:58.612 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 89, smoothed-train-loss = 9.059417 | smoothed-test-loss = 9.196086 | rel_loss_delta = -0.00045309713069385305 | loss_stagnation_count = 0

Epoch:  30%|██▉       | 89/300 [04:35<10:16,  2.92s/it]2026-10-15 17:40:01.551 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 90, smoothed-train-loss = 9.064571 | smoothed-test-loss = 9.210506 | rel_loss_delta = 0.0005688897781220518 | loss_stagnation_count = 0

Epoch:  30%|███       | 90/300 [04:38<10:14,  2.93s/it]2026-10-15 17:40:04.519 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 91, smoothed-train-loss = 9.068180 | smoothed-test-loss = 9.198634 | rel_loss_delta = 0.00039814196778366036 | loss_stagnation_count = 0

Epoch:  30%|███       | 91/300 [04:41<10:14,  2.94s/it]2026-10-15 17:40:07.566 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 92, smoothed-train-loss = 9.062606 | smoothed-test-loss = 9.225374 | rel_loss_delta = -0.0006146255416543681 | loss_stagnation_count = 0

Epoch:  31%|███       | 92/300 [04:44<10:18,  2.97s/it]2026-10-15 17:40:10.384 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 93, smoothed-train-loss = 9.059456 | smoothed-test-loss = 9.256070 | rel_loss_delta = -0.0003476042322948906 | loss_stagnation_count = 0

Epoch:  31%|███       | 93/300 [04:47<10:05,  2.93s/it]2026-10-15 17:40:13.269 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 94, smoothed-train-loss = 9.064724 | smoothed-test-loss = 9.238119 | rel_loss_delta = 0.0005814942419436266 | loss_stagnation_count = 0

Epoch:  31%|███▏      | 94/300 [04:50<10:00,  2.91s/it]2026-10-15 17:40:16.123 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 95, smoothed-train-loss = 9.062773 | smoothed-test-loss = 9.195540 | rel_loss_delta = -0.0002152564735999371 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 95/300 [04:53<09:53,  2.90s/it]2026-10-15 17:40:19.129 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 96, smoothed-train-loss = 9.061947 | smoothed-test-loss = 9.189169 | rel_loss_delta = -9.111555218604843e-05 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 96/300 [04:56<09:57,  2.93s/it]2026-10-15 17:40:22.254 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 97, smoothed-train-loss = 9.059806 | smoothed-test-loss = 9.203237 | rel_loss_delta = -0.00023624116797017942 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 97/300 [04:59<10:06,  2.99s/it]2026-10-15 17:40:25.228 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 98, smoothed-train-loss = 9.063815 | smoothed-test-loss = 9.194900 | rel_loss_delta = 0.00044250959961387277 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 98/300 [05:02<10:02,  2.98s/it]2026-10-15 17:40:28.206 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 99, smoothed-train-loss = 9.065449 | smoothed-test-loss = 9.187128 | rel_loss_delta = 0.00018028578355029705 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 99/300 [05:05<09:59,  2.98s/it]2026-10-15 17:40:31.207 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 100, smoothed-train-loss = 9.063118 | smoothed-test-loss = 9.197355 | rel_loss_delta = -0.0002572015146011321 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 100/300 [05:08<09:57,  2.99s/it]2026-10-15 17:40:34.011 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 101, smoothed-train-loss = 9.058199 | smoothed-test-loss = 9.187875 | rel_loss_delta = -0.0005427179852734437 | loss_stagnation_count = 0

Epoch:  34%|███▎      | 101/300 [05:11<09:43,  2.93s/it]2026-10-15 17:40:36.980 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 102, smoothed-train-loss = 9.059664 | smoothed-test-loss = 9.242982 | rel_loss_delta = 0.0001616817796557308 | loss_stagnation_count = 0

Epoch:  34%|███▍      | 102/300 [05:14<09:42,  2.94s/it]2026-10-15 17:40:39.861 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 103, smoothed-train-loss = 9.065057 | smoothed-test-loss = 9.206076 | rel_loss_delta = 0.0005953198791667927 | loss_stagnation_count = 0

Epoch:  34%|███▍      | 103/300 [05:17<09:36,  2.92s/it]2026-10-15 17:40:42.727 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 104, smoothed-train-loss = 9.058973 | smoothed-test-loss = 9.250673 | rel_loss_delta = -0.000671206828552258 | loss_stagnation_count = 0

Epoch:  35%|███▍      | 104/300 [05:20<09:29,  2.91s/it]2026-10-15 17:40:45.477 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 105, smoothed-train-loss = 9.060792 | smoothed-test-loss = 9.206544 | rel_loss_delta = 0.00020079243893453998 | loss_stagnation_count = 0

Epoch:  35%|███▌      | 105/300 [05:22<09:17,  2.86s/it]2026-10-15 17:40:48.335 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 106, smoothed-train-loss = 9.060705 | smoothed-test-loss = 9.217818 | rel_loss_delta = -9.540224739619862e-06 | loss_stagnation_count = 1

Epoch:  35%|███▌      | 106/300 [05:25<09:14,  2.86s/it]2026-10-15 17:40:51.553 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 107, smoothed-train-loss = 9.059176 | smoothed-test-loss = 9.251142 | rel_loss_delta = -0.00016880798191548088 | loss_stagnation_count = 0

Epoch:  36%|███▌      | 107/300 [05:28<09:32,  2.97s/it]2026-10-15 17:40:54.484 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 108, smoothed-train-loss = 9.060212 | smoothed-test-loss = 9.209473 | rel_loss_delta = 0.00011442418594418373 | loss_stagnation_count = 0

Epoch:  36%|███▌      | 108/300 [05:31<09:27,  2.96s/it]2026-10-15 17:40:57.267 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 109, smoothed-train-loss = 9.061916 | smoothed-test-loss = 9.190639 | rel_loss_delta = 0.00018800686389226022 | loss_stagnation_count = 0

Epoch:  36%|███▋      | 109/300 [05:34<09:14,  2.90s/it]2026-10-15 17:41:00.154 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 110, smoothed-train-loss = 9.058234 | smoothed-test-loss = 9.185813 | rel_loss_delta = -0.000406296835533367 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 110/300 [05:37<09:10,  2.90s/it]2026-10-15 17:41:03.011 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 111, smoothed-train-loss = 9.057387 | smoothed-test-loss = 9.189785 | rel_loss_delta = -9.345273269037659e-05 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 111/300 [05:40<09:05,  2.89s/it]2026-10-15 17:41:05.959 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 112, smoothed-train-loss = 9.059287 | smoothed-test-loss = 9.212287 | rel_loss_delta = 0.00020971594091238752 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 112/300 [05:43<09:06,  2.90s/it]2026-10-15 17:41:08.782 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 113, smoothed-train-loss = 9.058239 | smoothed-test-loss = 9.217052 | rel_loss_delta = -0.00011560205616459417 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 113/300 [05:46<08:58,  2.88s/it]2026-10-15 17:41:11.780 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 114, smoothed-train-loss = 9.062099 | smoothed-test-loss = 9.187222 | rel_loss_delta = 0.00042603907986901105 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 114/300 [05:49<09:02,  2.92s/it]2026-10-15 17:41:14.802 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 115, smoothed-train-loss = 9.059927 | smoothed-test-loss = 9.199358 | rel_loss_delta = -0.00023959236126352014 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 115/300 [05:52<09:05,  2.95s/it]2026-10-15 17:41:17.800 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 116, smoothed-train-loss = 9.055428 | smoothed-test-loss = 9.208905 | rel_loss_delta = -0.0004965811655636323 | loss_stagnation_count = 0

Epoch:  39%|███▊      | 116/300 [05:55<09:05,  2.96s/it]2026-10-15 17:41:20.855 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 117, smoothed-train-loss = 9.056304 | smoothed-test-loss = 9.204083 | rel_loss_delta = 9.667342144419809e-05 | loss_stagnation_count = 0

Epoch:  39%|███▉      | 117/300 [05:58<09:07,  2.99s/it]2026-10-15 17:41:23.908 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 118, smoothed-train-loss = 9.058164 | smoothed-test-loss = 9.208759 | rel_loss_delta = 0.00020542205691835295 | loss_stagnation_count = 0

Epoch:  39%|███▉      | 118/300 [06:01<09:07,  3.01s/it]2026-10-15 17:41:26.902 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 119, smoothed-train-loss = 9.061454 | smoothed-test-loss = 9.195637 | rel_loss_delta = 0.0003631610214692306 | loss_stagnation_count = 0

Epoch:  40%|███▉      | 119/300 [06:04<09:03,  3.00s/it]2026-10-15 17:41:29.885 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 120, smoothed-train-loss = 9.056155 | smoothed-test-loss = 9.195732 | rel_loss_delta = -0.0005847946225748696 | loss_stagnation_count = 0

Epoch:  40%|████      | 120/300 [06:07<08:59,  3.00s/it]2026-10-15 17:41:32.796 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 121, smoothed-train-loss = 9.058004 | smoothed-test-loss = 9.188182 | rel_loss_delta = 0.00020425183040616215 | loss_stagnation_count = 0

Epoch:  40%|████      | 121/300 [06:10<08:51,  2.97s/it]2026-10-15 17:41:36.018 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 122, smoothed-train-loss = 9.055901 | smoothed-test-loss = 9.200066 | rel_loss_delta = -0.0002321767700712941 | loss_stagnation_count = 0

Epoch:  41%|████      | 122/300 [06:13<09:02,  3.05s/it]2026-10-15 17:41:39.027 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 123, smoothed-train-loss = 9.056199 | smoothed-test-loss = 9.179719 | rel_loss_delta = 3.292659649423921e-05 | loss_stagnation_count = 1

Epoch:  41%|████      | 123/300 [06:16<08:57,  3.04s/it]2026-10-15 17:41:41.930 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 124, smoothed-train-loss = 9.057528 | smoothed-test-loss = 9.202386 | rel_loss_delta = 0.00014667019497422798 | loss_stagnation_count = 0

Epoch:  41%|████▏     | 124/300 [06:19<08:47,  3.00s/it]2026-10-15 17:41:44.744 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 125, smoothed-train-loss = 9.057422 | smoothed-test-loss = 9.199717 | rel_loss_delta = -1.162123123516799e-05 | loss_stagnation_count = 1

Epoch:  42%|████▏     | 125/300 [06:22<08:34,  2.94s/it]2026-10-15 17:41:47.647 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 126, smoothed-train-loss = 9.058956 | smoothed-test-loss = 9.207932 | rel_loss_delta = 0.0001692758368506398 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 126/300 [06:25<08:29,  2.93s/it]2026-10-15 17:41:50.689 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 127, smoothed-train-loss = 9.058234 | smoothed-test-loss = 9.179391 | rel_loss_delta = -7.968011709599033e-05 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 127/300 [06:28<08:32,  2.96s/it]2026-10-15 17:41:53.638 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 128, smoothed-train-loss = 9.057699 | smoothed-test-loss = 9.181792 | rel_loss_delta = -5.905543926462633e-05 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 128/300 [06:31<08:28,  2.96s/it]2026-10-15 17:41:56.437 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 129, smoothed-train-loss = 9.056204 | smoothed-test-loss = 9.190714 | rel_loss_delta = -0.00016499979159624573 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 129/300 [06:33<08:17,  2.91s/it]2026-10-15 17:41:59.411 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 130, smoothed-train-loss = 9.056856 | smoothed-test-loss = 9.191901 | rel_loss_delta = 7.197438450831448e-05 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 130/300 [06:36<08:18,  2.93s/it]2026-10-15 17:42:02.350 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 131, smoothed-train-loss = 9.056596 | smoothed-test-loss = 9.208042 | rel_loss_delta = -2.8725879832320972e-05 | loss_stagnation_count = 1

Epoch:  44%|████▎     | 131/300 [06:39<08:15,  2.93s/it]2026-10-15 17:42:05.311 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 132, smoothed-train-loss = 9.053929 | smoothed-test-loss = 9.196380 | rel_loss_delta = -0.00029453598389922735 | loss_stagnation_count = 0

Epoch:  44%|████▍     | 132/300 [06:42<08:14,  2.94s/it]2026-10-15 17:42:08.252 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 133, smoothed-train-loss = 9.055327 | smoothed-test-loss = 9.196161 | rel_loss_delta = 0.00015445889856735947 | loss_stagnation_count = 0

Epoch:  44%|████▍     | 133/300 [06:45<08:11,  2.94s/it]2026-10-15 17:42:11.166 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 134, smoothed-train-loss = 9.053841 | smoothed-test-loss = 9.205189 | rel_loss_delta = -0.00016413809203424072 | loss_stagnation_count = 0

Epoch:  45%|████▍     | 134/300 [06:48<08:06,  2.93s/it]2026-10-15 17:42:14.031 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 135, smoothed-train-loss = 9.054533 | smoothed-test-loss = 9.222707 | rel_loss_delta = 7.647608793252455e-05 | loss_stagnation_count = 0

Epoch:  45%|████▌     | 135/300 [06:51<08:00,  2.91s/it]2026-10-15 17:42:16.884 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 136, smoothed-train-loss = 9.056431 | smoothed-test-loss = 9.200747 | rel_loss_delta = 0.00020958308939104532 | loss_stagnation_count = 0

Epoch:  45%|████▌     | 136/300 [06:54<07:54,  2.89s/it]2026-10-15 17:42:19.766 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 137, smoothed-train-loss = 9.054475 | smoothed-test-loss = 9.200507 | rel_loss_delta = -0.00021594407920309266 | loss_stagnation_count = 0

Epoch:  46%|████▌     | 137/300 [06:57<07:51,  2.89s/it]2026-10-15 17:42:22.788 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 138, smoothed-train-loss = 9.054918 | smoothed-test-loss = 9.193882 | rel_loss_delta = 4.889241565625097e-05 | loss_stagnation_count = 1

Epoch:  46%|████▌     | 138/300 [07:00<07:54,  2.93s/it]2026-10-15 17:42:25.653 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 139, smoothed-train-loss = 9.054037 | smoothed-test-loss = 9.204175 | rel_loss_delta = -9.731206357374239e-05 | loss_stagnation_count = 0

Epoch:  46%|████▋     | 139/300 [07:03<07:48,  2.91s/it]2026-10-15 17:42:28.505 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 140, smoothed-train-loss = 9.053473 | smoothed-test-loss = 9.208327 | rel_loss_delta = -6.227048750886647e-05 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 140/300 [07:05<07:42,  2.89s/it]2026-10-15 17:42:31.414 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 141, smoothed-train-loss = 9.055190 | smoothed-test-loss = 9.201833 | rel_loss_delta = 0.00018964597517890215 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 141/300 [07:08<07:40,  2.90s/it]2026-10-15 17:42:34.475 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 142, smoothed-train-loss = 9.053777 | smoothed-test-loss = 9.207763 | rel_loss_delta = -0.00015605658781830882 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 142/300 [07:11<07:45,  2.95s/it]2026-10-15 17:42:37.565 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 143, smoothed-train-loss = 9.054263 | smoothed-test-loss = 9.191512 | rel_loss_delta = 5.3668980630126406e-05 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 143/300 [07:14<07:49,  2.99s/it]2026-10-15 17:42:40.419 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 144, smoothed-train-loss = 9.056494 | smoothed-test-loss = 9.177100 | rel_loss_delta = 0.00024642304034502193 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 144/300 [07:17<07:40,  2.95s/it]2026-10-15 17:42:43.192 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 145, smoothed-train-loss = 9.055932 | smoothed-test-loss = 9.186868 | rel_loss_delta = -6.206393943364966e-05 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 145/300 [07:20<07:28,  2.90s/it]2026-10-15 17:42:46.317 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 146, smoothed-train-loss = 9.055052 | smoothed-test-loss = 9.179589 | rel_loss_delta = -9.713363939641787e-05 | loss_stagnation_count = 0

Epoch:  49%|████▊     | 146/300 [07:23<07:36,  2.96s/it]2026-10-15 17:42:49.337 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 147, smoothed-train-loss = 9.054408 | smoothed-test-loss = 9.201387 | rel_loss_delta = -7.109077838269823e-05 | loss_stagnation_count = 0

Epoch:  49%|████▉     | 147/300 [07:26<07:36,  2.98s/it]2026-10-15 17:42:52.447 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 148, smoothed-train-loss = 9.053842 | smoothed-test-loss = 9.180159 | rel_loss_delta = -6.256255754837658e-05 | loss_stagnation_count = 0

Epoch:  49%|████▉     | 148/300 [07:29<07:39,  3.02s/it]2026-10-15 17:42:55.275 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 149, smoothed-train-loss = 9.054713 | smoothed-test-loss = 9.197089 | rel_loss_delta = 9.6267563951678e-05 | loss_stagnation_count = 0

Epoch:  50%|████▉     | 149/300 [07:32<07:27,  2.96s/it]2026-10-15 17:42:58.252 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 150, smoothed-train-loss = 9.053552 | smoothed-test-loss = 9.236162 | rel_loss_delta = -0.000128266670340287 | loss_stagnation_count = 0

Epoch:  50%|█████     | 150/300 [07:35<07:25,  2.97s/it]2026-10-15 17:43:01.173 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 151, smoothed-train-loss = 9.056059 | smoothed-test-loss = 9.204606 | rel_loss_delta = 0.0002769391621016156 | loss_stagnation_count = 0

Epoch:  50%|█████     | 151/300 [07:38<07:19,  2.95s/it]2026-10-15 17:43:04.172 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 152, smoothed-train-loss = 9.054088 | smoothed-test-loss = 9.193777 | rel_loss_delta = -0.0002176581360510504 | loss_stagnation_count = 0

Epoch:  51%|█████     | 152/300 [07:41<07:19,  2.97s/it]2026-10-15 17:43:07.214 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 153, smoothed-train-loss = 9.053044 | smoothed-test-loss = 9.201406 | rel_loss_delta = -0.00011530886952755211 | loss_stagnation_count = 0

Epoch:  51%|█████     | 153/300 [07:44<07:19,  2.99s/it]2026-10-15 17:43:10.251 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 154, smoothed-train-loss = 9.052375 | smoothed-test-loss = 9.177826 | rel_loss_delta = -7.38708054812147e-05 | loss_stagnation_count = 0

Epoch:  51%|█████▏    | 154/300 [07:47<07:18,  3.00s/it]2026-10-15 17:43:13.243 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 155, smoothed-train-loss = 9.050850 | smoothed-test-loss = 9.178356 | rel_loss_delta = -0.00016850403909646024 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 155/300 [07:50<07:15,  3.00s/it]2026-10-15 17:43:16.207 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 156, smoothed-train-loss = 9.054846 | smoothed-test-loss = 9.203295 | rel_loss_delta = 0.000441506963272626 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 156/300 [07:53<07:10,  2.99s/it]2026-10-15 17:43:19.025 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 157, smoothed-train-loss = 9.054539 | smoothed-test-loss = 9.181852 | rel_loss_delta = -3.391059554862817e-05 | loss_stagnation_count = 1

Epoch:  52%|█████▏    | 157/300 [07:56<07:00,  2.94s/it]2026-10-15 17:43:22.134 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 158, smoothed-train-loss = 9.050975 | smoothed-test-loss = 9.183561 | rel_loss_delta = -0.0003935832026558575 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 158/300 [07:59<07:04,  2.99s/it]2026-10-15 17:43:25.157 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 159, smoothed-train-loss = 9.052203 | smoothed-test-loss = 9.192134 | rel_loss_delta = 0.00013560989948900568 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 159/300 [08:02<07:02,  3.00s/it]2026-10-15 17:43:28.135 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 160, smoothed-train-loss = 9.051532 | smoothed-test-loss = 9.187652 | rel_loss_delta = -7.405281470277778e-05 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 160/300 [08:05<06:59,  2.99s/it]2026-10-15 17:43:31.039 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 161, smoothed-train-loss = 9.051684 | smoothed-test-loss = 9.176493 | rel_loss_delta = 1.675088489593329e-05 | loss_stagnation_count = 1

Epoch:  54%|█████▎    | 161/300 [08:08<06:52,  2.97s/it]2026-10-15 17:43:33.987 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 162, smoothed-train-loss = 9.049979 | smoothed-test-loss = 9.195976 | rel_loss_delta = -0.00018836299401967354 | loss_stagnation_count = 0

Epoch:  54%|█████▍    | 162/300 [08:11<06:48,  2.96s/it]2026-10-15 17:43:37.097 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 163, smoothed-train-loss = 9.051995 | smoothed-test-loss = 9.218508 | rel_loss_delta = 0.00022277552828443546 | loss_stagnation_count = 0

Epoch:  54%|█████▍    | 163/300 [08:14<06:51,  3.01s/it]2026-10-15 17:43:40.560 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 164, smoothed-train-loss = 9.053932 | smoothed-test-loss = 9.209524 | rel_loss_delta = 0.00021399846049684963 | loss_stagnation_count = 0

Epoch:  55%|█████▍    | 164/300 [08:17<07:07,  3.14s/it]2026-10-15 17:43:43.597 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 165, smoothed-train-loss = 9.049680 | smoothed-test-loss = 9.180207 | rel_loss_delta = -0.0004696254780793192 | loss_stagnation_count = 0

Epoch:  55%|█████▌    | 165/300 [08:20<06:59,  3.11s/it]2026-10-15 17:43:46.529 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 166, smoothed-train-loss = 9.049890 | smoothed-test-loss = 9.183524 | rel_loss_delta = 2.3180020270719085e-05 | loss_stagnation_count = 1

Epoch:  55%|█████▌    | 166/300 [08:23<06:49,  3.06s/it]2026-10-15 17:43:49.555 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 167, smoothed-train-loss = 9.052092 | smoothed-test-loss = 9.196533 | rel_loss_delta = 0.00024331066112527621 | loss_stagnation_count = 0

Epoch:  56%|█████▌    | 167/300 [08:26<06:45,  3.05s/it]2026-10-15 17:43:53.265 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 168, smoothed-train-loss = 9.052245 | smoothed-test-loss = 9.193843 | rel_loss_delta = 1.6912406700491024e-05 | loss_stagnation_count = 1

Epoch:  56%|█████▌    | 168/300 [08:30<07:08,  3.25s/it]2026-10-15 17:43:56.132 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 169, smoothed-train-loss = 9.051715 | smoothed-test-loss = 9.183770 | rel_loss_delta = -5.8570795267979694e-05 | loss_stagnation_count = 0

Epoch:  56%|█████▋    | 169/300 [08:33<06:50,  3.13s/it]2026-10-15 17:43:59.245 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 170, smoothed-train-loss = 9.049889 | smoothed-test-loss = 9.185406 | rel_loss_delta = -0.00020175604427111822 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 170/300 [08:36<06:46,  3.13s/it]2026-10-15 17:44:02.252 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 171, smoothed-train-loss = 9.050762 | smoothed-test-loss = 9.189798 | rel_loss_delta = 9.656251406164162e-05 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 171/300 [08:39<06:38,  3.09s/it]2026-10-15 17:44:05.300 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 172, smoothed-train-loss = 9.047892 | smoothed-test-loss = 9.199577 | rel_loss_delta = -0.00031712555749783606 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 172/300 [08:42<06:33,  3.08s/it]2026-10-15 17:44:08.305 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 173, smoothed-train-loss = 9.050580 | smoothed-test-loss = 9.184842 | rel_loss_delta = 0.0002970138552936242 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 173/300 [08:45<06:28,  3.06s/it]2026-10-15 17:44:11.191 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 174, smoothed-train-loss = 9.051989 | smoothed-test-loss = 9.202252 | rel_loss_delta = 0.00015577187233894866 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 174/300 [08:48<06:18,  3.01s/it]2026-10-15 17:44:14.188 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 175, smoothed-train-loss = 9.050220 | smoothed-test-loss = 9.179935 | rel_loss_delta = -0.00019552711860240275 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 175/300 [08:51<06:15,  3.00s/it]2026-10-15 17:44:17.149 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 176, smoothed-train-loss = 9.049730 | smoothed-test-loss = 9.177887 | rel_loss_delta = -5.412894935542203e-05 | loss_stagnation_count = 0

Epoch:  59%|█████▊    | 176/300 [08:54<06:10,  2.99s/it]2026-10-15 17:44:19.995 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 177, smoothed-train-loss = 9.047889 | smoothed-test-loss = 9.175357 | rel_loss_delta = -0.00020342377073260214 | loss_stagnation_count = 0

Epoch:  59%|█████▉    | 177/300 [08:57<06:02,  2.95s/it]2026-10-15 17:44:23.429 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 178, smoothed-train-loss = 9.047180 | smoothed-test-loss = 9.197731 | rel_loss_delta = -7.833223849713671e-05 | loss_stagnation_count = 0

Epoch:  59%|█████▉    | 178/300 [09:00<06:17,  3.09s/it]2026-10-15 17:44:26.887 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 179, smoothed-train-loss = 9.050003 | smoothed-test-loss = 9.180827 | rel_loss_delta = 0.0003120270917061488 | loss_stagnation_count = 0

Epoch:  60%|█████▉    | 179/300 [09:04<06:27,  3.20s/it]2026-10-15 17:44:30.461 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 180, smoothed-train-loss = 9.052636 | smoothed-test-loss = 9.189103 | rel_loss_delta = 0.0002909322347030073 | loss_stagnation_count = 0

Epoch:  60%|██████    | 180/300 [09:07<06:37,  3.31s/it]2026-10-15 17:44:34.065 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 181, smoothed-train-loss = 9.049213 | smoothed-test-loss = 9.194274 | rel_loss_delta = -0.00037810480185395865 | loss_stagnation_count = 0

Epoch:  60%|██████    | 181/300 [09:11<06:44,  3.40s/it]2026-10-15 17:44:37.423 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 182, smoothed-train-loss = 9.047970 | smoothed-test-loss = 9.205347 | rel_loss_delta = -0.00013741897417624596 | loss_stagnation_count = 0

Epoch:  61%|██████    | 182/300 [09:14<06:39,  3.39s/it]2026-10-15 17:44:40.510 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 183, smoothed-train-loss = 9.050218 | smoothed-test-loss = 9.186627 | rel_loss_delta = 0.00024854240738225643 | loss_stagnation_count = 0

Epoch:  61%|██████    | 183/300 [09:17<06:25,  3.30s/it]2026-10-15 17:44:44.119 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 184, smoothed-train-loss = 9.049781 | smoothed-test-loss = 9.179539 | rel_loss_delta = -4.832672066572394e-05 | loss_stagnation_count = 1

Epoch:  61%|██████▏   | 184/300 [09:21<06:33,  3.39s/it]2026-10-15 17:44:47.722 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 185, smoothed-train-loss = 9.048609 | smoothed-test-loss = 9.196257 | rel_loss_delta = -0.0001294614507160805 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 185/300 [09:25<06:37,  3.45s/it]2026-10-15 17:44:51.733 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 186, smoothed-train-loss = 9.046012 | smoothed-test-loss = 9.180421 | rel_loss_delta = -0.00028701745395711504 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 186/300 [09:29<06:52,  3.62s/it]2026-10-15 17:44:55.164 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 187, smoothed-train-loss = 9.047833 | smoothed-test-loss = 9.187957 | rel_loss_delta = 0.0002013117158332566 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 187/300 [09:32<06:42,  3.56s/it]2026-10-15 17:44:58.326 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 188, smoothed-train-loss = 9.049670 | smoothed-test-loss = 9.177417 | rel_loss_delta = 0.0002030060620875266 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 188/300 [09:35<06:25,  3.44s/it]2026-10-15 17:45:01.613 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 189, smoothed-train-loss = 9.045224 | smoothed-test-loss = 9.202024 | rel_loss_delta = -0.0004913213241234715 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 189/300 [09:38<06:17,  3.40s/it]2026-10-15 17:45:05.133 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 190, smoothed-train-loss = 9.048486 | smoothed-test-loss = 9.198001 | rel_loss_delta = 0.00036064246781133177 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 190/300 [09:42<06:17,  3.43s/it]2026-10-15 17:45:08.908 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 191, smoothed-train-loss = 9.050465 | smoothed-test-loss = 9.189813 | rel_loss_delta = 0.00021874338556052746 | loss_stagnation_count = 0

Epoch:  64%|██████▎   | 191/300 [09:46<06:25,  3.54s/it]2026-10-15 17:45:12.527 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 192, smoothed-train-loss = 9.046362 | smoothed-test-loss = 9.208600 | rel_loss_delta = -0.00045331966230293053 | loss_stagnation_count = 0

Epoch:  64%|██████▍   | 192/300 [09:49<06:24,  3.56s/it]2026-10-15 17:45:15.505 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 193, smoothed-train-loss = 9.045553 | smoothed-test-loss = 9.187995 | rel_loss_delta = -8.949985253419595e-05 | loss_stagnation_count = 0

Epoch:  64%|██████▍   | 193/300 [09:52<06:02,  3.39s/it]2026-10-15 17:45:18.476 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 194, smoothed-train-loss = 9.050445 | smoothed-test-loss = 9.177328 | rel_loss_delta = 0.0005408801052078028 | loss_stagnation_count = 0

Epoch:  65%|██████▍   | 194/300 [09:55<05:45,  3.26s/it]2026-10-15 17:45:21.712 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 195, smoothed-train-loss = 9.046995 | smoothed-test-loss = 9.179614 | rel_loss_delta = -0.0003812817025680593 | loss_stagnation_count = 0

Epoch:  65%|██████▌   | 195/300 [09:59<05:41,  3.25s/it]2026-10-15 17:45:25.195 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 196, smoothed-train-loss = 9.043905 | smoothed-test-loss = 9.182855 | rel_loss_delta = -0.0003414606611786894 | loss_stagnation_count = 0

Epoch:  65%|██████▌   | 196/300 [10:02<05:45,  3.32s/it]2026-10-15 17:45:28.044 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 197, smoothed-train-loss = 9.049357 | smoothed-test-loss = 9.180440 | rel_loss_delta = 0.0006027899884534964 | loss_stagnation_count = 0

Epoch:  66%|██████▌   | 197/300 [10:05<05:27,  3.18s/it]2026-10-15 17:45:31.248 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 198, smoothed-train-loss = 9.047688 | smoothed-test-loss = 9.195693 | rel_loss_delta = -0.0001844258543935563 | loss_stagnation_count = 0

Epoch:  66%|██████▌   | 198/300 [10:08<05:25,  3.19s/it]2026-10-15 17:45:34.544 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 199, smoothed-train-loss = 9.046187 | smoothed-test-loss = 9.183260 | rel_loss_delta = -0.00016591941497089312 | loss_stagnation_count = 0

Epoch:  66%|██████▋   | 199/300 [10:11<05:25,  3.22s/it]2026-10-15 17:45:38.099 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 200, smoothed-train-loss = 9.046889 | smoothed-test-loss = 9.193629 | rel_loss_delta = 7.761300577589402e-05 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 200/300 [10:15<05:32,  3.32s/it]2026-10-15 17:45:41.258 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 201, smoothed-train-loss = 9.048331 | smoothed-test-loss = 9.198592 | rel_loss_delta = 0.00015938935312994625 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 201/300 [10:18<05:23,  3.27s/it]2026-10-15 17:45:44.299 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 202, smoothed-train-loss = 9.047534 | smoothed-test-loss = 9.177928 | rel_loss_delta = -8.812090179028378e-05 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 202/300 [10:21<05:13,  3.20s/it]2026-10-15 17:45:47.564 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 203, smoothed-train-loss = 9.047605 | smoothed-test-loss = 9.217993 | rel_loss_delta = 7.913031763152066e-06 | loss_stagnation_count = 1

Epoch:  68%|██████▊   | 203/300 [10:24<05:12,  3.22s/it]2026-10-15 17:45:50.474 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 204, smoothed-train-loss = 9.047067 | smoothed-test-loss = 9.179316 | rel_loss_delta = -5.9440515241365476e-05 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 204/300 [10:27<05:00,  3.13s/it]2026-10-15 17:45:53.498 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 205, smoothed-train-loss = 9.047527 | smoothed-test-loss = 9.177900 | rel_loss_delta = 5.081888395240554e-05 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 205/300 [10:30<04:54,  3.10s/it]2026-10-15 17:45:56.443 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 206, smoothed-train-loss = 9.044765 | smoothed-test-loss = 9.178438 | rel_loss_delta = -0.0003053003225969266 | loss_stagnation_count = 0

Epoch:  69%|██████▊   | 206/300 [10:33<04:46,  3.05s/it]2026-10-15 17:45:59.321 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 207, smoothed-train-loss = 9.046500 | smoothed-test-loss = 9.173200 | rel_loss_delta = 0.00019183863928113907 | loss_stagnation_count = 0

Epoch:  69%|██████▉   | 207/300 [10:36<04:38,  3.00s/it]2026-10-15 17:46:02.519 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 208, smoothed-train-loss = 9.048009 | smoothed-test-loss = 9.211917 | rel_loss_delta = 0.00016676205634189934 | loss_stagnation_count = 0

Epoch:  69%|██████▉   | 208/300 [10:39<04:41,  3.06s/it]2026-10-15 17:46:05.521 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 209, smoothed-train-loss = 9.047698 | smoothed-test-loss = 9.169963 | rel_loss_delta = -3.4313798168608784e-05 | loss_stagnation_count = 1

Epoch:  70%|██████▉   | 209/300 [10:42<04:36,  3.04s/it]2026-10-15 17:46:08.668 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 210, smoothed-train-loss = 9.046785 | smoothed-test-loss = 9.195210 | rel_loss_delta = -0.00010089308380656825 | loss_stagnation_count = 0

Epoch:  70%|███████   | 210/300 [10:46<04:36,  3.07s/it]2026-10-15 17:46:11.938 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 211, smoothed-train-loss = 9.048740 | smoothed-test-loss = 9.179457 | rel_loss_delta = 0.00021606071727667888 | loss_stagnation_count = 0

Epoch:  70%|███████   | 211/300 [10:49<04:38,  3.13s/it]2026-10-15 17:46:15.186 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 212, smoothed-train-loss = 9.043046 | smoothed-test-loss = 9.169912 | rel_loss_delta = -0.000629211322642465 | loss_stagnation_count = 0

Epoch:  71%|███████   | 212/300 [10:52<04:38,  3.17s/it]2026-10-15 17:46:18.212 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 213, smoothed-train-loss = 9.045480 | smoothed-test-loss = 9.189287 | rel_loss_delta = 0.00026916603164436957 | loss_stagnation_count = 0

Epoch:  71%|███████   | 213/300 [10:55<04:31,  3.12s/it]2026-10-15 17:46:21.462 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 214, smoothed-train-loss = 9.049094 | smoothed-test-loss = 9.181055 | rel_loss_delta = 0.00039945965082523177 | loss_stagnation_count = 0

Epoch:  71%|███████▏  | 214/300 [10:58<04:31,  3.16s/it]2026-10-15 17:46:24.717 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 215, smoothed-train-loss = 9.049663 | smoothed-test-loss = 9.192400 | rel_loss_delta = 6.288844952905242e-05 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 215/300 [11:02<04:31,  3.19s/it]2026-10-15 17:46:27.823 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 216, smoothed-train-loss = 9.047684 | smoothed-test-loss = 9.188640 | rel_loss_delta = -0.00021871966542499067 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 216/300 [11:05<04:25,  3.16s/it]2026-10-15 17:46:30.760 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 217, smoothed-train-loss = 9.042891 | smoothed-test-loss = 9.179621 | rel_loss_delta = -0.0005296766973598874 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 217/300 [11:08<04:17,  3.10s/it]2026-10-15 17:46:33.903 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 218, smoothed-train-loss = 9.045929 | smoothed-test-loss = 9.168476 | rel_loss_delta = 0.0003358923105124062 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 218/300 [11:11<04:15,  3.11s/it]2026-10-15 17:46:37.047 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 219, smoothed-train-loss = 9.043888 | smoothed-test-loss = 9.179394 | rel_loss_delta = -0.0002255413106957888 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 219/300 [11:14<04:12,  3.12s/it]2026-10-15 17:46:40.950 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 220, smoothed-train-loss = 9.042246 | smoothed-test-loss = 9.163198 | rel_loss_delta = -0.0001816563726543334 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 220/300 [11:18<04:28,  3.36s/it]2026-10-15 17:46:44.157 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 221, smoothed-train-loss = 9.044857 | smoothed-test-loss = 9.191997 | rel_loss_delta = 0.0002888018614675793 | loss_stagnation_count = 0

Epoch:  74%|███████▎  | 221/300 [11:21<04:21,  3.31s/it]2026-10-15 17:46:47.467 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 222, smoothed-train-loss = 9.044635 | smoothed-test-loss = 9.176042 | rel_loss_delta = -2.4532174422057775e-05 | loss_stagnation_count = 1

Epoch:  74%|███████▍  | 222/300 [11:24<04:18,  3.31s/it]2026-10-15 17:46:50.858 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 223, smoothed-train-loss = 9.045307 | smoothed-test-loss = 9.224783 | rel_loss_delta = 7.428263101083437e-05 | loss_stagnation_count = 0

Epoch:  74%|███████▍  | 223/300 [11:28<04:16,  3.33s/it]2026-10-15 17:46:53.978 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 224, smoothed-train-loss = 9.040800 | smoothed-test-loss = 9.184309 | rel_loss_delta = -0.0004983017926119606 | loss_stagnation_count = 0

Epoch:  75%|███████▍  | 224/300 [11:31<04:08,  3.27s/it]2026-10-15 17:46:56.846 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 225, smoothed-train-loss = 9.043622 | smoothed-test-loss = 9.173460 | rel_loss_delta = 0.0003122100287018723 | loss_stagnation_count = 0

Epoch:  75%|███████▌  | 225/300 [11:34<03:56,  3.15s/it]2026-10-15 17:46:59.810 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 226, smoothed-train-loss = 9.044137 | smoothed-test-loss = 9.219728 | rel_loss_delta = 5.695449460403779e-05 | loss_stagnation_count = 0

Epoch:  75%|███████▌  | 226/300 [11:37<03:48,  3.09s/it]2026-10-15 17:47:02.762 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 227, smoothed-train-loss = 9.044368 | smoothed-test-loss = 9.204143 | rel_loss_delta = 2.548201674772203e-05 | loss_stagnation_count = 1

Epoch:  76%|███████▌  | 227/300 [11:40<03:42,  3.05s/it]2026-10-15 17:47:06.827 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 228, smoothed-train-loss = 9.039299 | smoothed-test-loss = 9.200480 | rel_loss_delta = -0.0005603848667405731 | loss_stagnation_count = 0

Epoch:  76%|███████▌  | 228/300 [11:44<04:01,  3.36s/it]2026-10-15 17:47:10.179 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 229, smoothed-train-loss = 9.041002 | smoothed-test-loss = 9.218466 | rel_loss_delta = 0.00018830387188789873 | loss_stagnation_count = 0

Epoch:  76%|███████▋  | 229/300 [11:47<03:58,  3.35s/it]2026-10-15 17:47:13.521 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 230, smoothed-train-loss = 9.041630 | smoothed-test-loss = 9.189943 | rel_loss_delta = 6.949372241972984e-05 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 230/300 [11:50<03:54,  3.35s/it]2026-10-15 17:47:16.898 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 231, smoothed-train-loss = 9.043921 | smoothed-test-loss = 9.187434 | rel_loss_delta = 0.00025338554583889385 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 231/300 [11:54<03:51,  3.36s/it]2026-10-15 17:47:20.132 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 232, smoothed-train-loss = 9.042926 | smoothed-test-loss = 9.181553 | rel_loss_delta = -0.00010997632301118973 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 232/300 [11:57<03:45,  3.32s/it]2026-10-15 17:47:23.216 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 233, smoothed-train-loss = 9.045496 | smoothed-test-loss = 9.206581 | rel_loss_delta = 0.00028414557157108427 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 233/300 [12:00<03:37,  3.25s/it]2026-10-15 17:47:26.166 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 234, smoothed-train-loss = 9.044143 | smoothed-test-loss = 9.194003 | rel_loss_delta = -0.00014949896597995526 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 234/300 [12:03<03:28,  3.16s/it]2026-10-15 17:47:30.182 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 235, smoothed-train-loss = 9.042662 | smoothed-test-loss = 9.172842 | rel_loss_delta = -0.00016383200855526236 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 235/300 [12:07<03:42,  3.42s/it]2026-10-15 17:47:33.260 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 236, smoothed-train-loss = 9.041674 | smoothed-test-loss = 9.171335 | rel_loss_delta = -0.00010920722821327697 | loss_stagnation_count = 0

Epoch:  79%|███████▊  | 236/300 [12:10<03:32,  3.32s/it]2026-10-15 17:47:36.394 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 237, smoothed-train-loss = 9.043083 | smoothed-test-loss = 9.202838 | rel_loss_delta = 0.00015579888634042173 | loss_stagnation_count = 0

Epoch:  79%|███████▉  | 237/300 [12:13<03:25,  3.26s/it]2026-10-15 17:47:39.400 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 238, smoothed-train-loss = 9.043262 | smoothed-test-loss = 9.180437 | rel_loss_delta = 1.9790894917202218e-05 | loss_stagnation_count = 1

Epoch:  79%|███████▉  | 238/300 [12:16<03:17,  3.18s/it]2026-10-15 17:47:42.489 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 239, smoothed-train-loss = 9.044467 | smoothed-test-loss = 9.177331 | rel_loss_delta = 0.00013324682016478418 | loss_stagnation_count = 0

Epoch:  80%|███████▉  | 239/300 [12:19<03:12,  3.16s/it]2026-10-15 17:47:45.703 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 240, smoothed-train-loss = 9.043830 | smoothed-test-loss = 9.186317 | rel_loss_delta = -7.039350247401007e-05 | loss_stagnation_count = 0

Epoch:  80%|████████  | 240/300 [12:23<03:10,  3.17s/it]2026-10-15 17:47:48.959 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 241, smoothed-train-loss = 9.045116 | smoothed-test-loss = 9.181379 | rel_loss_delta = 0.00014215775574651508 | loss_stagnation_count = 0

Epoch:  80%|████████  | 241/300 [12:26<03:08,  3.20s/it]2026-10-15 17:47:52.255 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 242, smoothed-train-loss = 9.044063 | smoothed-test-loss = 9.173343 | rel_loss_delta = -0.00011637408075801644 | loss_stagnation_count = 0

Epoch:  81%|████████  | 242/300 [12:29<03:07,  3.23s/it]2026-10-15 17:47:55.465 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 243, smoothed-train-loss = 9.042940 | smoothed-test-loss = 9.189374 | rel_loss_delta = -0.00012421969242935074 | loss_stagnation_count = 0

Epoch:  81%|████████  | 243/300 [12:32<03:03,  3.22s/it]2026-10-15 17:47:59.174 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 244, smoothed-train-loss = 9.042042 | smoothed-test-loss = 9.210468 | rel_loss_delta = -9.931464181618534e-05 | loss_stagnation_count = 0

Epoch:  81%|████████▏ | 244/300 [12:36<03:08,  3.37s/it]2026-10-15 17:48:02.493 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 245, smoothed-train-loss = 9.039597 | smoothed-test-loss = 9.181586 | rel_loss_delta = -0.0002704197501097376 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 245/300 [12:39<03:04,  3.35s/it]2026-10-15 17:48:05.646 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 246, smoothed-train-loss = 9.042735 | smoothed-test-loss = 9.177902 | rel_loss_delta = 0.0003471819219208164 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 246/300 [12:43<02:57,  3.29s/it]2026-10-15 17:48:09.337 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 247, smoothed-train-loss = 9.043445 | smoothed-test-loss = 9.174982 | rel_loss_delta = 7.851948950087797e-05 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 247/300 [12:46<03:00,  3.41s/it]2026-10-15 17:48:12.509 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 248, smoothed-train-loss = 9.045392 | smoothed-test-loss = 9.194821 | rel_loss_delta = 0.00021531732922044626 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 248/300 [12:49<02:53,  3.34s/it]2026-10-15 17:48:15.510 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 249, smoothed-train-loss = 9.041905 | smoothed-test-loss = 9.172767 | rel_loss_delta = -0.0003855183266580416 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 249/300 [12:52<02:45,  3.24s/it]2026-10-15 17:48:18.724 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 250, smoothed-train-loss = 9.039466 | smoothed-test-loss = 9.213024 | rel_loss_delta = -0.00026978198215793763 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 250/300 [12:56<02:41,  3.23s/it]2026-10-15 17:48:21.771 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 251, smoothed-train-loss = 9.041651 | smoothed-test-loss = 9.185935 | rel_loss_delta = 0.00024171958558602996 | loss_stagnation_count = 0

Epoch:  84%|████████▎ | 251/300 [12:59<02:35,  3.18s/it]2026-10-15 17:48:24.930 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 252, smoothed-train-loss = 9.043366 | smoothed-test-loss = 9.177051 | rel_loss_delta = 0.0001897616062069147 | loss_stagnation_count = 0

Epoch:  84%|████████▍ | 252/300 [13:02<02:32,  3.17s/it]2026-10-15 17:48:28.286 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 253, smoothed-train-loss = 9.043082 | smoothed-test-loss = 9.166282 | rel_loss_delta = -3.147475264408239e-05 | loss_stagnation_count = 1

Epoch:  84%|████████▍ | 253/300 [13:05<02:31,  3.23s/it]2026-10-15 17:48:32.304 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 254, smoothed-train-loss = 9.042785 | smoothed-test-loss = 9.179591 | rel_loss_delta = -3.285004586150205e-05 | loss_stagnation_count = 2

Epoch:  85%|████████▍ | 254/300 [13:09<02:39,  3.46s/it]2026-10-15 17:48:35.668 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 255, smoothed-train-loss = 9.041508 | smoothed-test-loss = 9.182066 | rel_loss_delta = -0.00014116463180504092 | loss_stagnation_count = 0

Epoch:  85%|████████▌ | 255/300 [13:13<02:34,  3.43s/it]2026-10-15 17:48:39.252 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 256, smoothed-train-loss = 9.042043 | smoothed-test-loss = 9.178440 | rel_loss_delta = 5.918740678083594e-05 | loss_stagnation_count = 0

Epoch:  85%|████████▌ | 256/300 [13:16<02:33,  3.48s/it]2026-10-15 17:48:42.604 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 257, smoothed-train-loss = 9.043539 | smoothed-test-loss = 9.171414 | rel_loss_delta = 0.00016539136525008772 | loss_stagnation_count = 0

Epoch:  86%|████████▌ | 257/300 [13:19<02:27,  3.44s/it]2026-10-15 17:48:45.863 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 258, smoothed-train-loss = 9.043344 | smoothed-test-loss = 9.168296 | rel_loss_delta = -2.1543951225194106e-05 | loss_stagnation_count = 1

Epoch:  86%|████████▌ | 258/300 [13:23<02:22,  3.39s/it]2026-10-15 17:48:49.489 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 259, smoothed-train-loss = 9.038483 | smoothed-test-loss = 9.164981 | rel_loss_delta = -0.0005375121053711147 | loss_stagnation_count = 0

Epoch:  86%|████████▋ | 259/300 [13:26<02:21,  3.46s/it]2026-10-15 17:48:53.088 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 260, smoothed-train-loss = 9.044505 | smoothed-test-loss = 9.184715 | rel_loss_delta = 0.0006662130771893203 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 260/300 [13:30<02:20,  3.50s/it]2026-10-15 17:48:56.291 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 261, smoothed-train-loss = 9.041136 | smoothed-test-loss = 9.168038 | rel_loss_delta = -0.0003724831370357436 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 261/300 [13:33<02:13,  3.41s/it]2026-10-15 17:48:59.472 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 262, smoothed-train-loss = 9.039496 | smoothed-test-loss = 9.173760 | rel_loss_delta = -0.00018135481893026943 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 262/300 [13:36<02:07,  3.34s/it]2026-10-15 17:49:02.577 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 263, smoothed-train-loss = 9.041039 | smoothed-test-loss = 9.184001 | rel_loss_delta = 0.000170691682829227 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 263/300 [13:39<02:01,  3.27s/it]2026-10-15 17:49:05.630 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 264, smoothed-train-loss = 9.039470 | smoothed-test-loss = 9.183954 | rel_loss_delta = -0.00017353083497238897 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 264/300 [13:43<01:55,  3.21s/it]2026-10-15 17:49:08.968 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 265, smoothed-train-loss = 9.042699 | smoothed-test-loss = 9.188756 | rel_loss_delta = 0.0003572102534662052 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 265/300 [13:46<01:53,  3.25s/it]2026-10-15 17:49:12.374 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 266, smoothed-train-loss = 9.041878 | smoothed-test-loss = 9.178438 | rel_loss_delta = -9.078689927571204e-05 | loss_stagnation_count = 0

Epoch:  89%|████████▊ | 266/300 [13:49<01:51,  3.29s/it]2026-10-15 17:49:15.789 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 267, smoothed-train-loss = 9.041527 | smoothed-test-loss = 9.198851 | rel_loss_delta = -3.8878465678584855e-05 | loss_stagnation_count = 1

Epoch:  89%|████████▉ | 267/300 [13:53<01:49,  3.33s/it]2026-10-15 17:49:18.826 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 268, smoothed-train-loss = 9.041869 | smoothed-test-loss = 9.188325 | rel_loss_delta = 3.782386206895148e-05 | loss_stagnation_count = 2

Epoch:  89%|████████▉ | 268/300 [13:56<01:43,  3.24s/it]2026-10-15 17:49:22.003 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 269, smoothed-train-loss = 9.040554 | smoothed-test-loss = 9.199906 | rel_loss_delta = -0.00014541088873504831 | loss_stagnation_count = 0

Epoch:  90%|████████▉ | 269/300 [13:59<01:39,  3.22s/it]2026-10-15 17:49:25.253 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 270, smoothed-train-loss = 9.041086 | smoothed-test-loss = 9.218432 | rel_loss_delta = 5.886587234780818e-05 | loss_stagnation_count = 0

Epoch:  90%|█████████ | 270/300 [14:02<01:36,  3.23s/it]2026-10-15 17:49:28.482 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 271, smoothed-train-loss = 9.041899 | smoothed-test-loss = 9.174172 | rel_loss_delta = 8.990987158512404e-05 | loss_stagnation_count = 0

Epoch:  90%|█████████ | 271/300 [14:05<01:33,  3.23s/it]2026-10-15 17:49:32.065 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 272, smoothed-train-loss = 9.037200 | smoothed-test-loss = 9.178774 | rel_loss_delta = -0.0005196291180701938 | loss_stagnation_count = 0

Epoch:  91%|█████████ | 272/300 [14:09<01:33,  3.34s/it]2026-10-15 17:49:35.493 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 273, smoothed-train-loss = 9.039264 | smoothed-test-loss = 9.183394 | rel_loss_delta = 0.00022832616190559017 | loss_stagnation_count = 0

Epoch:  91%|█████████ | 273/300 [14:12<01:30,  3.36s/it]2026-10-15 17:49:39.576 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 274, smoothed-train-loss = 9.038660 | smoothed-test-loss = 9.171960 | rel_loss_delta = -6.676744831093116e-05 | loss_stagnation_count = 0

Epoch:  91%|█████████▏| 274/300 [14:16<01:33,  3.58s/it]2026-10-15 17:49:42.760 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 275, smoothed-train-loss = 9.043966 | smoothed-test-loss = 9.173272 | rel_loss_delta = 0.0005869735734783442 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 275/300 [14:20<01:26,  3.46s/it]2026-10-15 17:49:45.974 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 276, smoothed-train-loss = 9.044428 | smoothed-test-loss = 9.178446 | rel_loss_delta = 5.115386644888991e-05 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 276/300 [14:23<01:21,  3.39s/it]2026-10-15 17:49:49.533 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 277, smoothed-train-loss = 9.040295 | smoothed-test-loss = 9.201729 | rel_loss_delta = -0.00045702641580091944 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 277/300 [14:26<01:19,  3.44s/it]2026-10-15 17:49:52.776 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 278, smoothed-train-loss = 9.041435 | smoothed-test-loss = 9.188587 | rel_loss_delta = 0.00012609242220420783 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 278/300 [14:30<01:14,  3.38s/it]2026-10-15 17:49:55.733 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 279, smoothed-train-loss = 9.039952 | smoothed-test-loss = 9.191704 | rel_loss_delta = -0.00016398092224253588 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 279/300 [14:33<01:08,  3.25s/it]2026-10-15 17:49:58.730 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 280, smoothed-train-loss = 9.042829 | smoothed-test-loss = 9.188130 | rel_loss_delta = 0.00031819883759698115 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 280/300 [14:36<01:03,  3.18s/it]2026-10-15 17:50:01.539 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 281, smoothed-train-loss = 9.043994 | smoothed-test-loss = 9.189657 | rel_loss_delta = 0.00012886494813214904 | loss_stagnation_count = 0

Epoch:  94%|█████████▎| 281/300 [14:38<00:58,  3.07s/it]2026-10-15 17:50:05.204 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 282, smoothed-train-loss = 9.039071 | smoothed-test-loss = 9.185583 | rel_loss_delta = -0.0005443686032068822 | loss_stagnation_count = 0

Epoch:  94%|█████████▍| 282/300 [14:42<00:58,  3.25s/it]2026-10-15 17:50:08.879 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 283, smoothed-train-loss = 9.040210 | smoothed-test-loss = 9.172070 | rel_loss_delta = 0.00012599618442756517 | loss_stagnation_count = 0

Epoch:  94%|█████████▍| 283/300 [14:46<00:57,  3.37s/it]2026-10-15 17:50:11.818 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 284, smoothed-train-loss = 9.041377 | smoothed-test-loss = 9.168611 | rel_loss_delta = 0.0001290776855959363 | loss_stagnation_count = 0

Epoch:  95%|█████████▍| 284/300 [14:49<00:51,  3.24s/it]2026-10-15 17:50:14.746 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 285, smoothed-train-loss = 9.043791 | smoothed-test-loss = 9.168728 | rel_loss_delta = 0.00026701811759195283 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 285/300 [14:52<00:47,  3.15s/it]2026-10-15 17:50:18.587 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 286, smoothed-train-loss = 9.040159 | smoothed-test-loss = 9.172311 | rel_loss_delta = -0.0004015688981744625 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 286/300 [14:55<00:46,  3.36s/it]2026-10-15 17:50:22.171 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 287, smoothed-train-loss = 9.038785 | smoothed-test-loss = 9.165774 | rel_loss_delta = -0.0001519577584867683 | loss_stagnation_count = 0

Epoch:  96%|█████████▌| 287/300 [14:59<00:44,  3.43s/it]2026-10-15 17:50:25.684 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 288, smoothed-train-loss = 9.043015 | smoothed-test-loss = 9.162093 | rel_loss_delta = 0.0004679807072850139 | loss_stagnation_count = 0

Epoch:  96%|█████████▌| 288/300 [15:03<00:41,  3.45s/it]2026-10-15 17:50:28.627 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 289, smoothed-train-loss = 9.041268 | smoothed-test-loss = 9.184859 | rel_loss_delta = -0.0001932570188249127 | loss_stagnation_count = 0

Epoch:  96%|█████████▋| 289/300 [15:06<00:36,  3.30s/it]2026-10-15 17:50:31.654 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 290, smoothed-train-loss = 9.039262 | smoothed-test-loss = 9.176253 | rel_loss_delta = -0.00022181268626701346 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 290/300 [15:09<00:32,  3.22s/it]2026-10-15 17:50:35.025 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 291, smoothed-train-loss = 9.040163 | smoothed-test-loss = 9.189828 | rel_loss_delta = 9.96520399823271e-05 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 291/300 [15:12<00:29,  3.26s/it]2026-10-15 17:50:38.207 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 292, smoothed-train-loss = 9.042226 | smoothed-test-loss = 9.174525 | rel_loss_delta = 0.0002281732846446613 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 292/300 [15:15<00:25,  3.24s/it]2026-10-15 17:50:41.411 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 293, smoothed-train-loss = 9.040779 | smoothed-test-loss = 9.174559 | rel_loss_delta = -0.0001600253351591874 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 293/300 [15:18<00:22,  3.23s/it]2026-10-15 17:50:45.149 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 294, smoothed-train-loss = 9.040339 | smoothed-test-loss = 9.196442 | rel_loss_delta = -4.864882826795469e-05 | loss_stagnation_count = 1

Epoch:  98%|█████████▊| 294/300 [15:22<00:20,  3.38s/it]2026-10-15 17:50:48.221 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 295, smoothed-train-loss = 9.038971 | smoothed-test-loss = 9.189962 | rel_loss_delta = -0.0001512611127887146 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 295/300 [15:25<00:16,  3.29s/it]2026-10-15 17:50:51.455 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 296, smoothed-train-loss = 9.038140 | smoothed-test-loss = 9.177079 | rel_loss_delta = -9.194029260820804e-05 | loss_stagnation_count = 0

Epoch:  99%|█████████▊| 296/300 [15:28<00:13,  3.27s/it]2026-10-15 17:50:54.373 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 297, smoothed-train-loss = 9.043369 | smoothed-test-loss = 9.183131 | rel_loss_delta = 0.0005784878124746783 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 297/300 [15:31<00:09,  3.17s/it]2026-10-15 17:50:57.305 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 298, smoothed-train-loss = 9.039567 | smoothed-test-loss = 9.185811 | rel_loss_delta = -0.000420352843535056 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 298/300 [15:34<00:06,  3.10s/it]2026-10-15 17:51:00.276 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 299, smoothed-train-loss = 9.037280 | smoothed-test-loss = 9.169020 | rel_loss_delta = -0.000253094396226302 | loss_stagnation_count = 0

Epoch: 100%|█████████▉| 299/300 [15:37<00:03,  3.06s/it]2026-10-15 17:51:03.292 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 300, smoothed-train-loss = 9.041064 | smoothed-test-loss = 9.169133 | rel_loss_delta = 0.00041875303376445874 | loss_stagnation_count = 0

Epoch: 100%|██████████| 300/300 [15:40<00:00,  3.14s/it]
2026-10-15 17:51:03.297 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:9.169145584106445
2026-10-15 17:51:03.298 | INFO     | __main__:<module>:282 - Given num_reflow_iterations = 0 and current reflow_iteration = 0, no further reflows are needed, exiting the reflow loop
reflow iterations:   0%|          | 0/1 [15:41<?, ?it/s]
2026-10-15 17:51:03.303 | INFO     | __main__:<module>:340 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_8_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-15T17:35:20.804524.pkl
2026-10-15 17:51:03.303 | INFO     | __main__:<module>:341 - vt training process finished successfully.

Process finished with exit code 0


```

# Inference 
## 1-Recflow
### `euler`
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : euler
  VT Trajectory Model  : global
  Num Reflows          : 0
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 5.48s
  Evaluations (t)      : 1000
  Eval Results         : {0.0038669902831315996: {'real_generated': {'mean': 0.2618657797574997, 'stddev': 0.043810424743808876}, 'real_real': {'mean': 0.15618237629532813, 'stddev': 0.08614998499087174}, 'generated_generated': {'mean': 0.13527657389640807, 'stddev': 0.062430381233327725}}, 0.009667475707828999: {'real_generated': {'mean': 0.3084324061870575, 'stddev': 0.09220000377831838}, 'real_real': {'mean': 0.1368156537413597, 'stddev': 0.06481567604759404}, 'generated_generated': {'mean': 0.13219963163137435, 'stddev': 0.06478301693870117}}, 0.01: {'real_generated': {'mean': 0.3537142276763916, 'stddev': 0.09869618550791875}, 'real_real': {'mean': 0.19167810976505278, 'stddev': 0.07029280837077755}, 'generated_generated': {'mean': 0.19222444146871567, 'stddev': 0.11406259216993415}}, 0.019334951415657997: {'real_generated': {'mean': 0.25075501799583433, 'stddev': 0.07140507687326421}, 'real_real': {'mean': 0.13589497059583663, 'stddev': 0.049904396807242034}, 'generated_generated': {'mean': 0.1116779588162899, 'stddev': 0.05431265735059546}}, 0.02: {'real_generated': {'mean': 0.34468490481376646, 'stddev': 0.06985969563819645}, 'real_real': {'mean': 0.16973766684532166, 'stddev': 0.04749504252484057}, 'generated_generated': {'mean': 0.13898431658744811, 'stddev': 0.0329426187094977}}, 0.038669902831315994: {'real_generated': {'mean': 0.3407731711864471, 'stddev': 0.15255545590149933}, 'real_real': {'mean': 0.16571780815720558, 'stddev': 0.09110262064831774}, 'generated_generated': {'mean': 0.12350649833679199, 'stddev': 0.04584463806932678}}, 0.05: {'real_generated': {'mean': 0.2925927549600601, 'stddev': 0.10685967509392783}, 'real_real': {'mean': 0.17740901112556456, 'stddev': 0.10186174730580837}, 'generated_generated': {'mean': 0.10099643021821976, 'stddev': 0.01911661766070288}}, 0.07733980566263199: {'real_generated': {'mean': 0.3307484805583954, 'stddev': 0.08976205121348364}, 'real_real': {'mean': 0.10576886534690857, 'stddev': 0.07197884778020706}, 'generated_generated': {'mean': 0.15017496272921563, 'stddev': 0.06399216092508551}}, 0.1: {'real_generated': {'mean': 0.3031585991382599, 'stddev': 0.09231019464520186}, 'real_real': {'mean': 0.18782680630683898, 'stddev': 0.09988626866307053}, 'generated_generated': {'mean': 0.17044447213411332, 'stddev': 0.1362275752152481}}, 0.2: {'real_generated': {'mean': 0.25847202241420747, 'stddev': 0.07463744019593453}, 'real_real': {'mean': 0.18486612141132355, 'stddev': 0.11978507377993684}, 'generated_generated': {'mean': 0.11268474757671357, 'stddev': 0.05598635754113879}}, 0.5: {'real_generated': {'mean': 0.28165406733751297, 'stddev': 0.10221615787287906}, 'real_real': {'mean': 0.16748639941215515, 'stddev': 0.07622498905718274}, 'generated_generated': {'mean': 0.13522952497005464, 'stddev': 0.08352852124648494}}, 1.0: {'real_generated': {'mean': 0.2901422679424286, 'stddev': 0.047458392189556414}, 'real_real': {'mean': 0.1503288932144642, 'stddev': 0.08616592047492139}, 'generated_generated': {'mean': 0.15648228526115418, 'stddev': 0.17327168083436525}}, 2.0: {'real_generated': {'mean': 0.13652418404817582, 'stddev': 0.01911507363288171}, 'real_real': {'mean': 0.13146761879324914, 'stddev': 0.08104647987122819}, 'generated_generated': {'mean': 0.10409569218754769, 'stddev': 0.06623556095566603}}, 5.0: {'real_generated': {'mean': 0.039438056945800784, 'stddev': 0.012741365380959062}, 'real_real': {'mean': 0.019261469319462775, 'stddev': 0.016052266206296892}, 'generated_generated': {'mean': 0.012946054339408875, 'stddev': 0.009077907756418932}}, 10.0: {'real_generated': {'mean': 0.019373439997434617, 'stddev': 0.004439537239122367}, 'real_real': {'mean': 0.013621512241661549, 'stddev': 0.009722301136876403}, 'generated_generated': {'mean': 0.010225255601108074, 'stddev': 0.0036909904621543026}}}
==================================================
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_8_timestamp_2026-10-16T12:42:46.082243_trajectory_model_global_ode_solver_method_euler_num_reflows_0.png
```
### `rk4`
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : rk4
  VT Trajectory Model  : global
  Num Reflows          : 0
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 21.23s
  Evaluations (t)      : 1000
  Eval Results         : {0.0038669902831315996: {'real_generated': {'mean': 0.3020250678062439, 'stddev': 0.10929056321616754}, 'real_real': {'mean': 0.15618237629532813, 'stddev': 0.08614998499087174}, 'generated_generated': {'mean': 0.22609237134456633, 'stddev': 0.04996071991071575}}, 0.009667475707828999: {'real_generated': {'mean': 0.28578963279724123, 'stddev': 0.117570862023416}, 'real_real': {'mean': 0.1368156537413597, 'stddev': 0.06481567604759404}, 'generated_generated': {'mean': 0.15432636588811874, 'stddev': 0.059265000153866806}}, 0.01: {'real_generated': {'mean': 0.26460211873054507, 'stddev': 0.05646615653593855}, 'real_real': {'mean': 0.19167810976505278, 'stddev': 0.07029280837077755}, 'generated_generated': {'mean': 0.13555929362773894, 'stddev': 0.06954829421593652}}, 0.019334951415657997: {'real_generated': {'mean': 0.3276574343442917, 'stddev': 0.07040552716088316}, 'real_real': {'mean': 0.13589497059583663, 'stddev': 0.049904396807242034}, 'generated_generated': {'mean': 0.15300975143909454, 'stddev': 0.05630968325680092}}, 0.02: {'real_generated': {'mean': 0.31749928593635557, 'stddev': 0.08468218809228069}, 'real_real': {'mean': 0.16973766684532166, 'stddev': 0.04749504252484057}, 'generated_generated': {'mean': 0.12678828835487366, 'stddev': 0.05753900853104522}}, 0.038669902831315994: {'real_generated': {'mean': 0.2990540027618408, 'stddev': 0.11942303330078054}, 'real_real': {'mean': 0.16571780815720558, 'stddev': 0.09110262064831774}, 'generated_generated': {'mean': 0.14938368797302246, 'stddev': 0.05844849486069889}}, 0.05: {'real_generated': {'mean': 0.22181516885757446, 'stddev': 0.045986434811456575}, 'real_real': {'mean': 0.17740901112556456, 'stddev': 0.10186174730580837}, 'generated_generated': {'mean': 0.11092725098133087, 'stddev': 0.031782355336492187}}, 0.07733980566263199: {'real_generated': {'mean': 0.2718777596950531, 'stddev': 0.10229051625793065}, 'real_real': {'mean': 0.10576886534690857, 'stddev': 0.07197884778020706}, 'generated_generated': {'mean': 0.15043575167655945, 'stddev': 0.12119721426849263}}, 0.1: {'real_generated': {'mean': 0.28725423812866213, 'stddev': 0.09339219723068548}, 'real_real': {'mean': 0.18782680630683898, 'stddev': 0.09988626866307053}, 'generated_generated': {'mean': 0.12181265652179718, 'stddev': 0.0800130366720006}}, 0.2: {'real_generated': {'mean': 0.33079734444618225, 'stddev': 0.09999341212957599}, 'real_real': {'mean': 0.18486612141132355, 'stddev': 0.11978507377993684}, 'generated_generated': {'mean': 0.200236676633358, 'stddev': 0.09010986589155703}}, 0.5: {'real_generated': {'mean': 0.22594461292028428, 'stddev': 0.07880121298658925}, 'real_real': {'mean': 0.16748639941215515, 'stddev': 0.07622498905718274}, 'generated_generated': {'mean': 0.06828635148704051, 'stddev': 0.03986840007348485}}, 1.0: {'real_generated': {'mean': 0.2853896528482437, 'stddev': 0.09568745026005727}, 'real_real': {'mean': 0.1503288932144642, 'stddev': 0.08616592047492139}, 'generated_generated': {'mean': 0.10293865650892257, 'stddev': 0.06552012587710518}}, 2.0: {'real_generated': {'mean': 0.1426343634724617, 'stddev': 0.043075462180569284}, 'real_real': {'mean': 0.13146761879324914, 'stddev': 0.08104647987122819}, 'generated_generated': {'mean': 0.05415386110544205, 'stddev': 0.0343040336455582}}, 5.0: {'real_generated': {'mean': 0.04201896041631699, 'stddev': 0.017758063108546582}, 'real_real': {'mean': 0.019261469319462775, 'stddev': 0.016052266206296892}, 'generated_generated': {'mean': 0.03356131389737129, 'stddev': 0.01793893933347565}}, 10.0: {'real_generated': {'mean': 0.017511437833309173, 'stddev': 0.007553143641228924}, 'real_real': {'mean': 0.013621512241661549, 'stddev': 0.009722301136876403}, 'generated_generated': {'mean': 0.013263619039207697, 'stddev': 0.01692358820110453}}}
==================================================
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_8_timestamp_2026-10-16T20:37:42.838811_trajectory_model_global_ode_solver_method_rk4_num_reflows_0.png
```
### `dopri8`
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : dopri8
  VT Trajectory Model  : global
  Num Reflows          : 0
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 63.36s
  Evaluations (t)      : 1000
  Eval Results         : {0.0038669902831315996: {'real_generated': {'mean': 0.23388259261846542, 'stddev': 0.13470110311898134}, 'real_real': {'mean': 0.15618237629532813, 'stddev': 0.08614998499087174}, 'generated_generated': {'mean': 0.1702297404408455, 'stddev': 0.08440706723031921}}, 0.009667475707828999: {'real_generated': {'mean': 0.2896098971366882, 'stddev': 0.07258985241633618}, 'real_real': {'mean': 0.1368156537413597, 'stddev': 0.06481567604759404}, 'generated_generated': {'mean': 0.17719248533248902, 'stddev': 0.09528989867704997}}, 0.01: {'real_generated': {'mean': 0.3924668312072754, 'stddev': 0.13072143211515216}, 'real_real': {'mean': 0.19167810976505278, 'stddev': 0.07029280837077755}, 'generated_generated': {'mean': 0.1111501544713974, 'stddev': 0.06038498359596992}}, 0.019334951415657997: {'real_generated': {'mean': 0.23819837272167205, 'stddev': 0.0509117898113814}, 'real_real': {'mean': 0.13589497059583663, 'stddev': 0.049904396807242034}, 'generated_generated': {'mean': 0.12702704817056656, 'stddev': 0.0490608605237038}}, 0.02: {'real_generated': {'mean': 0.3094451785087585, 'stddev': 0.060156314629616715}, 'real_real': {'mean': 0.16973766684532166, 'stddev': 0.04749504252484057}, 'generated_generated': {'mean': 0.14001620784401894, 'stddev': 0.10543047180919378}}, 0.038669902831315994: {'real_generated': {'mean': 0.329983714222908, 'stddev': 0.1344589984673403}, 'real_real': {'mean': 0.16571780815720558, 'stddev': 0.09110262064831774}, 'generated_generated': {'mean': 0.1878081426024437, 'stddev': 0.12405321417605102}}, 0.05: {'real_generated': {'mean': 0.2666875958442688, 'stddev': 0.0704602279140057}, 'real_real': {'mean': 0.17740901112556456, 'stddev': 0.10186174730580837}, 'generated_generated': {'mean': 0.173068667948246, 'stddev': 0.11501165779882142}}, 0.07733980566263199: {'real_generated': {'mean': 0.24240360856056214, 'stddev': 0.05688748721082788}, 'real_real': {'mean': 0.10576886534690857, 'stddev': 0.07197884778020706}, 'generated_generated': {'mean': 0.15295751243829728, 'stddev': 0.029460505719672257}}, 0.1: {'real_generated': {'mean': 0.22245972752571105, 'stddev': 0.09639971079052664}, 'real_real': {'mean': 0.18782680630683898, 'stddev': 0.09988626866307053}, 'generated_generated': {'mean': 0.10279516726732255, 'stddev': 0.030335121325154684}}, 0.2: {'real_generated': {'mean': 0.28163230419158936, 'stddev': 0.08194846591721965}, 'real_real': {'mean': 0.18486612141132355, 'stddev': 0.11978507377993684}, 'generated_generated': {'mean': 0.16361550539731978, 'stddev': 0.10998123000441815}}, 0.5: {'real_generated': {'mean': 0.2850402146577835, 'stddev': 0.06503978444173254}, 'real_real': {'mean': 0.16748639941215515, 'stddev': 0.07622498905718274}, 'generated_generated': {'mean': 0.12450003921985626, 'stddev': 0.05751085825711876}}, 1.0: {'real_generated': {'mean': 0.2703071892261505, 'stddev': 0.04375229992572659}, 'real_real': {'mean': 0.1503288932144642, 'stddev': 0.08616592047492139}, 'generated_generated': {'mean': 0.1384703427553177, 'stddev': 0.0492969323423074}}, 2.0: {'real_generated': {'mean': 0.1606363534927368, 'stddev': 0.03315065314004061}, 'real_real': {'mean': 0.13146761879324914, 'stddev': 0.08104647987122819}, 'generated_generated': {'mean': 0.08929542750120163, 'stddev': 0.051155375897280614}}, 5.0: {'real_generated': {'mean': 0.053848108649253844, 'stddev': 0.02596256726386268}, 'real_real': {'mean': 0.019261469319462775, 'stddev': 0.016052266206296892}, 'generated_generated': {'mean': 0.0214676427654922, 'stddev': 0.019635718411846085}}, 10.0: {'real_generated': {'mean': 0.018266652524471284, 'stddev': 0.005598973012385976}, 'real_real': {'mean': 0.013621512241661549, 'stddev': 0.009722301136876403}, 'generated_generated': {'mean': 0.01720188707113266, 'stddev': 0.01467517417412437}}}
==================================================
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_8_timestamp_2026-10-17T15:01:38.902953_trajectory_model_global_ode_solver_method_dopri8_num_reflows_0.png
```
## 2-RecFlow
### `euler`
```log
=================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : euler
  VT Trajectory Model  : global
  Num Reflows          : 1
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 5.60s
  Evaluations (t)      : 1000
  Eval Results         : {0.0038669902831315996: {'real_generated': {'mean': 0.4022770822048187, 'stddev': 0.07650903327073473}, 'real_real': {'mean': 0.15618237629532813, 'stddev': 0.08614998499087174}, 'generated_generated': {'mean': 0.11563918218016625, 'stddev': 0.07904277458890306}}, 0.009667475707828999: {'real_generated': {'mean': 0.38242179751396177, 'stddev': 0.0870629403446937}, 'real_real': {'mean': 0.1368156537413597, 'stddev': 0.06481567604759404}, 'generated_generated': {'mean': 0.14918185919523239, 'stddev': 0.0456642028529148}}, 0.01: {'real_generated': {'mean': 0.39500497579574584, 'stddev': 0.05924426196917381}, 'real_real': {'mean': 0.19167810976505278, 'stddev': 0.07029280837077755}, 'generated_generated': {'mean': 0.11355536580085754, 'stddev': 0.04945745331564375}}, 0.019334951415657997: {'real_generated': {'mean': 0.38996040225028994, 'stddev': 0.03936462535654634}, 'real_real': {'mean': 0.13589497059583663, 'stddev': 0.049904396807242034}, 'generated_generated': {'mean': 0.11903072819113732, 'stddev': 0.10001271189999168}}, 0.02: {'real_generated': {'mean': 0.38284971117973327, 'stddev': 0.10813980461366443}, 'real_real': {'mean': 0.16973766684532166, 'stddev': 0.04749504252484057}, 'generated_generated': {'mean': 0.08179149627685547, 'stddev': 0.014025443682257343}}, 0.038669902831315994: {'real_generated': {'mean': 0.40674978494644165, 'stddev': 0.14147989327971042}, 'real_real': {'mean': 0.16571780815720558, 'stddev': 0.09110262064831774}, 'generated_generated': {'mean': 0.16746592819690703, 'stddev': 0.05690038079553309}}, 0.05: {'real_generated': {'mean': 0.42917461693286896, 'stddev': 0.18004409350644576}, 'real_real': {'mean': 0.17740901112556456, 'stddev': 0.10186174730580837}, 'generated_generated': {'mean': 0.08945962861180305, 'stddev': 0.03277010831011424}}, 0.07733980566263199: {'real_generated': {'mean': 0.3916091561317444, 'stddev': 0.1495968761786895}, 'real_real': {'mean': 0.10576886534690857, 'stddev': 0.07197884778020706}, 'generated_generated': {'mean': 0.113139308989048, 'stddev': 0.042978821422227226}}, 0.1: {'real_generated': {'mean': 0.36967151165008544, 'stddev': 0.117239933824491}, 'real_real': {'mean': 0.18782680630683898, 'stddev': 0.09988626866307053}, 'generated_generated': {'mean': 0.132205531001091, 'stddev': 0.0388395624597523}}, 0.2: {'real_generated': {'mean': 0.43419055342674256, 'stddev': 0.05888467931197412}, 'real_real': {'mean': 0.18486612141132355, 'stddev': 0.11978507377993684}, 'generated_generated': {'mean': 0.07013117000460625, 'stddev': 0.04625697120686678}}, 0.5: {'real_generated': {'mean': 0.45775063037872316, 'stddev': 0.042194283959899825}, 'real_real': {'mean': 0.16748639941215515, 'stddev': 0.07622498905718274}, 'generated_generated': {'mean': 0.15528463423252106, 'stddev': 0.08481666606717944}}, 1.0: {'real_generated': {'mean': 0.33323135375976565, 'stddev': 0.0716502438572372}, 'real_real': {'mean': 0.1503288932144642, 'stddev': 0.08616592047492139}, 'generated_generated': {'mean': 0.07053330317139625, 'stddev': 0.03989843558327943}}, 2.0: {'real_generated': {'mean': 0.22253560423851013, 'stddev': 0.08109000303089844}, 'real_real': {'mean': 0.13146761879324914, 'stddev': 0.08104647987122819}, 'generated_generated': {'mean': 0.05052072592079639, 'stddev': 0.026697115195318137}}, 5.0: {'real_generated': {'mean': 0.06376960426568985, 'stddev': 0.04172558254506629}, 'real_real': {'mean': 0.019261469319462775, 'stddev': 0.016052266206296892}, 'generated_generated': {'mean': 0.03772421553730965, 'stddev': 0.022573370896386023}}, 10.0: {'real_generated': {'mean': 0.04042111039161682, 'stddev': 0.028607500218816133}, 'real_real': {'mean': 0.013621512241661549, 'stddev': 0.009722301136876403}, 'generated_generated': {'mean': 0.012248128280043602, 'stddev': 0.010037132224844878}}}
==================================================
```
### `rk4`
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : rk4
  VT Trajectory Model  : global
  Num Reflows          : 1
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 21.13s
  Evaluations (t)      : 1000
  Eval Results         : {0.0038669902831315996: {'real_generated': {'mean': 0.3271858632564545, 'stddev': 0.05214318308685702}, 'real_real': {'mean': 0.15618237629532813, 'stddev': 0.08614998499087174}, 'generated_generated': {'mean': 0.1118953637778759, 'stddev': 0.06452279243550368}}, 0.009667475707828999: {'real_generated': {'mean': 0.4459040522575378, 'stddev': 0.18874690511419492}, 'real_real': {'mean': 0.1368156537413597, 'stddev': 0.06481567604759404}, 'generated_generated': {'mean': 0.14915813878178596, 'stddev': 0.08361678513506707}}, 0.01: {'real_generated': {'mean': 0.3696045517921448, 'stddev': 0.15213255975945117}, 'real_real': {'mean': 0.19167810976505278, 'stddev': 0.07029280837077755}, 'generated_generated': {'mean': 0.11554945409297943, 'stddev': 0.06235475833816301}}, 0.019334951415657997: {'real_generated': {'mean': 0.3922003507614136, 'stddev': 0.07934518036311007}, 'real_real': {'mean': 0.13589497059583663, 'stddev': 0.049904396807242034}, 'generated_generated': {'mean': 0.12738120853900908, 'stddev': 0.03447253723844003}}, 0.02: {'real_generated': {'mean': 0.39075979590415955, 'stddev': 0.14763592828262218}, 'real_real': {'mean': 0.16973766684532166, 'stddev': 0.04749504252484057}, 'generated_generated': {'mean': 0.1273224249482155, 'stddev': 0.06633412499921376}}, 0.038669902831315994: {'real_generated': {'mean': 0.37916184067726133, 'stddev': 0.059973680333367994}, 'real_real': {'mean': 0.16571780815720558, 'stddev': 0.09110262064831774}, 'generated_generated': {'mean': 0.11912613064050674, 'stddev': 0.026694535300890995}}, 0.05: {'real_generated': {'mean': 0.3824977219104767, 'stddev': 0.06750946244168052}, 'real_real': {'mean': 0.17740901112556456, 'stddev': 0.10186174730580837}, 'generated_generated': {'mean': 0.09688757210969925, 'stddev': 0.03886405097579566}}, 0.07733980566263199: {'real_generated': {'mean': 0.3618424654006958, 'stddev': 0.12393622750671741}, 'real_real': {'mean': 0.10576886534690857, 'stddev': 0.07197884778020706}, 'generated_generated': {'mean': 0.06417374610900879, 'stddev': 0.026260731238718835}}, 0.1: {'real_generated': {'mean': 0.31955508291721346, 'stddev': 0.10755826979980634}, 'real_real': {'mean': 0.18782680630683898, 'stddev': 0.09988626866307053}, 'generated_generated': {'mean': 0.08370020836591721, 'stddev': 0.04494021694801847}}, 0.2: {'real_generated': {'mean': 0.47853938937187196, 'stddev': 0.20852464086381342}, 'real_real': {'mean': 0.18486612141132355, 'stddev': 0.11978507377993684}, 'generated_generated': {'mean': 0.15646352767944335, 'stddev': 0.060032316699620905}}, 0.5: {'real_generated': {'mean': 0.4410031199455261, 'stddev': 0.11883733970911714}, 'real_real': {'mean': 0.16748639941215515, 'stddev': 0.07622498905718274}, 'generated_generated': {'mean': 0.11596427857875824, 'stddev': 0.04134721934201939}}, 1.0: {'real_generated': {'mean': 0.35817912220954895, 'stddev': 0.10663684521954275}, 'real_real': {'mean': 0.1503288932144642, 'stddev': 0.08616592047492139}, 'generated_generated': {'mean': 0.14380307346582413, 'stddev': 0.0654612917941617}}, 2.0: {'real_generated': {'mean': 0.17565725445747377, 'stddev': 0.060838694218547214}, 'real_real': {'mean': 0.13146761879324914, 'stddev': 0.08104647987122819}, 'generated_generated': {'mean': 0.06541335731744766, 'stddev': 0.03415351498003018}}, 5.0: {'real_generated': {'mean': 0.06341473758220673, 'stddev': 0.02793972378936717}, 'real_real': {'mean': 0.019261469319462775, 'stddev': 0.016052266206296892}, 'generated_generated': {'mean': 0.019668570533394814, 'stddev': 0.008133516948257964}}, 10.0: {'real_generated': {'mean': 0.029299068450927734, 'stddev': 0.01844478008831242}, 'real_real': {'mean': 0.013621512241661549, 'stddev': 0.009722301136876403}, 'generated_generated': {'mean': 0.010317901149392128, 'stddev': 0.009863606672485002}}}
==================================================
```
### `dopri8`
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : dopri8
  VT Trajectory Model  : global
  Num Reflows          : 1
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 35.78s
  Evaluations (t)      : 1000
  Eval Results         : {0.0038669902831315996: {'real_generated': {'mean': 0.4034880638122559, 'stddev': 0.07796718271549512}, 'real_real': {'mean': 0.15618237629532813, 'stddev': 0.08614998499087174}, 'generated_generated': {'mean': 0.06172999292612076, 'stddev': 0.025819936074738262}}, 0.009667475707828999: {'real_generated': {'mean': 0.4414689242839813, 'stddev': 0.16788276422637216}, 'real_real': {'mean': 0.1368156537413597, 'stddev': 0.06481567604759404}, 'generated_generated': {'mean': 0.13631632998585702, 'stddev': 0.08055766524212257}}, 0.01: {'real_generated': {'mean': 0.41980360746383666, 'stddev': 0.08962833576404818}, 'real_real': {'mean': 0.19167810976505278, 'stddev': 0.07029280837077755}, 'generated_generated': {'mean': 0.08778851255774497, 'stddev': 0.058516082865108544}}, 0.019334951415657997: {'real_generated': {'mean': 0.4500507593154907, 'stddev': 0.09361097611816185}, 'real_real': {'mean': 0.13589497059583663, 'stddev': 0.049904396807242034}, 'generated_generated': {'mean': 0.12928165048360823, 'stddev': 0.06767058966131256}}, 0.02: {'real_generated': {'mean': 0.38506972789764404, 'stddev': 0.1323097193443457}, 'real_real': {'mean': 0.16973766684532166, 'stddev': 0.04749504252484057}, 'generated_generated': {'mean': 0.10190683752298355, 'stddev': 0.03080823901007903}}, 0.038669902831315994: {'real_generated': {'mean': 0.39170361757278443, 'stddev': 0.09953319551654823}, 'real_real': {'mean': 0.16571780815720558, 'stddev': 0.09110262064831774}, 'generated_generated': {'mean': 0.14415571838617325, 'stddev': 0.04586030593314648}}, 0.05: {'real_generated': {'mean': 0.37553409337997434, 'stddev': 0.09308267909811654}, 'real_real': {'mean': 0.17740901112556456, 'stddev': 0.10186174730580837}, 'generated_generated': {'mean': 0.07648236528038979, 'stddev': 0.051910592947331045}}, 0.07733980566263199: {'real_generated': {'mean': 0.33551437258720396, 'stddev': 0.08441513375603994}, 'real_real': {'mean': 0.10576886534690857, 'stddev': 0.07197884778020706}, 'generated_generated': {'mean': 0.10073834210634232, 'stddev': 0.04843850849712439}}, 0.1: {'real_generated': {'mean': 0.4233066439628601, 'stddev': 0.08363028442031148}, 'real_real': {'mean': 0.18782680630683898, 'stddev': 0.09988626866307053}, 'generated_generated': {'mean': 0.12745710611343383, 'stddev': 0.047645111915316084}}, 0.2: {'real_generated': {'mean': 0.4299877107143402, 'stddev': 0.012561123143702706}, 'real_real': {'mean': 0.18486612141132355, 'stddev': 0.11978507377993684}, 'generated_generated': {'mean': 0.13646263480186463, 'stddev': 0.11696350646025404}}, 0.5: {'real_generated': {'mean': 0.4512069821357727, 'stddev': 0.14282155839498864}, 'real_real': {'mean': 0.16748639941215515, 'stddev': 0.07622498905718274}, 'generated_generated': {'mean': 0.12379559054970742, 'stddev': 0.1041432108078379}}, 1.0: {'real_generated': {'mean': 0.43156073093414304, 'stddev': 0.15719345106982385}, 'real_real': {'mean': 0.1503288932144642, 'stddev': 0.08616592047492139}, 'generated_generated': {'mean': 0.11945398971438408, 'stddev': 0.06237056867003378}}, 2.0: {'real_generated': {'mean': 0.17490846067667007, 'stddev': 0.06937956292380243}, 'real_real': {'mean': 0.13146761879324914, 'stddev': 0.08104647987122819}, 'generated_generated': {'mean': 0.06637205332517623, 'stddev': 0.02803545291476041}}, 5.0: {'real_generated': {'mean': 0.06398098394274712, 'stddev': 0.01591772835679144}, 'real_real': {'mean': 0.019261469319462775, 'stddev': 0.016052266206296892}, 'generated_generated': {'mean': 0.0124377959407866, 'stddev': 0.007329221983173293}}, 10.0: {'real_generated': {'mean': 0.029687750339508056, 'stddev': 0.01709480528241505}, 'real_real': {'mean': 0.013621512241661549, 'stddev': 0.009722301136876403}, 'generated_generated': {'mean': 0.005253918841481209, 'stddev': 0.002178934928907616}}}
==================================================
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_8_timestamp_2026-10-17T22:47:29.221423_trajectory_model_global_ode_solver_method_dopri8_num_reflows_1.png
```