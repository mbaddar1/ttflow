# Desc 

**Objective** :compare two models : one without reflow and (1-Rectified-Flow)
and another with one-reflow iteration (2-Rectified-Flow)
the ReFlow and inference both use euler as the method for ODESOLVE
for speedup

**Dataset** : gaussian mixture n_components = 4

# Models and Training 

## Model Paths ( For experiments)

### 1-Rectified-Flow / 0-Reflow

```text
models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_4_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-13T23:48:45.910171.pkl
```

### 2-Rectified-Flow / 1-Reflow

```text
models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_4_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-14T00:26:57.171654.pkl
```
## ( A ) [1-Rectified Flow (0-Reflow)]

### Model
* Path
```log
models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_4_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-13T23:48:45.910171.pkl 
```

* Train log 
```log
 /home/mbaddar/Documents/mbaddar/ttflow/.venv/bin/python /home/mbaddar/Documents/mbaddar/ttflow/ttflow/training/nn/vt_trainer.py --data-snapshot-filepath data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_4_data_dim_2_n_train_test_eval_500000_200000_200000.pkl --vt-model nn --vt-opt adam --trajectory-model global --reflow-iterations 0 --nn-epochs 300 --reflow-iterations 0 --ode-solver-method euler 
2026-10-13 23:48:45.910 | INFO     | __main__:<module>:154 - args = Namespace(data_snapshot_filepath='data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_4_data_dim_2_n_train_test_eval_500000_200000_200000.pkl', use_gpu=True, vt_model='nn', vt_opt='adam', trajectory_model='global', ode_solver_method='euler', rectification_linearity='linear', reflow_iterations=0, nn_epochs=300, nn_hidden_dim=128, batch_size=2048, nn_depth=2, lr=0.001, nn_reg_lambda=0.001, patience=10, min_rel_loss_delta=5e-05, dry_run=False, logs_dir='logs', output_model_dir='models/experimental')
2026-10-13 23:48:45.910 | INFO     | __main__:<module>:166 - is GPU/CUDA available = True
2026-10-13 23:48:45.910 | INFO     | __main__:<module>:169 - device = cuda
2026-10-13 23:48:45.997 | INFO     | __main__:<module>:174 - Using data snapshot: data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_4_data_dim_2_n_train_test_eval_500000_200000_200000.pkl
2026-10-13 23:48:45.998 | INFO     | ttflow.utils:check_system_memory_and_swap:56 - ram = 30.96, swap = 40.0 GB
2026-10-13 23:48:46.014 | INFO     | __main__:<module>:203 - data pkl size = 0.01 GB
2026-10-13 23:48:46.017 | INFO     | __main__:<module>:214 - Generating training data for the global trajectory training process..
2026-10-13 23:48:46.195 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-13 23:48:46.769 | INFO     | __main__:<module>:241 - ***********************************************************Training Model Specs:
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
reflow iterations:   0%|          | 0/1 [00:00<?, ?it/s]2026-10-13 23:48:46.769 | INFO     | __main__:<module>:266 - Reflow iteration: 0
2026-10-13 23:48:47.685 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:57 - @epoch = 0 (before training), loss_fn = MSELoss(), loss_value = 52.067194
2026-10-13 23:48:47.690 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:64 - MSELoss(test) before training:52.06718826293945

Epoch:   0%|          | 0/300 [00:00<?, ?it/s]2026-10-13 23:48:50.657 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 1, smoothed-train-loss = 14.515076 | smoothed-test-loss = 9.825738 | rel_loss_delta = inf | loss_stagnation_count = 0

Epoch:   0%|          | 1/300 [00:02<14:47,  2.97s/it]2026-10-13 23:48:53.691 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 2, smoothed-train-loss = 9.999266 | smoothed-test-loss = 9.340540 | rel_loss_delta = inf | loss_stagnation_count = 0

Epoch:   1%|          | 2/300 [00:06<14:55,  3.01s/it]2026-10-13 23:48:56.824 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 3, smoothed-train-loss = 9.257495 | smoothed-test-loss = 9.043109 | rel_loss_delta = -0.07418246457893443 | loss_stagnation_count = 0

Epoch:   1%|          | 3/300 [00:09<15:10,  3.06s/it]2026-10-13 23:48:59.847 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 4, smoothed-train-loss = 8.941469 | smoothed-test-loss = 8.769622 | rel_loss_delta = -0.034137313225557665 | loss_stagnation_count = 0

Epoch:   1%|▏         | 4/300 [00:12<15:02,  3.05s/it]2026-10-13 23:49:02.852 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 5, smoothed-train-loss = 8.696742 | smoothed-test-loss = 8.565169 | rel_loss_delta = -0.027369910586918584 | loss_stagnation_count = 0

Epoch:   2%|▏         | 5/300 [00:15<14:54,  3.03s/it]2026-10-13 23:49:05.760 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 6, smoothed-train-loss = 8.516046 | smoothed-test-loss = 8.403814 | rel_loss_delta = -0.020777496846693572 | loss_stagnation_count = 0

Epoch:   2%|▏         | 6/300 [00:18<14:39,  2.99s/it]2026-10-13 23:49:08.697 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 7, smoothed-train-loss = 8.393574 | smoothed-test-loss = 8.351223 | rel_loss_delta = -0.014381292950288218 | loss_stagnation_count = 0

Epoch:   2%|▏         | 7/300 [00:21<14:30,  2.97s/it]2026-10-13 23:49:11.717 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 8, smoothed-train-loss = 8.316751 | smoothed-test-loss = 8.281039 | rel_loss_delta = -0.009152632316175423 | loss_stagnation_count = 0

Epoch:   3%|▎         | 8/300 [00:24<14:32,  2.99s/it]2026-10-13 23:49:14.587 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 9, smoothed-train-loss = 8.267869 | smoothed-test-loss = 8.280109 | rel_loss_delta = -0.005877514414777539 | loss_stagnation_count = 0

Epoch:   3%|▎         | 9/300 [00:26<14:18,  2.95s/it]2026-10-13 23:49:17.747 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 10, smoothed-train-loss = 8.233223 | smoothed-test-loss = 8.219522 | rel_loss_delta = -0.0041903802368249915 | loss_stagnation_count = 0

Epoch:   3%|▎         | 10/300 [00:30<14:34,  3.02s/it]2026-10-13 23:49:20.812 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 11, smoothed-train-loss = 8.209432 | smoothed-test-loss = 8.192587 | rel_loss_delta = -0.002889628954762106 | loss_stagnation_count = 0

Epoch:   4%|▎         | 11/300 [00:33<14:35,  3.03s/it]2026-10-13 23:49:23.742 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 12, smoothed-train-loss = 8.193375 | smoothed-test-loss = 8.169210 | rel_loss_delta = -0.001956025373225313 | loss_stagnation_count = 0

Epoch:   4%|▍         | 12/300 [00:36<14:23,  3.00s/it]2026-10-13 23:49:26.534 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 13, smoothed-train-loss = 8.185264 | smoothed-test-loss = 8.162140 | rel_loss_delta = -0.000989897065466921 | loss_stagnation_count = 0

Epoch:   4%|▍         | 13/300 [00:38<14:02,  2.94s/it]2026-10-13 23:49:29.457 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 14, smoothed-train-loss = 8.177658 | smoothed-test-loss = 8.182431 | rel_loss_delta = -0.0009292637794815022 | loss_stagnation_count = 0

Epoch:   5%|▍         | 14/300 [00:41<13:58,  2.93s/it]2026-10-13 23:49:32.509 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 15, smoothed-train-loss = 8.168772 | smoothed-test-loss = 8.168841 | rel_loss_delta = -0.0010865214530624 | loss_stagnation_count = 0

Epoch:   5%|▌         | 15/300 [00:44<14:06,  2.97s/it]2026-10-13 23:49:35.423 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 16, smoothed-train-loss = 8.165462 | smoothed-test-loss = 8.162814 | rel_loss_delta = -0.00040531589018568635 | loss_stagnation_count = 0

Epoch:   5%|▌         | 16/300 [00:47<13:58,  2.95s/it]2026-10-13 23:49:38.230 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 17, smoothed-train-loss = 8.157434 | smoothed-test-loss = 8.139472 | rel_loss_delta = -0.0009831110107923963 | loss_stagnation_count = 0

Epoch:   6%|▌         | 17/300 [00:50<13:43,  2.91s/it]2026-10-13 23:49:41.284 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 18, smoothed-train-loss = 8.154596 | smoothed-test-loss = 8.140311 | rel_loss_delta = -0.0003479545568129515 | loss_stagnation_count = 0

Epoch:   6%|▌         | 18/300 [00:53<13:52,  2.95s/it]2026-10-13 23:49:44.288 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 19, smoothed-train-loss = 8.152158 | smoothed-test-loss = 8.138265 | rel_loss_delta = -0.0002988578489158726 | loss_stagnation_count = 0

Epoch:   6%|▋         | 19/300 [00:56<13:53,  2.97s/it]2026-10-13 23:49:47.363 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 20, smoothed-train-loss = 8.150089 | smoothed-test-loss = 8.133917 | rel_loss_delta = -0.00025379836446195797 | loss_stagnation_count = 0

Epoch:   7%|▋         | 20/300 [00:59<13:59,  3.00s/it]2026-10-13 23:49:50.291 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 21, smoothed-train-loss = 8.149015 | smoothed-test-loss = 8.140872 | rel_loss_delta = -0.0001318737085504515 | loss_stagnation_count = 0

Epoch:   7%|▋         | 21/300 [01:02<13:50,  2.98s/it]2026-10-13 23:49:53.269 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 22, smoothed-train-loss = 8.147899 | smoothed-test-loss = 8.132637 | rel_loss_delta = -0.00013686538315603735 | loss_stagnation_count = 0

Epoch:   7%|▋         | 22/300 [01:05<13:47,  2.98s/it]2026-10-13 23:49:56.302 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 23, smoothed-train-loss = 8.142835 | smoothed-test-loss = 8.117263 | rel_loss_delta = -0.0006215543229518866 | loss_stagnation_count = 0

Epoch:   8%|▊         | 23/300 [01:08<13:49,  2.99s/it]2026-10-13 23:49:59.256 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 24, smoothed-train-loss = 8.141093 | smoothed-test-loss = 8.124886 | rel_loss_delta = -0.0002139391631135979 | loss_stagnation_count = 0

Epoch:   8%|▊         | 24/300 [01:11<13:43,  2.98s/it]2026-10-13 23:50:02.205 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 25, smoothed-train-loss = 8.142408 | smoothed-test-loss = 8.174352 | rel_loss_delta = 0.0001614840492001411 | loss_stagnation_count = 0

Epoch:   8%|▊         | 25/300 [01:14<13:37,  2.97s/it]2026-10-13 23:50:05.175 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 26, smoothed-train-loss = 8.140469 | smoothed-test-loss = 8.132703 | rel_loss_delta = -0.0002381373630052354 | loss_stagnation_count = 0

Epoch:   9%|▊         | 26/300 [01:17<13:34,  2.97s/it]2026-10-13 23:50:08.141 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 27, smoothed-train-loss = 8.141791 | smoothed-test-loss = 8.139027 | rel_loss_delta = 0.0001624547838911608 | loss_stagnation_count = 0

Epoch:   9%|▉         | 27/300 [01:20<13:30,  2.97s/it]2026-10-13 23:50:11.176 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 28, smoothed-train-loss = 8.137412 | smoothed-test-loss = 8.127631 | rel_loss_delta = -0.0005378313724125061 | loss_stagnation_count = 0

Epoch:   9%|▉         | 28/300 [01:23<13:33,  2.99s/it]2026-10-13 23:50:14.138 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 29, smoothed-train-loss = 8.135864 | smoothed-test-loss = 8.131857 | rel_loss_delta = -0.00019022248547362686 | loss_stagnation_count = 0

Epoch:  10%|▉         | 29/300 [01:26<13:27,  2.98s/it]2026-10-13 23:50:17.276 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 30, smoothed-train-loss = 8.136762 | smoothed-test-loss = 8.118512 | rel_loss_delta = 0.000110301195465193 | loss_stagnation_count = 0

Epoch:  10%|█         | 30/300 [01:29<13:37,  3.03s/it]2026-10-13 23:50:20.211 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 31, smoothed-train-loss = 8.136041 | smoothed-test-loss = 8.148996 | rel_loss_delta = -8.853475028326244e-05 | loss_stagnation_count = 0

Epoch:  10%|█         | 31/300 [01:32<13:27,  3.00s/it]2026-10-13 23:50:23.123 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 32, smoothed-train-loss = 8.133634 | smoothed-test-loss = 8.164945 | rel_loss_delta = -0.00029581495328080535 | loss_stagnation_count = 0

Epoch:  11%|█         | 32/300 [01:35<13:16,  2.97s/it]2026-10-13 23:50:25.926 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 33, smoothed-train-loss = 8.128244 | smoothed-test-loss = 8.141033 | rel_loss_delta = -0.0006627468373062051 | loss_stagnation_count = 0

Epoch:  11%|█         | 33/300 [01:38<13:00,  2.92s/it]2026-10-13 23:50:28.950 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 34, smoothed-train-loss = 8.133695 | smoothed-test-loss = 8.133485 | rel_loss_delta = 0.0006706366495669312 | loss_stagnation_count = 0

Epoch:  11%|█▏        | 34/300 [01:41<13:05,  2.95s/it]2026-10-13 23:50:31.967 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 35, smoothed-train-loss = 8.135991 | smoothed-test-loss = 8.132552 | rel_loss_delta = 0.0002822413804496007 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 35/300 [01:44<13:07,  2.97s/it]2026-10-13 23:50:34.922 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 36, smoothed-train-loss = 8.131106 | smoothed-test-loss = 8.118146 | rel_loss_delta = -0.0006004074082306435 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 36/300 [01:47<13:03,  2.97s/it]2026-10-13 23:50:37.837 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 37, smoothed-train-loss = 8.132727 | smoothed-test-loss = 8.128674 | rel_loss_delta = 0.00019935333906038235 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 37/300 [01:50<12:56,  2.95s/it]2026-10-13 23:50:40.814 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 38, smoothed-train-loss = 8.133423 | smoothed-test-loss = 8.120019 | rel_loss_delta = 8.5627923123408e-05 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 38/300 [01:53<12:55,  2.96s/it]2026-10-13 23:50:43.772 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 39, smoothed-train-loss = 8.127730 | smoothed-test-loss = 8.127749 | rel_loss_delta = -0.0006999252247772154 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 39/300 [01:56<12:52,  2.96s/it]2026-10-13 23:50:46.811 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 40, smoothed-train-loss = 8.130116 | smoothed-test-loss = 8.125131 | rel_loss_delta = 0.00029351072204732246 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 40/300 [01:59<12:55,  2.98s/it]2026-10-13 23:50:49.707 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 41, smoothed-train-loss = 8.133524 | smoothed-test-loss = 8.108494 | rel_loss_delta = 0.00041919351221700167 | loss_stagnation_count = 0

Epoch:  14%|█▎        | 41/300 [02:02<12:45,  2.96s/it]2026-10-13 23:50:52.635 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 42, smoothed-train-loss = 8.130803 | smoothed-test-loss = 8.152948 | rel_loss_delta = -0.000334525168261646 | loss_stagnation_count = 0

Epoch:  14%|█▍        | 42/300 [02:04<12:40,  2.95s/it]2026-10-13 23:50:55.693 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 43, smoothed-train-loss = 8.130426 | smoothed-test-loss = 8.113537 | rel_loss_delta = -4.638017817090114e-05 | loss_stagnation_count = 1

Epoch:  14%|█▍        | 43/300 [02:08<12:46,  2.98s/it]2026-10-13 23:50:58.692 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 44, smoothed-train-loss = 8.126620 | smoothed-test-loss = 8.152151 | rel_loss_delta = -0.000468109086600442 | loss_stagnation_count = 0

Epoch:  15%|█▍        | 44/300 [02:11<12:44,  2.99s/it]2026-10-13 23:51:01.683 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 45, smoothed-train-loss = 8.128257 | smoothed-test-loss = 8.161743 | rel_loss_delta = 0.0002013756724690872 | loss_stagnation_count = 0

Epoch:  15%|█▌        | 45/300 [02:13<12:41,  2.99s/it]2026-10-13 23:51:04.655 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 46, smoothed-train-loss = 8.127357 | smoothed-test-loss = 8.115733 | rel_loss_delta = -0.00011071151136535547 | loss_stagnation_count = 0

Epoch:  15%|█▌        | 46/300 [02:16<12:37,  2.98s/it]2026-10-13 23:51:07.628 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 47, smoothed-train-loss = 8.127619 | smoothed-test-loss = 8.131955 | rel_loss_delta = 3.223330864581373e-05 | loss_stagnation_count = 1

Epoch:  16%|█▌        | 47/300 [02:19<12:33,  2.98s/it]2026-10-13 23:51:10.538 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 48, smoothed-train-loss = 8.126663 | smoothed-test-loss = 8.111166 | rel_loss_delta = -0.00011753605935931672 | loss_stagnation_count = 0

Epoch:  16%|█▌        | 48/300 [02:22<12:25,  2.96s/it]2026-10-13 23:51:13.354 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 49, smoothed-train-loss = 8.125074 | smoothed-test-loss = 8.110116 | rel_loss_delta = -0.00019556199791261416 | loss_stagnation_count = 0

Epoch:  16%|█▋        | 49/300 [02:25<12:11,  2.92s/it]2026-10-13 23:51:16.413 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 50, smoothed-train-loss = 8.123162 | smoothed-test-loss = 8.103899 | rel_loss_delta = -0.00023530196385141265 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 50/300 [02:28<12:19,  2.96s/it]2026-10-13 23:51:19.427 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 51, smoothed-train-loss = 8.123243 | smoothed-test-loss = 8.109908 | rel_loss_delta = 9.909027460913878e-06 | loss_stagnation_count = 1

Epoch:  17%|█▋        | 51/300 [02:31<12:20,  2.98s/it]2026-10-13 23:51:22.386 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 52, smoothed-train-loss = 8.122697 | smoothed-test-loss = 8.114734 | rel_loss_delta = -6.719483123158714e-05 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 52/300 [02:34<12:16,  2.97s/it]2026-10-13 23:51:25.307 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 53, smoothed-train-loss = 8.124901 | smoothed-test-loss = 8.121679 | rel_loss_delta = 0.00027137507213122436 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 53/300 [02:37<12:10,  2.96s/it]2026-10-13 23:51:28.394 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 54, smoothed-train-loss = 8.124586 | smoothed-test-loss = 8.127595 | rel_loss_delta = -3.874872251587212e-05 | loss_stagnation_count = 1

Epoch:  18%|█▊        | 54/300 [02:40<12:16,  3.00s/it]2026-10-13 23:51:31.479 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 55, smoothed-train-loss = 8.124727 | smoothed-test-loss = 8.120680 | rel_loss_delta = 1.7344196659489856e-05 | loss_stagnation_count = 2

Epoch:  18%|█▊        | 55/300 [02:43<12:20,  3.02s/it]2026-10-13 23:51:34.490 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 56, smoothed-train-loss = 8.123304 | smoothed-test-loss = 8.136155 | rel_loss_delta = -0.000175217069425104 | loss_stagnation_count = 0

Epoch:  19%|█▊        | 56/300 [02:46<12:16,  3.02s/it]2026-10-13 23:51:37.302 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 57, smoothed-train-loss = 8.125093 | smoothed-test-loss = 8.104097 | rel_loss_delta = 0.00022030020549548523 | loss_stagnation_count = 0

Epoch:  19%|█▉        | 57/300 [02:49<11:58,  2.96s/it]2026-10-13 23:51:40.222 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 58, smoothed-train-loss = 8.124205 | smoothed-test-loss = 8.137938 | rel_loss_delta = -0.00010931532259282876 | loss_stagnation_count = 0

Epoch:  19%|█▉        | 58/300 [02:52<11:52,  2.95s/it]2026-10-13 23:51:43.186 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 59, smoothed-train-loss = 8.119557 | smoothed-test-loss = 8.114101 | rel_loss_delta = -0.0005721671299827281 | loss_stagnation_count = 0

Epoch:  20%|█▉        | 59/300 [02:55<11:51,  2.95s/it]2026-10-13 23:51:46.208 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 60, smoothed-train-loss = 8.123370 | smoothed-test-loss = 8.115515 | rel_loss_delta = 0.00046969927107365204 | loss_stagnation_count = 0

Epoch:  20%|██        | 60/300 [02:58<11:53,  2.97s/it]2026-10-13 23:51:49.171 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 61, smoothed-train-loss = 8.122812 | smoothed-test-loss = 8.125308 | rel_loss_delta = -6.875986509115745e-05 | loss_stagnation_count = 0

Epoch:  20%|██        | 61/300 [03:01<11:49,  2.97s/it]2026-10-13 23:51:52.097 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 62, smoothed-train-loss = 8.121886 | smoothed-test-loss = 8.099882 | rel_loss_delta = -0.00011396772323867473 | loss_stagnation_count = 0

Epoch:  21%|██        | 62/300 [03:04<11:43,  2.96s/it]2026-10-13 23:51:55.144 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 63, smoothed-train-loss = 8.120393 | smoothed-test-loss = 8.112045 | rel_loss_delta = -0.00018387176641527966 | loss_stagnation_count = 0

Epoch:  21%|██        | 63/300 [03:07<11:47,  2.98s/it]2026-10-13 23:51:58.084 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 64, smoothed-train-loss = 8.119133 | smoothed-test-loss = 8.115058 | rel_loss_delta = -0.00015515820112112607 | loss_stagnation_count = 0

Epoch:  21%|██▏       | 64/300 [03:10<11:41,  2.97s/it]2026-10-13 23:52:00.997 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 65, smoothed-train-loss = 8.116068 | smoothed-test-loss = 8.107692 | rel_loss_delta = -0.00037746872767351996 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 65/300 [03:13<11:34,  2.95s/it]2026-10-13 23:52:03.976 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 66, smoothed-train-loss = 8.121093 | smoothed-test-loss = 8.116933 | rel_loss_delta = 0.0006191496243916878 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 66/300 [03:16<11:32,  2.96s/it]2026-10-13 23:52:06.909 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 67, smoothed-train-loss = 8.120995 | smoothed-test-loss = 8.135937 | rel_loss_delta = -1.211467567512869e-05 | loss_stagnation_count = 1

Epoch:  22%|██▏       | 67/300 [03:19<11:27,  2.95s/it]2026-10-13 23:52:09.867 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 68, smoothed-train-loss = 8.120673 | smoothed-test-loss = 8.121826 | rel_loss_delta = -3.961456465746467e-05 | loss_stagnation_count = 2

Epoch:  23%|██▎       | 68/300 [03:22<11:25,  2.95s/it]2026-10-13 23:52:12.734 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 69, smoothed-train-loss = 8.120949 | smoothed-test-loss = 8.123757 | rel_loss_delta = 3.394956656223444e-05 | loss_stagnation_count = 3

Epoch:  23%|██▎       | 69/300 [03:25<11:16,  2.93s/it]2026-10-13 23:52:15.675 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 70, smoothed-train-loss = 8.117130 | smoothed-test-loss = 8.121486 | rel_loss_delta = -0.00047019498649087115 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 70/300 [03:27<11:14,  2.93s/it]2026-10-13 23:52:18.611 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 71, smoothed-train-loss = 8.117415 | smoothed-test-loss = 8.110542 | rel_loss_delta = 3.5011171332057944e-05 | loss_stagnation_count = 1

Epoch:  24%|██▎       | 71/300 [03:30<11:11,  2.93s/it]2026-10-13 23:52:21.623 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 72, smoothed-train-loss = 8.121896 | smoothed-test-loss = 8.099983 | rel_loss_delta = 0.0005521449468452423 | loss_stagnation_count = 0

Epoch:  24%|██▍       | 72/300 [03:33<11:14,  2.96s/it]2026-10-13 23:52:24.482 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 73, smoothed-train-loss = 8.116824 | smoothed-test-loss = 8.109049 | rel_loss_delta = -0.0006245026285984386 | loss_stagnation_count = 0

Epoch:  24%|██▍       | 73/300 [03:36<11:04,  2.93s/it]2026-10-13 23:52:27.479 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 74, smoothed-train-loss = 8.122514 | smoothed-test-loss = 8.114313 | rel_loss_delta = 0.0007010314200129875 | loss_stagnation_count = 0

Epoch:  25%|██▍       | 74/300 [03:39<11:06,  2.95s/it]2026-10-13 23:52:30.459 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 75, smoothed-train-loss = 8.123156 | smoothed-test-loss = 8.112141 | rel_loss_delta = 7.893370141014069e-05 | loss_stagnation_count = 0

Epoch:  25%|██▌       | 75/300 [03:42<11:05,  2.96s/it]2026-10-13 23:52:33.427 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 76, smoothed-train-loss = 8.118659 | smoothed-test-loss = 8.108154 | rel_loss_delta = -0.0005535425797614252 | loss_stagnation_count = 0

Epoch:  25%|██▌       | 76/300 [03:45<11:03,  2.96s/it]2026-10-13 23:52:36.281 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 77, smoothed-train-loss = 8.120124 | smoothed-test-loss = 8.112702 | rel_loss_delta = 0.00018046214156766899 | loss_stagnation_count = 0

Epoch:  26%|██▌       | 77/300 [03:48<10:53,  2.93s/it]2026-10-13 23:52:39.638 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 78, smoothed-train-loss = 8.113940 | smoothed-test-loss = 8.117679 | rel_loss_delta = -0.0007615722765452343 | loss_stagnation_count = 0

Epoch:  26%|██▌       | 78/300 [03:51<11:18,  3.06s/it]2026-10-13 23:52:42.704 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 79, smoothed-train-loss = 8.114664 | smoothed-test-loss = 8.131907 | rel_loss_delta = 8.921922915233087e-05 | loss_stagnation_count = 0

Epoch:  26%|██▋       | 79/300 [03:55<11:16,  3.06s/it]2026-10-13 23:52:45.926 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 80, smoothed-train-loss = 8.116166 | smoothed-test-loss = 8.121154 | rel_loss_delta = 0.00018502676055023854 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 80/300 [03:58<11:23,  3.11s/it]2026-10-13 23:52:49.056 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 81, smoothed-train-loss = 8.120518 | smoothed-test-loss = 8.105282 | rel_loss_delta = 0.0005362119521429746 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 81/300 [04:01<11:22,  3.11s/it]2026-10-13 23:52:52.548 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 82, smoothed-train-loss = 8.117992 | smoothed-test-loss = 8.119628 | rel_loss_delta = -0.00031101985796210327 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 82/300 [04:04<11:43,  3.23s/it]2026-10-13 23:52:56.210 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 83, smoothed-train-loss = 8.118347 | smoothed-test-loss = 8.100269 | rel_loss_delta = 4.3797371054430184e-05 | loss_stagnation_count = 1

Epoch:  28%|██▊       | 83/300 [04:08<12:08,  3.36s/it]2026-10-13 23:52:59.862 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 84, smoothed-train-loss = 8.115733 | smoothed-test-loss = 8.106043 | rel_loss_delta = -0.0003220609211791827 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 84/300 [04:12<12:24,  3.45s/it]2026-10-13 23:53:03.123 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 85, smoothed-train-loss = 8.113322 | smoothed-test-loss = 8.097387 | rel_loss_delta = -0.0002970639091599795 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 85/300 [04:15<12:09,  3.39s/it]2026-10-13 23:53:06.463 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 86, smoothed-train-loss = 8.115697 | smoothed-test-loss = 8.096481 | rel_loss_delta = 0.0002927982916935752 | loss_stagnation_count = 0

Epoch:  29%|██▊       | 86/300 [04:18<12:02,  3.38s/it]2026-10-13 23:53:09.717 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 87, smoothed-train-loss = 8.116067 | smoothed-test-loss = 8.118394 | rel_loss_delta = 4.548254127298113e-05 | loss_stagnation_count = 1

Epoch:  29%|██▉       | 87/300 [04:22<11:51,  3.34s/it]2026-10-13 23:53:12.958 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 88, smoothed-train-loss = 8.116354 | smoothed-test-loss = 8.103385 | rel_loss_delta = 3.534866116644316e-05 | loss_stagnation_count = 2

Epoch:  29%|██▉       | 88/300 [04:25<11:41,  3.31s/it]2026-10-13 23:53:16.078 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 89, smoothed-train-loss = 8.113611 | smoothed-test-loss = 8.149354 | rel_loss_delta = -0.00033786433927961775 | loss_stagnation_count = 0

Epoch:  30%|██▉       | 89/300 [04:28<11:26,  3.25s/it]2026-10-13 23:53:19.374 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 90, smoothed-train-loss = 8.117027 | smoothed-test-loss = 8.101225 | rel_loss_delta = 0.0004210170739153609 | loss_stagnation_count = 0

Epoch:  30%|███       | 90/300 [04:31<11:25,  3.27s/it]2026-10-13 23:53:22.431 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 91, smoothed-train-loss = 8.115063 | smoothed-test-loss = 8.110974 | rel_loss_delta = -0.0002419927475019079 | loss_stagnation_count = 0

Epoch:  30%|███       | 91/300 [04:34<11:09,  3.20s/it]2026-10-13 23:53:25.360 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 92, smoothed-train-loss = 8.117550 | smoothed-test-loss = 8.105396 | rel_loss_delta = 0.00030652172070632885 | loss_stagnation_count = 0

Epoch:  31%|███       | 92/300 [04:37<10:49,  3.12s/it]2026-10-13 23:53:28.257 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 93, smoothed-train-loss = 8.119965 | smoothed-test-loss = 8.130052 | rel_loss_delta = 0.0002973990998690575 | loss_stagnation_count = 0

Epoch:  31%|███       | 93/300 [04:40<10:32,  3.05s/it]2026-10-13 23:53:31.324 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 94, smoothed-train-loss = 8.116944 | smoothed-test-loss = 8.115763 | rel_loss_delta = -0.0003719343007672127 | loss_stagnation_count = 0

Epoch:  31%|███▏      | 94/300 [04:43<10:29,  3.06s/it]2026-10-13 23:53:34.513 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 95, smoothed-train-loss = 8.114537 | smoothed-test-loss = 8.100183 | rel_loss_delta = -0.0002966164407683407 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 95/300 [04:46<10:34,  3.10s/it]2026-10-13 23:53:37.574 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 96, smoothed-train-loss = 8.114504 | smoothed-test-loss = 8.118552 | rel_loss_delta = -4.024346113843975e-06 | loss_stagnation_count = 1

Epoch:  32%|███▏      | 96/300 [04:49<10:29,  3.09s/it]2026-10-13 23:53:41.014 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 97, smoothed-train-loss = 8.116438 | smoothed-test-loss = 8.116174 | rel_loss_delta = 0.00023832322851200893 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 97/300 [04:53<10:48,  3.19s/it]2026-10-13 23:53:44.822 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 98, smoothed-train-loss = 8.119060 | smoothed-test-loss = 8.117510 | rel_loss_delta = 0.00032299418763326586 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 98/300 [04:57<11:22,  3.38s/it]2026-10-13 23:53:48.301 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 99, smoothed-train-loss = 8.115732 | smoothed-test-loss = 8.121347 | rel_loss_delta = -0.00040990125108880936 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 99/300 [05:00<11:24,  3.41s/it]2026-10-13 23:53:51.950 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 100, smoothed-train-loss = 8.114652 | smoothed-test-loss = 8.121021 | rel_loss_delta = -0.0001330501613096242 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 100/300 [05:04<11:35,  3.48s/it]2026-10-13 23:53:55.191 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 101, smoothed-train-loss = 8.113866 | smoothed-test-loss = 8.096493 | rel_loss_delta = -9.690004016020232e-05 | loss_stagnation_count = 0

Epoch:  34%|███▎      | 101/300 [05:07<11:18,  3.41s/it]2026-10-13 23:53:58.471 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 102, smoothed-train-loss = 8.114181 | smoothed-test-loss = 8.114054 | rel_loss_delta = 3.894186191210587e-05 | loss_stagnation_count = 1

Epoch:  34%|███▍      | 102/300 [05:10<11:07,  3.37s/it]2026-10-13 23:54:01.463 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 103, smoothed-train-loss = 8.110751 | smoothed-test-loss = 8.111504 | rel_loss_delta = -0.00042280053794764187 | loss_stagnation_count = 0

Epoch:  34%|███▍      | 103/300 [05:13<10:41,  3.26s/it]2026-10-13 23:54:04.472 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 104, smoothed-train-loss = 8.111758 | smoothed-test-loss = 8.101228 | rel_loss_delta = 0.00012412364399397343 | loss_stagnation_count = 0

Epoch:  35%|███▍      | 104/300 [05:16<10:23,  3.18s/it]2026-10-13 23:54:07.362 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 105, smoothed-train-loss = 8.109612 | smoothed-test-loss = 8.097109 | rel_loss_delta = -0.0002644830578618624 | loss_stagnation_count = 0

Epoch:  35%|███▌      | 105/300 [05:19<10:03,  3.09s/it]2026-10-13 23:54:10.324 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 106, smoothed-train-loss = 8.112122 | smoothed-test-loss = 8.115110 | rel_loss_delta = 0.0003094468320809378 | loss_stagnation_count = 0

Epoch:  35%|███▌      | 106/300 [05:22<09:52,  3.05s/it]2026-10-13 23:54:13.349 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 107, smoothed-train-loss = 8.113518 | smoothed-test-loss = 8.137000 | rel_loss_delta = 0.00017210090102948852 | loss_stagnation_count = 0

Epoch:  36%|███▌      | 107/300 [05:25<09:47,  3.05s/it]2026-10-13 23:54:16.244 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 108, smoothed-train-loss = 8.111111 | smoothed-test-loss = 8.108280 | rel_loss_delta = -0.00029660440437643865 | loss_stagnation_count = 0

Epoch:  36%|███▌      | 108/300 [05:28<09:36,  3.00s/it]2026-10-13 23:54:19.091 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 109, smoothed-train-loss = 8.115045 | smoothed-test-loss = 8.104139 | rel_loss_delta = 0.0004849862418760955 | loss_stagnation_count = 0

Epoch:  36%|███▋      | 109/300 [05:31<09:24,  2.95s/it]2026-10-13 23:54:22.096 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 110, smoothed-train-loss = 8.115587 | smoothed-test-loss = 8.123051 | rel_loss_delta = 6.674206823345811e-05 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 110/300 [05:34<09:24,  2.97s/it]2026-10-13 23:54:25.082 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 111, smoothed-train-loss = 8.116137 | smoothed-test-loss = 8.111258 | rel_loss_delta = 6.787569432903735e-05 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 111/300 [05:37<09:22,  2.97s/it]2026-10-13 23:54:28.042 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 112, smoothed-train-loss = 8.110535 | smoothed-test-loss = 8.137783 | rel_loss_delta = -0.000690318254200331 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 112/300 [05:40<09:18,  2.97s/it]2026-10-13 23:54:30.912 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 113, smoothed-train-loss = 8.114755 | smoothed-test-loss = 8.125505 | rel_loss_delta = 0.0005203717683068031 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 113/300 [05:43<09:09,  2.94s/it]2026-10-13 23:54:33.888 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 114, smoothed-train-loss = 8.110180 | smoothed-test-loss = 8.096306 | rel_loss_delta = -0.0005638219989344102 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 114/300 [05:46<09:08,  2.95s/it]2026-10-13 23:54:36.903 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 115, smoothed-train-loss = 8.111502 | smoothed-test-loss = 8.096176 | rel_loss_delta = 0.00016296817094326154 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 115/300 [05:49<09:09,  2.97s/it]2026-10-13 23:54:39.941 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 116, smoothed-train-loss = 8.108047 | smoothed-test-loss = 8.101719 | rel_loss_delta = -0.0004258714759720858 | loss_stagnation_count = 0

Epoch:  39%|███▊      | 116/300 [05:52<09:10,  2.99s/it]2026-10-13 23:54:42.771 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 117, smoothed-train-loss = 8.111610 | smoothed-test-loss = 8.100956 | rel_loss_delta = 0.00043936339738382416 | loss_stagnation_count = 0

Epoch:  39%|███▉      | 117/300 [05:55<08:58,  2.94s/it]2026-10-13 23:54:45.827 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 118, smoothed-train-loss = 8.113004 | smoothed-test-loss = 8.104990 | rel_loss_delta = 0.0001719627361616157 | loss_stagnation_count = 0

Epoch:  39%|███▉      | 118/300 [05:58<09:01,  2.98s/it]2026-10-13 23:54:48.766 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 119, smoothed-train-loss = 8.111548 | smoothed-test-loss = 8.110125 | rel_loss_delta = -0.00017952265319111062 | loss_stagnation_count = 0

Epoch:  40%|███▉      | 119/300 [06:01<08:56,  2.97s/it]2026-10-13 23:54:51.827 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 120, smoothed-train-loss = 8.111893 | smoothed-test-loss = 8.105522 | rel_loss_delta = 4.250658032704565e-05 | loss_stagnation_count = 1

Epoch:  40%|████      | 120/300 [06:04<08:58,  2.99s/it]2026-10-13 23:54:54.814 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 121, smoothed-train-loss = 8.108742 | smoothed-test-loss = 8.105400 | rel_loss_delta = -0.00038845745919820456 | loss_stagnation_count = 0

Epoch:  40%|████      | 121/300 [06:07<08:55,  2.99s/it]2026-10-13 23:54:57.779 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 122, smoothed-train-loss = 8.111845 | smoothed-test-loss = 8.106983 | rel_loss_delta = 0.0003827462752750278 | loss_stagnation_count = 0

Epoch:  41%|████      | 122/300 [06:10<08:51,  2.98s/it]2026-10-13 23:55:00.828 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 123, smoothed-train-loss = 8.112069 | smoothed-test-loss = 8.102929 | rel_loss_delta = 2.752724217424336e-05 | loss_stagnation_count = 1

Epoch:  41%|████      | 123/300 [06:13<08:51,  3.00s/it]2026-10-13 23:55:03.840 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 124, smoothed-train-loss = 8.110422 | smoothed-test-loss = 8.111321 | rel_loss_delta = -0.00020296111614002832 | loss_stagnation_count = 0

Epoch:  41%|████▏     | 124/300 [06:16<08:49,  3.01s/it]2026-10-13 23:55:06.682 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 125, smoothed-train-loss = 8.108614 | smoothed-test-loss = 8.108235 | rel_loss_delta = -0.00022291685116711897 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 125/300 [06:18<08:37,  2.96s/it]2026-10-13 23:55:09.650 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 126, smoothed-train-loss = 8.109159 | smoothed-test-loss = 8.112965 | rel_loss_delta = 6.71395793431877e-05 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 126/300 [06:21<08:35,  2.96s/it]2026-10-13 23:55:12.662 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 127, smoothed-train-loss = 8.110096 | smoothed-test-loss = 8.108842 | rel_loss_delta = 0.00011554249864347795 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 127/300 [06:24<08:34,  2.98s/it]2026-10-13 23:55:15.641 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 128, smoothed-train-loss = 8.109549 | smoothed-test-loss = 8.108950 | rel_loss_delta = -6.736993155590224e-05 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 128/300 [06:27<08:31,  2.98s/it]2026-10-13 23:55:18.552 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 129, smoothed-train-loss = 8.112171 | smoothed-test-loss = 8.119713 | rel_loss_delta = 0.0003233362894077821 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 129/300 [06:30<08:25,  2.96s/it]2026-10-13 23:55:21.522 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 130, smoothed-train-loss = 8.109449 | smoothed-test-loss = 8.096066 | rel_loss_delta = -0.00033560123022303734 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 130/300 [06:33<08:23,  2.96s/it]2026-10-13 23:55:24.462 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 131, smoothed-train-loss = 8.107289 | smoothed-test-loss = 8.095161 | rel_loss_delta = -0.0002662810858793706 | loss_stagnation_count = 0

Epoch:  44%|████▎     | 131/300 [06:36<08:19,  2.95s/it]2026-10-13 23:55:27.490 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 132, smoothed-train-loss = 8.109815 | smoothed-test-loss = 8.100262 | rel_loss_delta = 0.00031154263026290983 | loss_stagnation_count = 0

Epoch:  44%|████▍     | 132/300 [06:39<08:20,  2.98s/it]2026-10-13 23:55:30.370 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 133, smoothed-train-loss = 8.107818 | smoothed-test-loss = 8.103204 | rel_loss_delta = -0.00024627155875855516 | loss_stagnation_count = 0

Epoch:  44%|████▍     | 133/300 [06:42<08:12,  2.95s/it]2026-10-13 23:55:33.322 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 134, smoothed-train-loss = 8.108201 | smoothed-test-loss = 8.104083 | rel_loss_delta = 4.726630102949858e-05 | loss_stagnation_count = 1

Epoch:  45%|████▍     | 134/300 [06:45<08:09,  2.95s/it]2026-10-13 23:55:36.225 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 135, smoothed-train-loss = 8.106744 | smoothed-test-loss = 8.093278 | rel_loss_delta = -0.00017969236954919012 | loss_stagnation_count = 0

Epoch:  45%|████▌     | 135/300 [06:48<08:04,  2.94s/it]2026-10-13 23:55:39.191 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 136, smoothed-train-loss = 8.107767 | smoothed-test-loss = 8.101507 | rel_loss_delta = 0.00012618967012780909 | loss_stagnation_count = 0

Epoch:  45%|████▌     | 136/300 [06:51<08:02,  2.94s/it]2026-10-13 23:55:42.038 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 137, smoothed-train-loss = 8.108086 | smoothed-test-loss = 8.108356 | rel_loss_delta = 3.9295683929200595e-05 | loss_stagnation_count = 1

Epoch:  46%|████▌     | 137/300 [06:54<07:55,  2.92s/it]2026-10-13 23:55:45.049 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 138, smoothed-train-loss = 8.109979 | smoothed-test-loss = 8.101772 | rel_loss_delta = 0.00023343839159596858 | loss_stagnation_count = 0

Epoch:  46%|████▌     | 138/300 [06:57<07:56,  2.94s/it]2026-10-13 23:55:47.960 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 139, smoothed-train-loss = 8.109029 | smoothed-test-loss = 8.137421 | rel_loss_delta = -0.00011708518950564606 | loss_stagnation_count = 0

Epoch:  46%|████▋     | 139/300 [07:00<07:52,  2.93s/it]2026-10-13 23:55:50.945 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 140, smoothed-train-loss = 8.109393 | smoothed-test-loss = 8.118672 | rel_loss_delta = 4.484042635831144e-05 | loss_stagnation_count = 1

Epoch:  47%|████▋     | 140/300 [07:03<07:51,  2.95s/it]2026-10-13 23:55:53.909 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 141, smoothed-train-loss = 8.107944 | smoothed-test-loss = 8.100292 | rel_loss_delta = -0.0001785873639136051 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 141/300 [07:06<07:49,  2.95s/it]2026-10-13 23:55:56.933 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 142, smoothed-train-loss = 8.111725 | smoothed-test-loss = 8.096119 | rel_loss_delta = 0.0004662793573087909 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 142/300 [07:09<07:50,  2.97s/it]2026-10-13 23:55:59.931 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 143, smoothed-train-loss = 8.109716 | smoothed-test-loss = 8.094768 | rel_loss_delta = -0.0002476726137094531 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 143/300 [07:12<07:48,  2.98s/it]2026-10-13 23:56:03.041 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 144, smoothed-train-loss = 8.106776 | smoothed-test-loss = 8.104530 | rel_loss_delta = -0.0003624705456105974 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 144/300 [07:15<07:51,  3.02s/it]2026-10-13 23:56:05.839 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 145, smoothed-train-loss = 8.107687 | smoothed-test-loss = 8.095526 | rel_loss_delta = 0.00011234386201532214 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 145/300 [07:18<07:37,  2.95s/it]2026-10-13 23:56:08.843 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 146, smoothed-train-loss = 8.108816 | smoothed-test-loss = 8.118333 | rel_loss_delta = 0.00013921879296662604 | loss_stagnation_count = 0

Epoch:  49%|████▊     | 146/300 [07:21<07:37,  2.97s/it]2026-10-13 23:56:11.841 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 147, smoothed-train-loss = 8.107337 | smoothed-test-loss = 8.100034 | rel_loss_delta = -0.00018235156931655842 | loss_stagnation_count = 0

Epoch:  49%|████▉     | 147/300 [07:24<07:35,  2.98s/it]2026-10-13 23:56:14.780 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 148, smoothed-train-loss = 8.108291 | smoothed-test-loss = 8.103368 | rel_loss_delta = 0.00011765831256714669 | loss_stagnation_count = 0

Epoch:  49%|████▉     | 148/300 [07:27<07:30,  2.97s/it]2026-10-13 23:56:17.653 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 149, smoothed-train-loss = 8.105956 | smoothed-test-loss = 8.097786 | rel_loss_delta = -0.0002879558448327846 | loss_stagnation_count = 0

Epoch:  50%|████▉     | 149/300 [07:29<07:23,  2.94s/it]2026-10-13 23:56:20.567 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 150, smoothed-train-loss = 8.104852 | smoothed-test-loss = 8.107372 | rel_loss_delta = -0.00013617447592961966 | loss_stagnation_count = 0

Epoch:  50%|█████     | 150/300 [07:32<07:19,  2.93s/it]2026-10-13 23:56:23.561 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 151, smoothed-train-loss = 8.107659 | smoothed-test-loss = 8.092951 | rel_loss_delta = 0.0003462470575581945 | loss_stagnation_count = 0

Epoch:  50%|█████     | 151/300 [07:35<07:19,  2.95s/it]2026-10-13 23:56:26.657 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 152, smoothed-train-loss = 8.108418 | smoothed-test-loss = 8.085338 | rel_loss_delta = 9.370117293929331e-05 | loss_stagnation_count = 0

Epoch:  51%|█████     | 152/300 [07:38<07:23,  2.99s/it]2026-10-13 23:56:29.754 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 153, smoothed-train-loss = 8.107658 | smoothed-test-loss = 8.092472 | rel_loss_delta = -9.378854208298313e-05 | loss_stagnation_count = 0

Epoch:  51%|█████     | 153/300 [07:42<07:24,  3.02s/it]2026-10-13 23:56:32.817 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 154, smoothed-train-loss = 8.107729 | smoothed-test-loss = 8.103958 | rel_loss_delta = 8.77607478991423e-06 | loss_stagnation_count = 1

Epoch:  51%|█████▏    | 154/300 [07:45<07:23,  3.04s/it]2026-10-13 23:56:35.741 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 155, smoothed-train-loss = 8.103665 | smoothed-test-loss = 8.097582 | rel_loss_delta = -0.0005012507653892813 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 155/300 [07:48<07:15,  3.00s/it]2026-10-13 23:56:38.776 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 156, smoothed-train-loss = 8.106348 | smoothed-test-loss = 8.110811 | rel_loss_delta = 0.0003310602882747913 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 156/300 [07:51<07:13,  3.01s/it]2026-10-13 23:56:41.695 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 157, smoothed-train-loss = 8.109285 | smoothed-test-loss = 8.097924 | rel_loss_delta = 0.000362326749086957 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 157/300 [07:54<07:06,  2.98s/it]2026-10-13 23:56:44.695 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 158, smoothed-train-loss = 8.109087 | smoothed-test-loss = 8.103302 | rel_loss_delta = -2.4442594099141272e-05 | loss_stagnation_count = 1

Epoch:  53%|█████▎    | 158/300 [07:57<07:04,  2.99s/it]2026-10-13 23:56:47.742 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 159, smoothed-train-loss = 8.108536 | smoothed-test-loss = 8.103210 | rel_loss_delta = -6.790320361516148e-05 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 159/300 [08:00<07:03,  3.01s/it]2026-10-13 23:56:50.707 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 160, smoothed-train-loss = 8.106786 | smoothed-test-loss = 8.095502 | rel_loss_delta = -0.00021581573314975063 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 160/300 [08:03<06:59,  2.99s/it]2026-10-13 23:56:53.544 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 161, smoothed-train-loss = 8.110087 | smoothed-test-loss = 8.094824 | rel_loss_delta = 0.0004071553788012114 | loss_stagnation_count = 0

Epoch:  54%|█████▎    | 161/300 [08:05<06:49,  2.95s/it]2026-10-13 23:56:56.440 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 162, smoothed-train-loss = 8.107401 | smoothed-test-loss = 8.097809 | rel_loss_delta = -0.0003311768902823817 | loss_stagnation_count = 0

Epoch:  54%|█████▍    | 162/300 [08:08<06:44,  2.93s/it]2026-10-13 23:56:59.387 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 163, smoothed-train-loss = 8.104638 | smoothed-test-loss = 8.101614 | rel_loss_delta = -0.0003407706494153877 | loss_stagnation_count = 0

Epoch:  54%|█████▍    | 163/300 [08:11<06:42,  2.94s/it]2026-10-13 23:57:02.367 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 164, smoothed-train-loss = 8.103719 | smoothed-test-loss = 8.096907 | rel_loss_delta = -0.00011338006523468942 | loss_stagnation_count = 0

Epoch:  55%|█████▍    | 164/300 [08:14<06:41,  2.95s/it]2026-10-13 23:57:05.310 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 165, smoothed-train-loss = 8.106462 | smoothed-test-loss = 8.113898 | rel_loss_delta = 0.0003383905585524581 | loss_stagnation_count = 0

Epoch:  55%|█████▌    | 165/300 [08:17<06:37,  2.95s/it]2026-10-13 23:57:08.445 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 166, smoothed-train-loss = 8.105494 | smoothed-test-loss = 8.096166 | rel_loss_delta = -0.00011930540297579735 | loss_stagnation_count = 0

Epoch:  55%|█████▌    | 166/300 [08:20<06:42,  3.00s/it]2026-10-13 23:57:11.465 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 167, smoothed-train-loss = 8.109245 | smoothed-test-loss = 8.094658 | rel_loss_delta = 0.00046277010905830323 | loss_stagnation_count = 0

Epoch:  56%|█████▌    | 167/300 [08:23<06:40,  3.01s/it]2026-10-13 23:57:14.378 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 168, smoothed-train-loss = 8.107441 | smoothed-test-loss = 8.099467 | rel_loss_delta = -0.00022248107765309108 | loss_stagnation_count = 0

Epoch:  56%|█████▌    | 168/300 [08:26<06:33,  2.98s/it]2026-10-13 23:57:17.274 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 169, smoothed-train-loss = 8.104558 | smoothed-test-loss = 8.107161 | rel_loss_delta = -0.00035564574166244384 | loss_stagnation_count = 0

Epoch:  56%|█████▋    | 169/300 [08:29<06:27,  2.95s/it]2026-10-13 23:57:20.314 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 170, smoothed-train-loss = 8.104716 | smoothed-test-loss = 8.094779 | rel_loss_delta = 1.9513175437871625e-05 | loss_stagnation_count = 1

Epoch:  57%|█████▋    | 170/300 [08:32<06:27,  2.98s/it]2026-10-13 23:57:23.284 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 171, smoothed-train-loss = 8.107123 | smoothed-test-loss = 8.118824 | rel_loss_delta = 0.0002969700617349708 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 171/300 [08:35<06:24,  2.98s/it]2026-10-13 23:57:26.181 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 172, smoothed-train-loss = 8.104262 | smoothed-test-loss = 8.103977 | rel_loss_delta = -0.0003529347730909618 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 172/300 [08:38<06:18,  2.95s/it]2026-10-13 23:57:29.017 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 173, smoothed-train-loss = 8.106550 | smoothed-test-loss = 8.099820 | rel_loss_delta = 0.00028231408900343004 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 173/300 [08:41<06:10,  2.92s/it]2026-10-13 23:57:31.943 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 174, smoothed-train-loss = 8.107524 | smoothed-test-loss = 8.109411 | rel_loss_delta = 0.00012022336423907673 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 174/300 [08:44<06:07,  2.92s/it]2026-10-13 23:57:34.921 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 175, smoothed-train-loss = 8.105177 | smoothed-test-loss = 8.096058 | rel_loss_delta = -0.00028955042233839556 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 175/300 [08:47<06:07,  2.94s/it]2026-10-13 23:57:37.850 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 176, smoothed-train-loss = 8.103369 | smoothed-test-loss = 8.098695 | rel_loss_delta = -0.00022300588044968231 | loss_stagnation_count = 0

Epoch:  59%|█████▊    | 176/300 [08:50<06:03,  2.94s/it]2026-10-13 23:57:40.727 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 177, smoothed-train-loss = 8.105910 | smoothed-test-loss = 8.094715 | rel_loss_delta = 0.00031353564877674965 | loss_stagnation_count = 0

Epoch:  59%|█████▉    | 177/300 [08:53<05:58,  2.92s/it]2026-10-13 23:57:43.947 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 178, smoothed-train-loss = 8.106466 | smoothed-test-loss = 8.108655 | rel_loss_delta = 6.862429676946566e-05 | loss_stagnation_count = 0

Epoch:  59%|█████▉    | 178/300 [08:56<06:07,  3.01s/it]2026-10-13 23:57:46.961 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 179, smoothed-train-loss = 8.106774 | smoothed-test-loss = 8.102553 | rel_loss_delta = 3.804146856847204e-05 | loss_stagnation_count = 1

Epoch:  60%|█████▉    | 179/300 [08:59<06:04,  3.01s/it]2026-10-13 23:57:49.937 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 180, smoothed-train-loss = 8.104552 | smoothed-test-loss = 8.097074 | rel_loss_delta = -0.0002741203529406962 | loss_stagnation_count = 0

Epoch:  60%|██████    | 180/300 [09:02<05:59,  3.00s/it]2026-10-13 23:57:53.139 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 181, smoothed-train-loss = 8.104226 | smoothed-test-loss = 8.094468 | rel_loss_delta = -4.0274773912564893e-05 | loss_stagnation_count = 1

Epoch:  60%|██████    | 181/300 [09:05<06:04,  3.06s/it]2026-10-13 23:57:56.742 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 182, smoothed-train-loss = 8.104648 | smoothed-test-loss = 8.110809 | rel_loss_delta = 5.204611988685535e-05 | loss_stagnation_count = 0

Epoch:  61%|██████    | 182/300 [09:09<06:20,  3.22s/it]2026-10-13 23:58:00.219 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 183, smoothed-train-loss = 8.104389 | smoothed-test-loss = 8.099264 | rel_loss_delta = -3.1879950973446194e-05 | loss_stagnation_count = 1

Epoch:  61%|██████    | 183/300 [09:12<06:26,  3.30s/it]2026-10-13 23:58:03.289 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 184, smoothed-train-loss = 8.104527 | smoothed-test-loss = 8.104391 | rel_loss_delta = 1.699009320300941e-05 | loss_stagnation_count = 2

Epoch:  61%|██████▏   | 184/300 [09:15<06:14,  3.23s/it]2026-10-13 23:58:06.251 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 185, smoothed-train-loss = 8.103899 | smoothed-test-loss = 8.101563 | rel_loss_delta = -7.753600942148671e-05 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 185/300 [09:18<06:02,  3.15s/it]2026-10-13 23:58:09.187 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 186, smoothed-train-loss = 8.104660 | smoothed-test-loss = 8.100373 | rel_loss_delta = 9.389643739553366e-05 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 186/300 [09:21<05:51,  3.09s/it]2026-10-13 23:58:12.238 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 187, smoothed-train-loss = 8.103060 | smoothed-test-loss = 8.096055 | rel_loss_delta = -0.0001973426240092636 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 187/300 [09:24<05:47,  3.08s/it]2026-10-13 23:58:15.306 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 188, smoothed-train-loss = 8.103031 | smoothed-test-loss = 8.104850 | rel_loss_delta = -3.6381541394144815e-06 | loss_stagnation_count = 1

Epoch:  63%|██████▎   | 188/300 [09:27<05:44,  3.07s/it]2026-10-13 23:58:18.317 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 189, smoothed-train-loss = 8.102984 | smoothed-test-loss = 8.118474 | rel_loss_delta = -5.708561105351263e-06 | loss_stagnation_count = 2

Epoch:  63%|██████▎   | 189/300 [09:30<05:39,  3.05s/it]2026-10-13 23:58:21.302 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 190, smoothed-train-loss = 8.101183 | smoothed-test-loss = 8.101803 | rel_loss_delta = -0.0002223496104187435 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 190/300 [09:33<05:33,  3.03s/it]2026-10-13 23:58:24.820 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 191, smoothed-train-loss = 8.103836 | smoothed-test-loss = 8.102464 | rel_loss_delta = 0.00032757337397892324 | loss_stagnation_count = 0

Epoch:  64%|██████▎   | 191/300 [09:37<05:46,  3.18s/it]2026-10-13 23:58:28.678 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 192, smoothed-train-loss = 8.104421 | smoothed-test-loss = 8.098737 | rel_loss_delta = 7.211516118114093e-05 | loss_stagnation_count = 0

Epoch:  64%|██████▍   | 192/300 [09:40<06:05,  3.38s/it]2026-10-13 23:58:32.138 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 193, smoothed-train-loss = 8.101487 | smoothed-test-loss = 8.117127 | rel_loss_delta = -0.0003619896655636531 | loss_stagnation_count = 0

Epoch:  64%|██████▍   | 193/300 [09:44<06:04,  3.41s/it]2026-10-13 23:58:35.231 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 194, smoothed-train-loss = 8.104941 | smoothed-test-loss = 8.104132 | rel_loss_delta = 0.00042637803206483396 | loss_stagnation_count = 0

Epoch:  65%|██████▍   | 194/300 [09:47<05:51,  3.31s/it]2026-10-13 23:58:38.503 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 195, smoothed-train-loss = 8.101688 | smoothed-test-loss = 8.107356 | rel_loss_delta = -0.00040144115194238315 | loss_stagnation_count = 0

Epoch:  65%|██████▌   | 195/300 [09:50<05:46,  3.30s/it]2026-10-13 23:58:41.799 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 196, smoothed-train-loss = 8.100361 | smoothed-test-loss = 8.090306 | rel_loss_delta = -0.00016374064126598427 | loss_stagnation_count = 0

Epoch:  65%|██████▌   | 196/300 [09:54<05:43,  3.30s/it]2026-10-13 23:58:44.988 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 197, smoothed-train-loss = 8.101607 | smoothed-test-loss = 8.088567 | rel_loss_delta = 0.00015376362744318816 | loss_stagnation_count = 0

Epoch:  66%|██████▌   | 197/300 [09:57<05:36,  3.27s/it]2026-10-13 23:58:47.964 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 198, smoothed-train-loss = 8.102277 | smoothed-test-loss = 8.103438 | rel_loss_delta = 8.273906446629868e-05 | loss_stagnation_count = 0

Epoch:  66%|██████▌   | 198/300 [10:00<05:24,  3.18s/it]2026-10-13 23:58:51.024 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 199, smoothed-train-loss = 8.105849 | smoothed-test-loss = 8.106731 | rel_loss_delta = 0.0004408525657528473 | loss_stagnation_count = 0

Epoch:  66%|██████▋   | 199/300 [10:03<05:17,  3.14s/it]2026-10-13 23:58:53.973 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 200, smoothed-train-loss = 8.101238 | smoothed-test-loss = 8.095883 | rel_loss_delta = -0.0005687867837674929 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 200/300 [10:06<05:08,  3.09s/it]2026-10-13 23:58:56.877 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 201, smoothed-train-loss = 8.103966 | smoothed-test-loss = 8.098263 | rel_loss_delta = 0.000336738959815359 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 201/300 [10:09<05:00,  3.03s/it]2026-10-13 23:59:00.032 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 202, smoothed-train-loss = 8.101897 | smoothed-test-loss = 8.089060 | rel_loss_delta = -0.00025541651748763267 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 202/300 [10:12<05:00,  3.07s/it]2026-10-13 23:59:03.324 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 203, smoothed-train-loss = 8.101770 | smoothed-test-loss = 8.100265 | rel_loss_delta = -1.5673909025533954e-05 | loss_stagnation_count = 1

Epoch:  68%|██████▊   | 203/300 [10:15<05:04,  3.14s/it]2026-10-13 23:59:06.914 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 204, smoothed-train-loss = 8.102568 | smoothed-test-loss = 8.095934 | rel_loss_delta = 9.854390602376092e-05 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 204/300 [10:19<05:14,  3.27s/it]2026-10-13 23:59:10.109 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 205, smoothed-train-loss = 8.103592 | smoothed-test-loss = 8.102415 | rel_loss_delta = 0.00012636875735694015 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 205/300 [10:22<05:08,  3.25s/it]2026-10-13 23:59:13.741 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 206, smoothed-train-loss = 8.103981 | smoothed-test-loss = 8.098028 | rel_loss_delta = 4.797342613352608e-05 | loss_stagnation_count = 1

Epoch:  69%|██████▊   | 206/300 [10:26<05:16,  3.36s/it]2026-10-13 23:59:17.470 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 207, smoothed-train-loss = 8.101578 | smoothed-test-loss = 8.094443 | rel_loss_delta = -0.0002964479594606051 | loss_stagnation_count = 0

Epoch:  69%|██████▉   | 207/300 [10:29<05:23,  3.47s/it]2026-10-13 23:59:21.169 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 208, smoothed-train-loss = 8.101333 | smoothed-test-loss = 8.121797 | rel_loss_delta = -3.0234745159781947e-05 | loss_stagnation_count = 1

Epoch:  69%|██████▉   | 208/300 [10:33<05:25,  3.54s/it]2026-10-13 23:59:24.629 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 209, smoothed-train-loss = 8.101087 | smoothed-test-loss = 8.105509 | rel_loss_delta = -3.03527128655308e-05 | loss_stagnation_count = 2

Epoch:  70%|██████▉   | 209/300 [10:36<05:20,  3.52s/it]2026-10-13 23:59:27.834 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 210, smoothed-train-loss = 8.106231 | smoothed-test-loss = 8.107580 | rel_loss_delta = 0.0006348836196620416 | loss_stagnation_count = 0

Epoch:  70%|███████   | 210/300 [10:40<05:08,  3.42s/it]2026-10-13 23:59:31.240 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 211, smoothed-train-loss = 8.102484 | smoothed-test-loss = 8.089661 | rel_loss_delta = -0.0004621564020175447 | loss_stagnation_count = 0

Epoch:  70%|███████   | 211/300 [10:43<05:04,  3.42s/it]2026-10-13 23:59:34.707 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 212, smoothed-train-loss = 8.102417 | smoothed-test-loss = 8.091171 | rel_loss_delta = -8.333093099393415e-06 | loss_stagnation_count = 1

Epoch:  71%|███████   | 212/300 [10:47<05:02,  3.43s/it]2026-10-13 23:59:38.273 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 213, smoothed-train-loss = 8.100866 | smoothed-test-loss = 8.093616 | rel_loss_delta = -0.00019136271542174346 | loss_stagnation_count = 0

Epoch:  71%|███████   | 213/300 [10:50<05:02,  3.47s/it]2026-10-13 23:59:41.790 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 214, smoothed-train-loss = 8.103163 | smoothed-test-loss = 8.096582 | rel_loss_delta = 0.00028352163066116317 | loss_stagnation_count = 0

Epoch:  71%|███████▏  | 214/300 [10:54<04:59,  3.49s/it]2026-10-13 23:59:45.250 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 215, smoothed-train-loss = 8.101542 | smoothed-test-loss = 8.092994 | rel_loss_delta = -0.00019999750992371406 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 215/300 [10:57<04:55,  3.48s/it]2026-10-13 23:59:48.647 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 216, smoothed-train-loss = 8.102235 | smoothed-test-loss = 8.087879 | rel_loss_delta = 8.544427469667097e-05 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 216/300 [11:00<04:50,  3.45s/it]2026-10-13 23:59:51.751 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 217, smoothed-train-loss = 8.102859 | smoothed-test-loss = 8.102778 | rel_loss_delta = 7.705483946937472e-05 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 217/300 [11:04<04:37,  3.35s/it]2026-10-13 23:59:55.136 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 218, smoothed-train-loss = 8.102139 | smoothed-test-loss = 8.098592 | rel_loss_delta = -8.886844887392242e-05 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 218/300 [11:07<04:35,  3.36s/it]2026-10-13 23:59:58.388 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 219, smoothed-train-loss = 8.101531 | smoothed-test-loss = 8.107818 | rel_loss_delta = -7.497954417388032e-05 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 219/300 [11:10<04:29,  3.33s/it]2026-10-14 00:00:01.782 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 220, smoothed-train-loss = 8.102818 | smoothed-test-loss = 8.098605 | rel_loss_delta = 0.00015877308132558585 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 220/300 [11:14<04:27,  3.35s/it]2026-10-14 00:00:05.053 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 221, smoothed-train-loss = 8.101551 | smoothed-test-loss = 8.093551 | rel_loss_delta = -0.0001563813347738541 | loss_stagnation_count = 0

Epoch:  74%|███████▎  | 221/300 [11:17<04:22,  3.32s/it]2026-10-14 00:00:08.304 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 222, smoothed-train-loss = 8.101545 | smoothed-test-loss = 8.096120 | rel_loss_delta = -6.221970661256976e-07 | loss_stagnation_count = 1

Epoch:  74%|███████▍  | 222/300 [11:20<04:17,  3.30s/it]2026-10-14 00:00:11.635 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 223, smoothed-train-loss = 8.101779 | smoothed-test-loss = 8.092660 | rel_loss_delta = 2.880063728789458e-05 | loss_stagnation_count = 2

Epoch:  74%|███████▍  | 223/300 [11:23<04:14,  3.31s/it]2026-10-14 00:00:14.576 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 224, smoothed-train-loss = 8.101014 | smoothed-test-loss = 8.096827 | rel_loss_delta = -9.439548765777211e-05 | loss_stagnation_count = 0

Epoch:  75%|███████▍  | 224/300 [11:26<04:03,  3.20s/it]2026-10-14 00:00:17.572 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 225, smoothed-train-loss = 8.101574 | smoothed-test-loss = 8.092806 | rel_loss_delta = 6.909113754838369e-05 | loss_stagnation_count = 0

Epoch:  75%|███████▌  | 225/300 [11:29<03:55,  3.14s/it]2026-10-14 00:00:20.671 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 226, smoothed-train-loss = 8.100909 | smoothed-test-loss = 8.102998 | rel_loss_delta = -8.20704474779156e-05 | loss_stagnation_count = 0

Epoch:  75%|███████▌  | 226/300 [11:32<03:51,  3.13s/it]2026-10-14 00:00:23.760 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 227, smoothed-train-loss = 8.102971 | smoothed-test-loss = 8.104274 | rel_loss_delta = 0.00025460920401478487 | loss_stagnation_count = 0

Epoch:  76%|███████▌  | 227/300 [11:36<03:47,  3.12s/it]2026-10-14 00:00:26.926 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 228, smoothed-train-loss = 8.099650 | smoothed-test-loss = 8.115590 | rel_loss_delta = -0.0004099393197324023 | loss_stagnation_count = 0

Epoch:  76%|███████▌  | 228/300 [11:39<03:45,  3.13s/it]2026-10-14 00:00:30.130 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 229, smoothed-train-loss = 8.103816 | smoothed-test-loss = 8.118901 | rel_loss_delta = 0.0005143394687764278 | loss_stagnation_count = 0

Epoch:  76%|███████▋  | 229/300 [11:42<03:43,  3.15s/it]2026-10-14 00:00:33.675 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 230, smoothed-train-loss = 8.103070 | smoothed-test-loss = 8.095169 | rel_loss_delta = -9.197310567473617e-05 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 230/300 [11:45<03:48,  3.27s/it]2026-10-14 00:00:36.718 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 231, smoothed-train-loss = 8.102501 | smoothed-test-loss = 8.102822 | rel_loss_delta = -7.028917618100505e-05 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 231/300 [11:49<03:40,  3.20s/it]2026-10-14 00:00:39.729 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 232, smoothed-train-loss = 8.099351 | smoothed-test-loss = 8.096290 | rel_loss_delta = -0.0003887518032605054 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 232/300 [11:52<03:33,  3.14s/it]2026-10-14 00:00:42.943 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 233, smoothed-train-loss = 8.103484 | smoothed-test-loss = 8.096482 | rel_loss_delta = 0.0005103136702373015 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 233/300 [11:55<03:32,  3.17s/it]2026-10-14 00:00:46.550 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 234, smoothed-train-loss = 8.102402 | smoothed-test-loss = 8.097899 | rel_loss_delta = -0.00013358197080938017 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 234/300 [11:58<03:37,  3.30s/it]2026-10-14 00:00:49.672 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 235, smoothed-train-loss = 8.100638 | smoothed-test-loss = 8.098704 | rel_loss_delta = -0.00021772389416542965 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 235/300 [12:01<03:30,  3.25s/it]2026-10-14 00:00:53.270 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 236, smoothed-train-loss = 8.101518 | smoothed-test-loss = 8.096676 | rel_loss_delta = 0.00010866495040500943 | loss_stagnation_count = 0

Epoch:  79%|███████▊  | 236/300 [12:05<03:34,  3.35s/it]2026-10-14 00:00:56.592 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 237, smoothed-train-loss = 8.101852 | smoothed-test-loss = 8.099861 | rel_loss_delta = 4.1262786060006696e-05 | loss_stagnation_count = 1

Epoch:  79%|███████▉  | 237/300 [12:08<03:30,  3.34s/it]2026-10-14 00:00:59.946 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 238, smoothed-train-loss = 8.100971 | smoothed-test-loss = 8.091427 | rel_loss_delta = -0.0001086945911704789 | loss_stagnation_count = 0

Epoch:  79%|███████▉  | 238/300 [12:12<03:27,  3.35s/it]2026-10-14 00:01:03.243 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 239, smoothed-train-loss = 8.100623 | smoothed-test-loss = 8.112507 | rel_loss_delta = -4.301768097180249e-05 | loss_stagnation_count = 1

Epoch:  80%|███████▉  | 239/300 [12:15<03:23,  3.33s/it]2026-10-14 00:01:06.670 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 240, smoothed-train-loss = 8.100171 | smoothed-test-loss = 8.106752 | rel_loss_delta = -5.5831287944872056e-05 | loss_stagnation_count = 0

Epoch:  80%|████████  | 240/300 [12:18<03:21,  3.36s/it]2026-10-14 00:01:10.224 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 241, smoothed-train-loss = 8.101074 | smoothed-test-loss = 8.112208 | rel_loss_delta = 0.00011150571546811657 | loss_stagnation_count = 0

Epoch:  80%|████████  | 241/300 [12:22<03:21,  3.42s/it]2026-10-14 00:01:13.269 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 242, smoothed-train-loss = 8.102951 | smoothed-test-loss = 8.095104 | rel_loss_delta = 0.0002316511598639128 | loss_stagnation_count = 0

Epoch:  81%|████████  | 242/300 [12:25<03:11,  3.31s/it]2026-10-14 00:01:16.362 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 243, smoothed-train-loss = 8.100398 | smoothed-test-loss = 8.092846 | rel_loss_delta = -0.000315032584557122 | loss_stagnation_count = 0

Epoch:  81%|████████  | 243/300 [12:28<03:04,  3.24s/it]2026-10-14 00:01:20.253 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 244, smoothed-train-loss = 8.100024 | smoothed-test-loss = 8.091098 | rel_loss_delta = -4.616274866603607e-05 | loss_stagnation_count = 1

Epoch:  81%|████████▏ | 244/300 [12:32<03:12,  3.44s/it]2026-10-14 00:01:23.378 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 245, smoothed-train-loss = 8.100363 | smoothed-test-loss = 8.090661 | rel_loss_delta = 4.182424867426307e-05 | loss_stagnation_count = 2

Epoch:  82%|████████▏ | 245/300 [12:35<03:03,  3.34s/it]2026-10-14 00:01:26.547 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 246, smoothed-train-loss = 8.100464 | smoothed-test-loss = 8.087824 | rel_loss_delta = 1.2509473087973456e-05 | loss_stagnation_count = 3

Epoch:  82%|████████▏ | 246/300 [12:38<02:57,  3.29s/it]2026-10-14 00:01:29.926 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 247, smoothed-train-loss = 8.101744 | smoothed-test-loss = 8.098746 | rel_loss_delta = 0.00015803477592631784 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 247/300 [12:42<02:55,  3.32s/it]2026-10-14 00:01:33.002 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 248, smoothed-train-loss = 8.101439 | smoothed-test-loss = 8.088746 | rel_loss_delta = -3.76485424915776e-05 | loss_stagnation_count = 1

Epoch:  83%|████████▎ | 248/300 [12:45<02:48,  3.24s/it]2026-10-14 00:01:35.988 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 249, smoothed-train-loss = 8.101735 | smoothed-test-loss = 8.103774 | rel_loss_delta = 3.65630844130448e-05 | loss_stagnation_count = 2

Epoch:  83%|████████▎ | 249/300 [12:48<02:41,  3.17s/it]2026-10-14 00:01:39.314 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 250, smoothed-train-loss = 8.100648 | smoothed-test-loss = 8.093763 | rel_loss_delta = -0.00013418144424531172 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 250/300 [12:51<02:40,  3.21s/it]2026-10-14 00:01:42.554 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 251, smoothed-train-loss = 8.100569 | smoothed-test-loss = 8.098663 | rel_loss_delta = -9.76535075005113e-06 | loss_stagnation_count = 1

Epoch:  84%|████████▎ | 251/300 [12:54<02:37,  3.22s/it]2026-10-14 00:01:45.916 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 252, smoothed-train-loss = 8.101982 | smoothed-test-loss = 8.094758 | rel_loss_delta = 0.0001743842440974114 | loss_stagnation_count = 0

Epoch:  84%|████████▍ | 252/300 [12:58<02:36,  3.26s/it]2026-10-14 00:01:48.832 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 253, smoothed-train-loss = 8.100022 | smoothed-test-loss = 8.088275 | rel_loss_delta = -0.0002419273400617495 | loss_stagnation_count = 0

Epoch:  84%|████████▍ | 253/300 [13:01<02:28,  3.16s/it]2026-10-14 00:01:52.074 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 254, smoothed-train-loss = 8.099115 | smoothed-test-loss = 8.086160 | rel_loss_delta = -0.00011190283799454017 | loss_stagnation_count = 0

Epoch:  85%|████████▍ | 254/300 [13:04<02:26,  3.18s/it]2026-10-14 00:01:55.388 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 255, smoothed-train-loss = 8.099112 | smoothed-test-loss = 8.083513 | rel_loss_delta = -3.877675728391896e-07 | loss_stagnation_count = 1

Epoch:  85%|████████▌ | 255/300 [13:07<02:25,  3.22s/it]2026-10-14 00:01:58.720 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 256, smoothed-train-loss = 8.101284 | smoothed-test-loss = 8.096955 | rel_loss_delta = 0.00026812129277860834 | loss_stagnation_count = 0

Epoch:  85%|████████▌ | 256/300 [13:11<02:23,  3.26s/it]2026-10-14 00:02:01.796 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 257, smoothed-train-loss = 8.104293 | smoothed-test-loss = 8.101042 | rel_loss_delta = 0.00037140036507928427 | loss_stagnation_count = 0

Epoch:  86%|████████▌ | 257/300 [13:14<02:17,  3.20s/it]2026-10-14 00:02:05.015 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 258, smoothed-train-loss = 8.101269 | smoothed-test-loss = 8.086886 | rel_loss_delta = -0.0003730451899892285 | loss_stagnation_count = 0

Epoch:  86%|████████▌ | 258/300 [13:17<02:14,  3.21s/it]2026-10-14 00:02:08.154 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 259, smoothed-train-loss = 8.101000 | smoothed-test-loss = 8.093460 | rel_loss_delta = -3.328698803407765e-05 | loss_stagnation_count = 1

Epoch:  86%|████████▋ | 259/300 [13:20<02:10,  3.19s/it]2026-10-14 00:02:11.471 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 260, smoothed-train-loss = 8.101672 | smoothed-test-loss = 8.090114 | rel_loss_delta = 8.295903446331762e-05 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 260/300 [13:23<02:09,  3.23s/it]2026-10-14 00:02:14.710 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 261, smoothed-train-loss = 8.100401 | smoothed-test-loss = 8.098767 | rel_loss_delta = -0.00015684212862105668 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 261/300 [13:27<02:05,  3.23s/it]2026-10-14 00:02:18.384 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 262, smoothed-train-loss = 8.101229 | smoothed-test-loss = 8.096717 | rel_loss_delta = 0.00010217804373442779 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 262/300 [13:30<02:07,  3.36s/it]2026-10-14 00:02:21.484 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 263, smoothed-train-loss = 8.097687 | smoothed-test-loss = 8.098035 | rel_loss_delta = -0.0004371916868933471 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 263/300 [13:33<02:01,  3.28s/it]2026-10-14 00:02:24.739 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 264, smoothed-train-loss = 8.099112 | smoothed-test-loss = 8.086692 | rel_loss_delta = 0.0001759850809897995 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 264/300 [13:37<01:57,  3.28s/it]2026-10-14 00:02:28.177 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 265, smoothed-train-loss = 8.100834 | smoothed-test-loss = 8.095503 | rel_loss_delta = 0.00021268606268867875 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 265/300 [13:40<01:56,  3.32s/it]2026-10-14 00:02:31.227 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 266, smoothed-train-loss = 8.099265 | smoothed-test-loss = 8.122837 | rel_loss_delta = -0.00019377053237218142 | loss_stagnation_count = 0

Epoch:  89%|████████▊ | 266/300 [13:43<01:50,  3.24s/it]2026-10-14 00:02:34.315 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 267, smoothed-train-loss = 8.103253 | smoothed-test-loss = 8.092342 | rel_loss_delta = 0.0004924551800803439 | loss_stagnation_count = 0

Epoch:  89%|████████▉ | 267/300 [13:46<01:45,  3.20s/it]2026-10-14 00:02:37.410 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 268, smoothed-train-loss = 8.099396 | smoothed-test-loss = 8.098048 | rel_loss_delta = -0.0004759994177102244 | loss_stagnation_count = 0

Epoch:  89%|████████▉ | 268/300 [13:49<01:41,  3.17s/it]2026-10-14 00:02:40.328 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 269, smoothed-train-loss = 8.101344 | smoothed-test-loss = 8.097485 | rel_loss_delta = 0.00024044453260762034 | loss_stagnation_count = 0

Epoch:  90%|████████▉ | 269/300 [13:52<01:35,  3.09s/it]2026-10-14 00:02:43.578 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 270, smoothed-train-loss = 8.101685 | smoothed-test-loss = 8.109068 | rel_loss_delta = 4.212834431765437e-05 | loss_stagnation_count = 1

Epoch:  90%|█████████ | 270/300 [13:55<01:34,  3.14s/it]2026-10-14 00:02:46.820 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 271, smoothed-train-loss = 8.099334 | smoothed-test-loss = 8.092419 | rel_loss_delta = -0.0002901804237764636 | loss_stagnation_count = 0

Epoch:  90%|█████████ | 271/300 [13:59<01:31,  3.17s/it]2026-10-14 00:02:49.998 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 272, smoothed-train-loss = 8.099958 | smoothed-test-loss = 8.090270 | rel_loss_delta = 7.70464080968663e-05 | loss_stagnation_count = 0

Epoch:  91%|█████████ | 272/300 [14:02<01:28,  3.17s/it]2026-10-14 00:02:53.122 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 273, smoothed-train-loss = 8.098976 | smoothed-test-loss = 8.091428 | rel_loss_delta = -0.00012117848092783191 | loss_stagnation_count = 0

Epoch:  91%|█████████ | 273/300 [14:05<01:25,  3.16s/it]2026-10-14 00:02:56.583 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 274, smoothed-train-loss = 8.097433 | smoothed-test-loss = 8.088703 | rel_loss_delta = -0.00019055678866578094 | loss_stagnation_count = 0

Epoch:  91%|█████████▏| 274/300 [14:08<01:24,  3.25s/it]2026-10-14 00:03:00.328 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 275, smoothed-train-loss = 8.099376 | smoothed-test-loss = 8.099653 | rel_loss_delta = 0.000239966245054255 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 275/300 [14:12<01:24,  3.40s/it]2026-10-14 00:03:03.794 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 276, smoothed-train-loss = 8.099649 | smoothed-test-loss = 8.093210 | rel_loss_delta = 3.366051337138052e-05 | loss_stagnation_count = 1

Epoch:  92%|█████████▏| 276/300 [14:16<01:22,  3.42s/it]2026-10-14 00:03:06.949 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 277, smoothed-train-loss = 8.101157 | smoothed-test-loss = 8.092045 | rel_loss_delta = 0.00018613863161300568 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 277/300 [14:19<01:16,  3.34s/it]2026-10-14 00:03:10.278 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 278, smoothed-train-loss = 8.098286 | smoothed-test-loss = 8.093702 | rel_loss_delta = -0.0003543176736611106 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 278/300 [14:22<01:13,  3.34s/it]2026-10-14 00:03:13.304 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 279, smoothed-train-loss = 8.098815 | smoothed-test-loss = 8.091416 | rel_loss_delta = 6.527535270550621e-05 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 279/300 [14:25<01:08,  3.24s/it]2026-10-14 00:03:16.256 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 280, smoothed-train-loss = 8.099050 | smoothed-test-loss = 8.090530 | rel_loss_delta = 2.9054724693615057e-05 | loss_stagnation_count = 1

Epoch:  93%|█████████▎| 280/300 [14:28<01:03,  3.16s/it]2026-10-14 00:03:19.239 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 281, smoothed-train-loss = 8.099069 | smoothed-test-loss = 8.092170 | rel_loss_delta = 2.3420454853701643e-06 | loss_stagnation_count = 2

Epoch:  94%|█████████▎| 281/300 [14:31<00:58,  3.10s/it]2026-10-14 00:03:22.344 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 282, smoothed-train-loss = 8.098844 | smoothed-test-loss = 8.092447 | rel_loss_delta = -2.7733582910122362e-05 | loss_stagnation_count = 3

Epoch:  94%|█████████▍| 282/300 [14:34<00:55,  3.10s/it]2026-10-14 00:03:25.560 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 283, smoothed-train-loss = 8.100422 | smoothed-test-loss = 8.094156 | rel_loss_delta = 0.00019478626724766602 | loss_stagnation_count = 0

Epoch:  94%|█████████▍| 283/300 [14:37<00:53,  3.14s/it]2026-10-14 00:03:28.857 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 284, smoothed-train-loss = 8.097329 | smoothed-test-loss = 8.100241 | rel_loss_delta = -0.00038185653397570114 | loss_stagnation_count = 0

Epoch:  95%|█████████▍| 284/300 [14:41<00:50,  3.19s/it]2026-10-14 00:03:31.791 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 285, smoothed-train-loss = 8.101917 | smoothed-test-loss = 8.116325 | rel_loss_delta = 0.0005666897216928654 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 285/300 [14:44<00:46,  3.11s/it]2026-10-14 00:03:34.816 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 286, smoothed-train-loss = 8.097814 | smoothed-test-loss = 8.099269 | rel_loss_delta = -0.0005064890832465723 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 286/300 [14:47<00:43,  3.08s/it]2026-10-14 00:03:38.487 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 287, smoothed-train-loss = 8.097920 | smoothed-test-loss = 8.098638 | rel_loss_delta = 1.3069466447012008e-05 | loss_stagnation_count = 1

Epoch:  96%|█████████▌| 287/300 [14:50<00:42,  3.26s/it]2026-10-14 00:03:42.025 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 288, smoothed-train-loss = 8.097862 | smoothed-test-loss = 8.097065 | rel_loss_delta = -7.132863419858787e-06 | loss_stagnation_count = 2

Epoch:  96%|█████████▌| 288/300 [14:54<00:40,  3.34s/it]2026-10-14 00:03:45.524 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 289, smoothed-train-loss = 8.101816 | smoothed-test-loss = 8.100402 | rel_loss_delta = 0.0004883299130322024 | loss_stagnation_count = 0

Epoch:  96%|█████████▋| 289/300 [14:57<00:37,  3.39s/it]2026-10-14 00:03:48.749 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 290, smoothed-train-loss = 8.101144 | smoothed-test-loss = 8.091518 | rel_loss_delta = -8.296723515695063e-05 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 290/300 [15:01<00:33,  3.34s/it]2026-10-14 00:03:51.974 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 291, smoothed-train-loss = 8.099576 | smoothed-test-loss = 8.087441 | rel_loss_delta = -0.00019359816732562364 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 291/300 [15:04<00:29,  3.31s/it]2026-10-14 00:03:55.077 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 292, smoothed-train-loss = 8.099079 | smoothed-test-loss = 8.099395 | rel_loss_delta = -6.140006840396596e-05 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 292/300 [15:07<00:25,  3.25s/it]2026-10-14 00:03:58.216 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 293, smoothed-train-loss = 8.098941 | smoothed-test-loss = 8.089537 | rel_loss_delta = -1.701042197659569e-05 | loss_stagnation_count = 1

Epoch:  98%|█████████▊| 293/300 [15:10<00:22,  3.21s/it]2026-10-14 00:04:01.521 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 294, smoothed-train-loss = 8.097445 | smoothed-test-loss = 8.086094 | rel_loss_delta = -0.00018467118858330148 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 294/300 [15:13<00:19,  3.24s/it]2026-10-14 00:04:04.846 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 295, smoothed-train-loss = 8.097446 | smoothed-test-loss = 8.082998 | rel_loss_delta = 1.2850675745877564e-07 | loss_stagnation_count = 1

Epoch:  98%|█████████▊| 295/300 [15:17<00:16,  3.27s/it]2026-10-14 00:04:08.030 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 296, smoothed-train-loss = 8.097240 | smoothed-test-loss = 8.093066 | rel_loss_delta = -2.5401462992303983e-05 | loss_stagnation_count = 2

Epoch:  99%|█████████▊| 296/300 [15:20<00:12,  3.24s/it]2026-10-14 00:04:11.131 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 297, smoothed-train-loss = 8.097002 | smoothed-test-loss = 8.109503 | rel_loss_delta = -2.9426828931077153e-05 | loss_stagnation_count = 3

Epoch:  99%|█████████▉| 297/300 [15:23<00:09,  3.20s/it]2026-10-14 00:04:14.353 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 298, smoothed-train-loss = 8.099957 | smoothed-test-loss = 8.090366 | rel_loss_delta = 0.0003649588555385932 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 298/300 [15:26<00:06,  3.21s/it]2026-10-14 00:04:17.581 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 299, smoothed-train-loss = 8.100002 | smoothed-test-loss = 8.096033 | rel_loss_delta = 5.479359053577622e-06 | loss_stagnation_count = 1

Epoch: 100%|█████████▉| 299/300 [15:29<00:03,  3.21s/it]2026-10-14 00:04:20.805 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 300, smoothed-train-loss = 8.097898 | smoothed-test-loss = 8.085014 | rel_loss_delta = -0.0002597051398887037 | loss_stagnation_count = 0

Epoch: 100%|██████████| 300/300 [15:33<00:00,  3.11s/it]
2026-10-14 00:04:20.810 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:8.083788871765137
2026-10-14 00:04:20.811 | INFO     | __main__:<module>:282 - Given num_reflow_iterations = 0 and current reflow_iteration = 0, no further reflows are needed, exiting the reflow loop
reflow iterations:   0%|          | 0/1 [15:34<?, ?it/s]
2026-10-14 00:04:20.816 | INFO     | __main__:<module>:340 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_4_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-13T23:48:45.910171.pkl
2026-10-14 00:04:20.816 | INFO     | __main__:<module>:341 - vt training process finished successfully.

Process finished with exit code 0
```

## ( B ) [2-Rectified Flow (1-Reflow)]
### Model

* Path
```log
models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_4_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-14T00:26:57.171654.pkl
```
* Log
```log
/home/mbaddar/Documents/mbaddar/ttflow/.venv/bin/python /home/mbaddar/Documents/mbaddar/ttflow/ttflow/training/nn/vt_trainer.py --data-snapshot-filepath data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_4_data_dim_2_n_train_test_eval_500000_200000_200000.pkl --vt-model nn --vt-opt adam --trajectory-model global --reflow-iterations 0 --nn-epochs 300 --reflow-iterations 1 --ode-solver-method euler 
2026-10-14 00:26:57.171 | INFO     | __main__:<module>:154 - args = Namespace(data_snapshot_filepath='data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_4_data_dim_2_n_train_test_eval_500000_200000_200000.pkl', use_gpu=True, vt_model='nn', vt_opt='adam', trajectory_model='global', ode_solver_method='euler', rectification_linearity='linear', reflow_iterations=1, nn_epochs=300, nn_hidden_dim=128, batch_size=2048, nn_depth=2, lr=0.001, nn_reg_lambda=0.001, patience=10, min_rel_loss_delta=5e-05, dry_run=False, logs_dir='logs', output_model_dir='models/experimental')
2026-10-14 00:26:57.171 | INFO     | __main__:<module>:166 - is GPU/CUDA available = True
2026-10-14 00:26:57.171 | INFO     | __main__:<module>:169 - device = cuda
2026-10-14 00:26:57.266 | INFO     | __main__:<module>:174 - Using data snapshot: data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_4_data_dim_2_n_train_test_eval_500000_200000_200000.pkl
2026-10-14 00:26:57.266 | INFO     | ttflow.utils:check_system_memory_and_swap:56 - ram = 30.96, swap = 40.0 GB
2026-10-14 00:26:57.283 | INFO     | __main__:<module>:203 - data pkl size = 0.01 GB
2026-10-14 00:26:57.286 | INFO     | __main__:<module>:214 - Generating training data for the global trajectory training process..
2026-10-14 00:26:57.466 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-14 00:26:58.101 | INFO     | __main__:<module>:241 - ***********************************************************Training Model Specs:
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
    - Reflow Iterations:  1**********************************************************
reflow iterations:   0%|          | 0/2 [00:00<?, ?it/s]2026-10-14 00:26:58.101 | INFO     | __main__:<module>:266 - Reflow iteration: 0
2026-10-14 00:26:59.074 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:57 - @epoch = 0 (before training), loss_fn = MSELoss(), loss_value = 49.014271
2026-10-14 00:26:59.079 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:64 - MSELoss(test) before training:49.01426696777344

Epoch:   0%|          | 0/300 [00:00<?, ?it/s]2026-10-14 00:27:02.466 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 1, smoothed-train-loss = 14.297655 | smoothed-test-loss = 9.801542 | rel_loss_delta = inf | loss_stagnation_count = 0

Epoch:   0%|          | 1/300 [00:03<16:52,  3.39s/it]2026-10-14 00:27:05.834 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 2, smoothed-train-loss = 9.950359 | smoothed-test-loss = 9.382481 | rel_loss_delta = inf | loss_stagnation_count = 0

Epoch:   1%|          | 2/300 [00:06<16:46,  3.38s/it]2026-10-14 00:27:09.099 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 3, smoothed-train-loss = 9.266893 | smoothed-test-loss = 9.122937 | rel_loss_delta = -0.0686875709764164 | loss_stagnation_count = 0

Epoch:   1%|          | 3/300 [00:10<16:27,  3.32s/it]2026-10-14 00:27:12.208 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 4, smoothed-train-loss = 8.973404 | smoothed-test-loss = 8.884329 | rel_loss_delta = -0.03167072025418464 | loss_stagnation_count = 0

Epoch:   1%|▏         | 4/300 [00:13<15:59,  3.24s/it]2026-10-14 00:27:15.154 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 5, smoothed-train-loss = 8.732083 | smoothed-test-loss = 8.655772 | rel_loss_delta = -0.026892938332316624 | loss_stagnation_count = 0

Epoch:   2%|▏         | 5/300 [00:16<15:24,  3.13s/it]2026-10-14 00:27:18.227 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 6, smoothed-train-loss = 8.541638 | smoothed-test-loss = 8.493772 | rel_loss_delta = -0.021809785080953686 | loss_stagnation_count = 0

Epoch:   2%|▏         | 6/300 [00:19<15:15,  3.11s/it]2026-10-14 00:27:21.191 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 7, smoothed-train-loss = 8.416873 | smoothed-test-loss = 8.397410 | rel_loss_delta = -0.014606704818707041 | loss_stagnation_count = 0

Epoch:   2%|▏         | 7/300 [00:22<14:57,  3.06s/it]2026-10-14 00:27:24.358 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 8, smoothed-train-loss = 8.332960 | smoothed-test-loss = 8.323498 | rel_loss_delta = -0.009969552498276735 | loss_stagnation_count = 0

Epoch:   3%|▎         | 8/300 [00:25<15:04,  3.10s/it]2026-10-14 00:27:27.790 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 9, smoothed-train-loss = 8.275301 | smoothed-test-loss = 8.289374 | rel_loss_delta = -0.006919387450391817 | loss_stagnation_count = 0

Epoch:   3%|▎         | 9/300 [00:28<15:31,  3.20s/it]2026-10-14 00:27:31.460 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 10, smoothed-train-loss = 8.231980 | smoothed-test-loss = 8.249178 | rel_loss_delta = -0.005235043901900464 | loss_stagnation_count = 0

Epoch:   3%|▎         | 10/300 [00:32<16:10,  3.35s/it]2026-10-14 00:27:35.121 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 11, smoothed-train-loss = 8.202822 | smoothed-test-loss = 8.226268 | rel_loss_delta = -0.0035419677480491357 | loss_stagnation_count = 0

Epoch:   4%|▎         | 11/300 [00:36<16:34,  3.44s/it]2026-10-14 00:27:38.561 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 12, smoothed-train-loss = 8.182914 | smoothed-test-loss = 8.206597 | rel_loss_delta = -0.002427055898821887 | loss_stagnation_count = 0

Epoch:   4%|▍         | 12/300 [00:39<16:31,  3.44s/it]2026-10-14 00:27:42.468 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 13, smoothed-train-loss = 8.165016 | smoothed-test-loss = 8.175618 | rel_loss_delta = -0.0021871792896788977 | loss_stagnation_count = 0

Epoch:   4%|▍         | 13/300 [00:43<17:08,  3.58s/it]2026-10-14 00:27:46.096 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 14, smoothed-train-loss = 8.154587 | smoothed-test-loss = 8.190674 | rel_loss_delta = -0.001277287489406277 | loss_stagnation_count = 0

Epoch:   5%|▍         | 14/300 [00:47<17:08,  3.60s/it]2026-10-14 00:27:49.441 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 15, smoothed-train-loss = 8.143922 | smoothed-test-loss = 8.177521 | rel_loss_delta = -0.001307882675340284 | loss_stagnation_count = 0

Epoch:   5%|▌         | 15/300 [00:50<16:43,  3.52s/it]2026-10-14 00:27:52.529 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 16, smoothed-train-loss = 8.136205 | smoothed-test-loss = 8.158296 | rel_loss_delta = -0.0009475312059217648 | loss_stagnation_count = 0

Epoch:   5%|▌         | 16/300 [00:53<16:02,  3.39s/it]2026-10-14 00:27:55.782 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 17, smoothed-train-loss = 8.129240 | smoothed-test-loss = 8.155151 | rel_loss_delta = -0.0008561252937518836 | loss_stagnation_count = 0

Epoch:   6%|▌         | 17/300 [00:56<15:47,  3.35s/it]2026-10-14 00:27:59.188 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 18, smoothed-train-loss = 8.123405 | smoothed-test-loss = 8.160936 | rel_loss_delta = -0.0007176758924535284 | loss_stagnation_count = 0

Epoch:   6%|▌         | 18/300 [01:00<15:49,  3.37s/it]2026-10-14 00:28:02.553 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 19, smoothed-train-loss = 8.119014 | smoothed-test-loss = 8.168489 | rel_loss_delta = -0.0005405298982386692 | loss_stagnation_count = 0

Epoch:   6%|▋         | 19/300 [01:03<15:45,  3.37s/it]2026-10-14 00:28:05.838 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 20, smoothed-train-loss = 8.117463 | smoothed-test-loss = 8.139602 | rel_loss_delta = -0.0001911349948081431 | loss_stagnation_count = 0

Epoch:   7%|▋         | 20/300 [01:06<15:35,  3.34s/it]2026-10-14 00:28:09.459 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 21, smoothed-train-loss = 8.114154 | smoothed-test-loss = 8.156344 | rel_loss_delta = -0.00040764147043175885 | loss_stagnation_count = 0

Epoch:   7%|▋         | 21/300 [01:10<15:55,  3.43s/it]2026-10-14 00:28:12.442 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 22, smoothed-train-loss = 8.110752 | smoothed-test-loss = 8.131337 | rel_loss_delta = -0.0004192771745826233 | loss_stagnation_count = 0

Epoch:   7%|▋         | 22/300 [01:13<15:15,  3.29s/it]2026-10-14 00:28:15.547 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 23, smoothed-train-loss = 8.109596 | smoothed-test-loss = 8.149032 | rel_loss_delta = -0.00014246078373741133 | loss_stagnation_count = 0

Epoch:   8%|▊         | 23/300 [01:16<14:56,  3.24s/it]2026-10-14 00:28:18.464 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 24, smoothed-train-loss = 8.105820 | smoothed-test-loss = 8.129962 | rel_loss_delta = -0.0004656691106472279 | loss_stagnation_count = 0

Epoch:   8%|▊         | 24/300 [01:19<14:26,  3.14s/it]2026-10-14 00:28:21.435 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 25, smoothed-train-loss = 8.104609 | smoothed-test-loss = 8.160654 | rel_loss_delta = -0.00014930148095863372 | loss_stagnation_count = 0

Epoch:   8%|▊         | 25/300 [01:22<14:09,  3.09s/it]2026-10-14 00:28:24.507 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 26, smoothed-train-loss = 8.104544 | smoothed-test-loss = 8.136939 | rel_loss_delta = -8.060592920397996e-06 | loss_stagnation_count = 1

Epoch:   9%|▊         | 26/300 [01:25<14:05,  3.08s/it]2026-10-14 00:28:27.458 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 27, smoothed-train-loss = 8.104655 | smoothed-test-loss = 8.127229 | rel_loss_delta = 1.3689928554066994e-05 | loss_stagnation_count = 2

Epoch:   9%|▉         | 27/300 [01:28<13:51,  3.04s/it]2026-10-14 00:28:30.483 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 28, smoothed-train-loss = 8.103264 | smoothed-test-loss = 8.120724 | rel_loss_delta = -0.00017164845446498983 | loss_stagnation_count = 0

Epoch:   9%|▉         | 28/300 [01:31<13:46,  3.04s/it]2026-10-14 00:28:33.394 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 29, smoothed-train-loss = 8.102981 | smoothed-test-loss = 8.137478 | rel_loss_delta = -3.4877703779460516e-05 | loss_stagnation_count = 1

Epoch:  10%|▉         | 29/300 [01:34<13:33,  3.00s/it]2026-10-14 00:28:36.369 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 30, smoothed-train-loss = 8.099165 | smoothed-test-loss = 8.143162 | rel_loss_delta = -0.00047098842990147985 | loss_stagnation_count = 0

Epoch:  10%|█         | 30/300 [01:37<13:28,  2.99s/it]2026-10-14 00:28:39.342 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 31, smoothed-train-loss = 8.099152 | smoothed-test-loss = 8.125907 | rel_loss_delta = -1.5846373228965618e-06 | loss_stagnation_count = 1

Epoch:  10%|█         | 31/300 [01:40<13:23,  2.99s/it]2026-10-14 00:28:42.332 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 32, smoothed-train-loss = 8.100219 | smoothed-test-loss = 8.142849 | rel_loss_delta = 0.0001316911878346036 | loss_stagnation_count = 0

Epoch:  11%|█         | 32/300 [01:43<13:20,  2.99s/it]2026-10-14 00:28:45.162 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 33, smoothed-train-loss = 8.095792 | smoothed-test-loss = 8.125749 | rel_loss_delta = -0.0005464657731162854 | loss_stagnation_count = 0

Epoch:  11%|█         | 33/300 [01:46<13:05,  2.94s/it]2026-10-14 00:28:48.345 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 34, smoothed-train-loss = 8.095491 | smoothed-test-loss = 8.136117 | rel_loss_delta = -3.724606202020928e-05 | loss_stagnation_count = 1

Epoch:  11%|█▏        | 34/300 [01:49<13:21,  3.01s/it]2026-10-14 00:28:51.443 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 35, smoothed-train-loss = 8.097254 | smoothed-test-loss = 8.124500 | rel_loss_delta = 0.00021776071106711784 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 35/300 [01:52<13:25,  3.04s/it]2026-10-14 00:28:54.456 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 36, smoothed-train-loss = 8.093729 | smoothed-test-loss = 8.128750 | rel_loss_delta = -0.000435274334875038 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 36/300 [01:55<13:20,  3.03s/it]2026-10-14 00:28:57.322 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 37, smoothed-train-loss = 8.092549 | smoothed-test-loss = 8.144295 | rel_loss_delta = -0.00014582333845601972 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 37/300 [01:58<13:04,  2.98s/it]2026-10-14 00:29:00.329 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 38, smoothed-train-loss = 8.091772 | smoothed-test-loss = 8.118057 | rel_loss_delta = -9.595762939185213e-05 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 38/300 [02:01<13:03,  2.99s/it]2026-10-14 00:29:03.281 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 39, smoothed-train-loss = 8.091477 | smoothed-test-loss = 8.125752 | rel_loss_delta = -3.6450397650411377e-05 | loss_stagnation_count = 1

Epoch:  13%|█▎        | 39/300 [02:04<12:57,  2.98s/it]2026-10-14 00:29:06.206 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 40, smoothed-train-loss = 8.089682 | smoothed-test-loss = 8.121174 | rel_loss_delta = -0.00022189963618742757 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 40/300 [02:07<12:50,  2.96s/it]2026-10-14 00:29:09.232 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 41, smoothed-train-loss = 8.091821 | smoothed-test-loss = 8.177408 | rel_loss_delta = 0.00026447885676497094 | loss_stagnation_count = 0

Epoch:  14%|█▎        | 41/300 [02:10<12:52,  2.98s/it]2026-10-14 00:29:12.147 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 42, smoothed-train-loss = 8.090870 | smoothed-test-loss = 8.138317 | rel_loss_delta = -0.00011752402768751315 | loss_stagnation_count = 0

Epoch:  14%|█▍        | 42/300 [02:13<12:44,  2.96s/it]2026-10-14 00:29:15.139 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 43, smoothed-train-loss = 8.088582 | smoothed-test-loss = 8.136740 | rel_loss_delta = -0.00028285446316999143 | loss_stagnation_count = 0

Epoch:  14%|█▍        | 43/300 [02:16<12:43,  2.97s/it]2026-10-14 00:29:18.122 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 44, smoothed-train-loss = 8.090481 | smoothed-test-loss = 8.135320 | rel_loss_delta = 0.00023480195693862773 | loss_stagnation_count = 0

Epoch:  15%|█▍        | 44/300 [02:19<12:41,  2.97s/it]2026-10-14 00:29:21.012 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 45, smoothed-train-loss = 8.089304 | smoothed-test-loss = 8.112667 | rel_loss_delta = -0.00014552143428674255 | loss_stagnation_count = 0

Epoch:  15%|█▌        | 45/300 [02:21<12:31,  2.95s/it]2026-10-14 00:29:23.966 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 46, smoothed-train-loss = 8.089668 | smoothed-test-loss = 8.115746 | rel_loss_delta = 4.5018218075183794e-05 | loss_stagnation_count = 1

Epoch:  15%|█▌        | 46/300 [02:24<12:29,  2.95s/it]2026-10-14 00:29:26.875 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 47, smoothed-train-loss = 8.087894 | smoothed-test-loss = 8.134765 | rel_loss_delta = -0.00021926616514639832 | loss_stagnation_count = 0

Epoch:  16%|█▌        | 47/300 [02:27<12:23,  2.94s/it]2026-10-14 00:29:29.888 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 48, smoothed-train-loss = 8.090360 | smoothed-test-loss = 8.123879 | rel_loss_delta = 0.00030488789254703134 | loss_stagnation_count = 0

Epoch:  16%|█▌        | 48/300 [02:30<12:26,  2.96s/it]2026-10-14 00:29:32.774 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 49, smoothed-train-loss = 8.086118 | smoothed-test-loss = 8.108509 | rel_loss_delta = -0.0005243072592919501 | loss_stagnation_count = 0

Epoch:  16%|█▋        | 49/300 [02:33<12:17,  2.94s/it]2026-10-14 00:29:35.726 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 50, smoothed-train-loss = 8.085958 | smoothed-test-loss = 8.114670 | rel_loss_delta = -1.9750665771698465e-05 | loss_stagnation_count = 1

Epoch:  17%|█▋        | 50/300 [02:36<12:15,  2.94s/it]2026-10-14 00:29:38.702 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 51, smoothed-train-loss = 8.085757 | smoothed-test-loss = 8.131696 | rel_loss_delta = -2.4894024000735192e-05 | loss_stagnation_count = 2

Epoch:  17%|█▋        | 51/300 [02:39<12:15,  2.95s/it]2026-10-14 00:29:41.671 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 52, smoothed-train-loss = 8.082349 | smoothed-test-loss = 8.114657 | rel_loss_delta = -0.000421431114989803 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 52/300 [02:42<12:13,  2.96s/it]2026-10-14 00:29:44.584 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 53, smoothed-train-loss = 8.083182 | smoothed-test-loss = 8.151402 | rel_loss_delta = 0.00010306315725339805 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 53/300 [02:45<12:07,  2.94s/it]2026-10-14 00:29:47.543 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 54, smoothed-train-loss = 8.084982 | smoothed-test-loss = 8.125265 | rel_loss_delta = 0.00022265230152341913 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 54/300 [02:48<12:05,  2.95s/it]2026-10-14 00:29:50.539 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 55, smoothed-train-loss = 8.081981 | smoothed-test-loss = 8.119218 | rel_loss_delta = -0.00037121908896801546 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 55/300 [02:51<12:05,  2.96s/it]2026-10-14 00:29:53.497 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 56, smoothed-train-loss = 8.083272 | smoothed-test-loss = 8.112232 | rel_loss_delta = 0.00015976738910137528 | loss_stagnation_count = 0

Epoch:  19%|█▊        | 56/300 [02:54<12:02,  2.96s/it]2026-10-14 00:29:56.355 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 57, smoothed-train-loss = 8.082007 | smoothed-test-loss = 8.145494 | rel_loss_delta = -0.000156475982771671 | loss_stagnation_count = 0

Epoch:  19%|█▉        | 57/300 [02:57<11:52,  2.93s/it]2026-10-14 00:29:59.280 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 58, smoothed-train-loss = 8.086339 | smoothed-test-loss = 8.127158 | rel_loss_delta = 0.0005359946358867286 | loss_stagnation_count = 0

Epoch:  19%|█▉        | 58/300 [03:00<11:48,  2.93s/it]2026-10-14 00:30:02.334 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 59, smoothed-train-loss = 8.083510 | smoothed-test-loss = 8.111270 | rel_loss_delta = -0.0003499313868065729 | loss_stagnation_count = 0

Epoch:  20%|█▉        | 59/300 [03:03<11:54,  2.97s/it]2026-10-14 00:30:05.334 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 60, smoothed-train-loss = 8.080704 | smoothed-test-loss = 8.120872 | rel_loss_delta = -0.00034707838297203225 | loss_stagnation_count = 0

Epoch:  20%|██        | 60/300 [03:06<11:54,  2.98s/it]2026-10-14 00:30:08.249 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 61, smoothed-train-loss = 8.080272 | smoothed-test-loss = 8.120446 | rel_loss_delta = -5.339460831055206e-05 | loss_stagnation_count = 0

Epoch:  20%|██        | 61/300 [03:09<11:46,  2.96s/it]2026-10-14 00:30:11.281 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 62, smoothed-train-loss = 8.083181 | smoothed-test-loss = 8.134784 | rel_loss_delta = 0.00036000360817406787 | loss_stagnation_count = 0

Epoch:  21%|██        | 62/300 [03:12<11:49,  2.98s/it]2026-10-14 00:30:14.245 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 63, smoothed-train-loss = 8.083000 | smoothed-test-loss = 8.118454 | rel_loss_delta = -2.2493689641207215e-05 | loss_stagnation_count = 1

Epoch:  21%|██        | 63/300 [03:15<11:45,  2.98s/it]2026-10-14 00:30:17.150 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 64, smoothed-train-loss = 8.081541 | smoothed-test-loss = 8.104773 | rel_loss_delta = -0.00018049649601082225 | loss_stagnation_count = 0

Epoch:  21%|██▏       | 64/300 [03:18<11:37,  2.95s/it]2026-10-14 00:30:19.989 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 65, smoothed-train-loss = 8.080836 | smoothed-test-loss = 8.120287 | rel_loss_delta = -8.716934989751277e-05 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 65/300 [03:20<11:26,  2.92s/it]2026-10-14 00:30:23.008 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 66, smoothed-train-loss = 8.083267 | smoothed-test-loss = 8.113592 | rel_loss_delta = 0.0003008004599449809 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 66/300 [03:23<11:30,  2.95s/it]2026-10-14 00:30:25.963 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 67, smoothed-train-loss = 8.081712 | smoothed-test-loss = 8.103855 | rel_loss_delta = -0.00019239478776344432 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 67/300 [03:26<11:27,  2.95s/it]2026-10-14 00:30:28.852 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 68, smoothed-train-loss = 8.081710 | smoothed-test-loss = 8.122016 | rel_loss_delta = -2.1799039137890886e-07 | loss_stagnation_count = 1

Epoch:  23%|██▎       | 68/300 [03:29<11:20,  2.93s/it]2026-10-14 00:30:31.685 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 69, smoothed-train-loss = 8.084171 | smoothed-test-loss = 8.110883 | rel_loss_delta = 0.0003045328647006492 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 69/300 [03:32<11:10,  2.90s/it]2026-10-14 00:30:34.721 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 70, smoothed-train-loss = 8.079626 | smoothed-test-loss = 8.104058 | rel_loss_delta = -0.0005621924479800849 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 70/300 [03:35<11:16,  2.94s/it]2026-10-14 00:30:37.680 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 71, smoothed-train-loss = 8.083934 | smoothed-test-loss = 8.097558 | rel_loss_delta = 0.0005332215812768661 | loss_stagnation_count = 0

Epoch:  24%|██▎       | 71/300 [03:38<11:14,  2.95s/it]2026-10-14 00:30:40.773 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 72, smoothed-train-loss = 8.082269 | smoothed-test-loss = 8.114768 | rel_loss_delta = -0.00020598864664868392 | loss_stagnation_count = 0

Epoch:  24%|██▍       | 72/300 [03:41<11:21,  2.99s/it]2026-10-14 00:30:43.575 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 73, smoothed-train-loss = 8.087527 | smoothed-test-loss = 8.102458 | rel_loss_delta = 0.0006504664332557989 | loss_stagnation_count = 0

Epoch:  24%|██▍       | 73/300 [03:44<11:06,  2.93s/it]2026-10-14 00:30:46.579 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 74, smoothed-train-loss = 8.080249 | smoothed-test-loss = 8.113972 | rel_loss_delta = -0.000899879294296772 | loss_stagnation_count = 0

Epoch:  25%|██▍       | 74/300 [03:47<11:07,  2.96s/it]2026-10-14 00:30:49.596 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 75, smoothed-train-loss = 8.077805 | smoothed-test-loss = 8.119284 | rel_loss_delta = -0.00030237966393175547 | loss_stagnation_count = 0

Epoch:  25%|██▌       | 75/300 [03:50<11:09,  2.97s/it]2026-10-14 00:30:52.697 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 76, smoothed-train-loss = 8.081376 | smoothed-test-loss = 8.131495 | rel_loss_delta = 0.0004419609419033247 | loss_stagnation_count = 0

Epoch:  25%|██▌       | 76/300 [03:53<11:14,  3.01s/it]2026-10-14 00:30:55.497 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 77, smoothed-train-loss = 8.079541 | smoothed-test-loss = 8.119486 | rel_loss_delta = -0.00022695574783462106 | loss_stagnation_count = 0

Epoch:  26%|██▌       | 77/300 [03:56<10:57,  2.95s/it]2026-10-14 00:30:58.408 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 78, smoothed-train-loss = 8.078249 | smoothed-test-loss = 8.112637 | rel_loss_delta = -0.0001599439964477185 | loss_stagnation_count = 0

Epoch:  26%|██▌       | 78/300 [03:59<10:52,  2.94s/it]2026-10-14 00:31:01.442 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 79, smoothed-train-loss = 8.078316 | smoothed-test-loss = 8.108252 | rel_loss_delta = 8.232496372468072e-06 | loss_stagnation_count = 1

Epoch:  26%|██▋       | 79/300 [04:02<10:55,  2.97s/it]2026-10-14 00:31:04.439 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 80, smoothed-train-loss = 8.078957 | smoothed-test-loss = 8.100940 | rel_loss_delta = 7.935139128051663e-05 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 80/300 [04:05<10:54,  2.98s/it]2026-10-14 00:31:07.366 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 81, smoothed-train-loss = 8.076344 | smoothed-test-loss = 8.102454 | rel_loss_delta = -0.000323440470088661 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 81/300 [04:08<10:48,  2.96s/it]2026-10-14 00:31:10.428 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 82, smoothed-train-loss = 8.078229 | smoothed-test-loss = 8.123351 | rel_loss_delta = 0.00023350257013606115 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 82/300 [04:11<10:52,  2.99s/it]2026-10-14 00:31:13.531 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 83, smoothed-train-loss = 8.079653 | smoothed-test-loss = 8.117451 | rel_loss_delta = 0.0001761869049814366 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 83/300 [04:14<10:56,  3.02s/it]2026-10-14 00:31:16.488 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 84, smoothed-train-loss = 8.077639 | smoothed-test-loss = 8.111663 | rel_loss_delta = -0.0002492820120729404 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 84/300 [04:17<10:48,  3.00s/it]2026-10-14 00:31:19.379 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 85, smoothed-train-loss = 8.078158 | smoothed-test-loss = 8.098541 | rel_loss_delta = 6.424626682503535e-05 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 85/300 [04:20<10:38,  2.97s/it]2026-10-14 00:31:22.351 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 86, smoothed-train-loss = 8.079156 | smoothed-test-loss = 8.119224 | rel_loss_delta = 0.00012356493129814613 | loss_stagnation_count = 0

Epoch:  29%|██▊       | 86/300 [04:23<10:35,  2.97s/it]2026-10-14 00:31:25.304 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 87, smoothed-train-loss = 8.079485 | smoothed-test-loss = 8.103098 | rel_loss_delta = 4.07241676759897e-05 | loss_stagnation_count = 1

Epoch:  29%|██▉       | 87/300 [04:26<10:31,  2.97s/it]2026-10-14 00:31:28.427 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 88, smoothed-train-loss = 8.077060 | smoothed-test-loss = 8.117786 | rel_loss_delta = -0.00030015620249942813 | loss_stagnation_count = 0

Epoch:  29%|██▉       | 88/300 [04:29<10:38,  3.01s/it]2026-10-14 00:31:31.666 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 89, smoothed-train-loss = 8.076287 | smoothed-test-loss = 8.117932 | rel_loss_delta = -9.564786125184068e-05 | loss_stagnation_count = 0

Epoch:  30%|██▉       | 89/300 [04:32<10:50,  3.08s/it]2026-10-14 00:31:34.924 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 90, smoothed-train-loss = 8.078772 | smoothed-test-loss = 8.098610 | rel_loss_delta = 0.0003077115809044201 | loss_stagnation_count = 0

Epoch:  30%|███       | 90/300 [04:35<10:58,  3.13s/it]2026-10-14 00:31:37.933 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 91, smoothed-train-loss = 8.080509 | smoothed-test-loss = 8.106884 | rel_loss_delta = 0.00021502757904166405 | loss_stagnation_count = 0

Epoch:  30%|███       | 91/300 [04:38<10:47,  3.10s/it]2026-10-14 00:31:41.216 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 92, smoothed-train-loss = 8.075537 | smoothed-test-loss = 8.112924 | rel_loss_delta = -0.0006153795432689742 | loss_stagnation_count = 0

Epoch:  31%|███       | 92/300 [04:42<10:55,  3.15s/it]2026-10-14 00:31:44.500 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 93, smoothed-train-loss = 8.077308 | smoothed-test-loss = 8.105060 | rel_loss_delta = 0.00021927736166095523 | loss_stagnation_count = 0

Epoch:  31%|███       | 93/300 [04:45<11:00,  3.19s/it]2026-10-14 00:31:47.602 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 94, smoothed-train-loss = 8.078837 | smoothed-test-loss = 8.101180 | rel_loss_delta = 0.00018935838746146717 | loss_stagnation_count = 0

Epoch:  31%|███▏      | 94/300 [04:48<10:51,  3.16s/it]2026-10-14 00:31:50.815 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 95, smoothed-train-loss = 8.076353 | smoothed-test-loss = 8.113073 | rel_loss_delta = -0.0003074871074146273 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 95/300 [04:51<10:51,  3.18s/it]2026-10-14 00:31:54.096 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 96, smoothed-train-loss = 8.075659 | smoothed-test-loss = 8.103331 | rel_loss_delta = -8.595554540882787e-05 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 96/300 [04:55<10:54,  3.21s/it]2026-10-14 00:31:57.238 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 97, smoothed-train-loss = 8.075301 | smoothed-test-loss = 8.113668 | rel_loss_delta = -4.429398840219966e-05 | loss_stagnation_count = 1

Epoch:  32%|███▏      | 97/300 [04:58<10:47,  3.19s/it]2026-10-14 00:32:00.451 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 98, smoothed-train-loss = 8.077371 | smoothed-test-loss = 8.100164 | rel_loss_delta = 0.000256268384027592 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 98/300 [05:01<10:45,  3.20s/it]2026-10-14 00:32:03.497 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 99, smoothed-train-loss = 8.075004 | smoothed-test-loss = 8.093996 | rel_loss_delta = -0.0002929522896227892 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 99/300 [05:04<10:33,  3.15s/it]2026-10-14 00:32:06.638 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 100, smoothed-train-loss = 8.072057 | smoothed-test-loss = 8.124117 | rel_loss_delta = -0.00036498347519727604 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 100/300 [05:07<10:29,  3.15s/it]2026-10-14 00:32:09.629 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 101, smoothed-train-loss = 8.074938 | smoothed-test-loss = 8.110066 | rel_loss_delta = 0.0003569195045988128 | loss_stagnation_count = 0

Epoch:  34%|███▎      | 101/300 [05:10<10:17,  3.10s/it]2026-10-14 00:32:12.691 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 102, smoothed-train-loss = 8.073542 | smoothed-test-loss = 8.108868 | rel_loss_delta = -0.00017293470443507317 | loss_stagnation_count = 0

Epoch:  34%|███▍      | 102/300 [05:13<10:11,  3.09s/it]2026-10-14 00:32:16.081 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 103, smoothed-train-loss = 8.075100 | smoothed-test-loss = 8.109346 | rel_loss_delta = 0.00019296816009034006 | loss_stagnation_count = 0

Epoch:  34%|███▍      | 103/300 [05:17<10:26,  3.18s/it]2026-10-14 00:32:19.053 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 104, smoothed-train-loss = 8.076291 | smoothed-test-loss = 8.108032 | rel_loss_delta = 0.0001475155870904592 | loss_stagnation_count = 0

Epoch:  35%|███▍      | 104/300 [05:19<10:10,  3.12s/it]2026-10-14 00:32:22.049 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 105, smoothed-train-loss = 8.072122 | smoothed-test-loss = 8.098308 | rel_loss_delta = -0.0005161891288490335 | loss_stagnation_count = 0

Epoch:  35%|███▌      | 105/300 [05:22<10:00,  3.08s/it]2026-10-14 00:32:25.015 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 106, smoothed-train-loss = 8.073515 | smoothed-test-loss = 8.101141 | rel_loss_delta = 0.00017263549477314932 | loss_stagnation_count = 0

Epoch:  35%|███▌      | 106/300 [05:25<09:51,  3.05s/it]2026-10-14 00:32:28.019 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 107, smoothed-train-loss = 8.074136 | smoothed-test-loss = 8.106058 | rel_loss_delta = 7.692368828067652e-05 | loss_stagnation_count = 0

Epoch:  36%|███▌      | 107/300 [05:28<09:45,  3.03s/it]2026-10-14 00:32:31.108 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 108, smoothed-train-loss = 8.071657 | smoothed-test-loss = 8.095504 | rel_loss_delta = -0.00030703793661073355 | loss_stagnation_count = 0

Epoch:  36%|███▌      | 108/300 [05:32<09:45,  3.05s/it]2026-10-14 00:32:34.079 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 109, smoothed-train-loss = 8.074679 | smoothed-test-loss = 8.102873 | rel_loss_delta = 0.0003743717049255052 | loss_stagnation_count = 0

Epoch:  36%|███▋      | 109/300 [05:34<09:38,  3.03s/it]2026-10-14 00:32:37.079 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 110, smoothed-train-loss = 8.073171 | smoothed-test-loss = 8.122265 | rel_loss_delta = -0.00018674618622895934 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 110/300 [05:37<09:33,  3.02s/it]2026-10-14 00:32:40.184 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 111, smoothed-train-loss = 8.074347 | smoothed-test-loss = 8.101269 | rel_loss_delta = 0.0001456251911055265 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 111/300 [05:41<09:35,  3.04s/it]2026-10-14 00:32:43.176 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 112, smoothed-train-loss = 8.071915 | smoothed-test-loss = 8.105070 | rel_loss_delta = -0.0003012212684537128 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 112/300 [05:44<09:29,  3.03s/it]2026-10-14 00:32:45.999 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 113, smoothed-train-loss = 8.074576 | smoothed-test-loss = 8.097604 | rel_loss_delta = 0.0003296904980758886 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 113/300 [05:46<09:14,  2.97s/it]2026-10-14 00:32:48.924 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 114, smoothed-train-loss = 8.074581 | smoothed-test-loss = 8.110070 | rel_loss_delta = 5.734070465275602e-07 | loss_stagnation_count = 1

Epoch:  38%|███▊      | 114/300 [05:49<09:09,  2.95s/it]2026-10-14 00:32:51.858 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 115, smoothed-train-loss = 8.073168 | smoothed-test-loss = 8.107140 | rel_loss_delta = -0.00017490660650005196 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 115/300 [05:52<09:05,  2.95s/it]2026-10-14 00:32:54.934 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 116, smoothed-train-loss = 8.077292 | smoothed-test-loss = 8.134574 | rel_loss_delta = 0.0005107694914190821 | loss_stagnation_count = 0

Epoch:  39%|███▊      | 116/300 [05:55<09:09,  2.99s/it]2026-10-14 00:32:57.903 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 117, smoothed-train-loss = 8.077288 | smoothed-test-loss = 8.111438 | rel_loss_delta = -4.557978987363427e-07 | loss_stagnation_count = 1

Epoch:  39%|███▉      | 117/300 [05:58<09:05,  2.98s/it]2026-10-14 00:33:00.900 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 118, smoothed-train-loss = 8.074393 | smoothed-test-loss = 8.104527 | rel_loss_delta = -0.0003584013054671079 | loss_stagnation_count = 0

Epoch:  39%|███▉      | 118/300 [06:01<09:03,  2.99s/it]2026-10-14 00:33:03.869 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 119, smoothed-train-loss = 8.073158 | smoothed-test-loss = 8.107938 | rel_loss_delta = -0.00015295179188258744 | loss_stagnation_count = 0

Epoch:  40%|███▉      | 119/300 [06:04<08:59,  2.98s/it]2026-10-14 00:33:07.007 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 120, smoothed-train-loss = 8.073154 | smoothed-test-loss = 8.097470 | rel_loss_delta = -5.778901927911601e-07 | loss_stagnation_count = 1

Epoch:  40%|████      | 120/300 [06:07<09:05,  3.03s/it]2026-10-14 00:33:09.971 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 121, smoothed-train-loss = 8.074886 | smoothed-test-loss = 8.105422 | rel_loss_delta = 0.00021462776063188192 | loss_stagnation_count = 0

Epoch:  40%|████      | 121/300 [06:10<08:58,  3.01s/it]2026-10-14 00:33:12.901 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 122, smoothed-train-loss = 8.072552 | smoothed-test-loss = 8.102680 | rel_loss_delta = -0.00028906745908089624 | loss_stagnation_count = 0

Epoch:  41%|████      | 122/300 [06:13<08:51,  2.99s/it]2026-10-14 00:33:15.809 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 123, smoothed-train-loss = 8.072143 | smoothed-test-loss = 8.105739 | rel_loss_delta = -5.070164843248567e-05 | loss_stagnation_count = 0

Epoch:  41%|████      | 123/300 [06:16<08:44,  2.96s/it]2026-10-14 00:33:18.851 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 124, smoothed-train-loss = 8.075417 | smoothed-test-loss = 8.110688 | rel_loss_delta = 0.0004056390276290535 | loss_stagnation_count = 0

Epoch:  41%|████▏     | 124/300 [06:19<08:45,  2.99s/it]2026-10-14 00:33:21.819 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 125, smoothed-train-loss = 8.071962 | smoothed-test-loss = 8.093795 | rel_loss_delta = -0.00042785119781274517 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 125/300 [06:22<08:41,  2.98s/it]2026-10-14 00:33:24.838 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 126, smoothed-train-loss = 8.072033 | smoothed-test-loss = 8.092691 | rel_loss_delta = 8.836899582013386e-06 | loss_stagnation_count = 1

Epoch:  42%|████▏     | 126/300 [06:25<08:40,  2.99s/it]2026-10-14 00:33:27.827 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 127, smoothed-train-loss = 8.073104 | smoothed-test-loss = 8.097270 | rel_loss_delta = 0.00013262528724321924 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 127/300 [06:28<08:37,  2.99s/it]2026-10-14 00:33:30.837 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 128, smoothed-train-loss = 8.072424 | smoothed-test-loss = 8.090438 | rel_loss_delta = -8.417882530467365e-05 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 128/300 [06:31<08:35,  3.00s/it]2026-10-14 00:33:33.643 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 129, smoothed-train-loss = 8.073803 | smoothed-test-loss = 8.102513 | rel_loss_delta = 0.00017081599280859287 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 129/300 [06:34<08:22,  2.94s/it]2026-10-14 00:33:36.581 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 130, smoothed-train-loss = 8.068597 | smoothed-test-loss = 8.097883 | rel_loss_delta = -0.0006448042795004609 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 130/300 [06:37<08:19,  2.94s/it]2026-10-14 00:33:39.563 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 131, smoothed-train-loss = 8.069060 | smoothed-test-loss = 8.098034 | rel_loss_delta = 5.7316085622634394e-05 | loss_stagnation_count = 0

Epoch:  44%|████▎     | 131/300 [06:40<08:18,  2.95s/it]2026-10-14 00:33:42.603 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 132, smoothed-train-loss = 8.070830 | smoothed-test-loss = 8.115495 | rel_loss_delta = 0.00021942197183750177 | loss_stagnation_count = 0

Epoch:  44%|████▍     | 132/300 [06:43<08:20,  2.98s/it]2026-10-14 00:33:45.472 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 133, smoothed-train-loss = 8.073349 | smoothed-test-loss = 8.108148 | rel_loss_delta = 0.00031212036940716516 | loss_stagnation_count = 0

Epoch:  44%|████▍     | 133/300 [06:46<08:11,  2.95s/it]2026-10-14 00:33:48.569 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 134, smoothed-train-loss = 8.071205 | smoothed-test-loss = 8.095635 | rel_loss_delta = -0.00026567134963436203 | loss_stagnation_count = 0

Epoch:  45%|████▍     | 134/300 [06:49<08:16,  2.99s/it]2026-10-14 00:33:51.496 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 135, smoothed-train-loss = 8.068996 | smoothed-test-loss = 8.109648 | rel_loss_delta = -0.000273571746652084 | loss_stagnation_count = 0

Epoch:  45%|████▌     | 135/300 [06:52<08:10,  2.97s/it]2026-10-14 00:33:54.453 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 136, smoothed-train-loss = 8.072231 | smoothed-test-loss = 8.090413 | rel_loss_delta = 0.000400801441346356 | loss_stagnation_count = 0

Epoch:  45%|████▌     | 136/300 [06:55<08:06,  2.97s/it]2026-10-14 00:33:57.296 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 137, smoothed-train-loss = 8.072444 | smoothed-test-loss = 8.100275 | rel_loss_delta = 2.6493572008115128e-05 | loss_stagnation_count = 1

Epoch:  46%|████▌     | 137/300 [06:58<07:57,  2.93s/it]2026-10-14 00:34:00.287 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 138, smoothed-train-loss = 8.072927 | smoothed-test-loss = 8.092746 | rel_loss_delta = 5.975383072753017e-05 | loss_stagnation_count = 0

Epoch:  46%|████▌     | 138/300 [07:01<07:57,  2.95s/it]2026-10-14 00:34:03.235 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 139, smoothed-train-loss = 8.073225 | smoothed-test-loss = 8.104974 | rel_loss_delta = 3.699181902646387e-05 | loss_stagnation_count = 1

Epoch:  46%|████▋     | 139/300 [07:04<07:54,  2.95s/it]2026-10-14 00:34:06.161 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 140, smoothed-train-loss = 8.073016 | smoothed-test-loss = 8.098825 | rel_loss_delta = -2.5959191438232715e-05 | loss_stagnation_count = 2

Epoch:  47%|████▋     | 140/300 [07:07<07:50,  2.94s/it]2026-10-14 00:34:09.149 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 141, smoothed-train-loss = 8.069698 | smoothed-test-loss = 8.098625 | rel_loss_delta = -0.0004109616976162953 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 141/300 [07:10<07:49,  2.96s/it]2026-10-14 00:34:12.218 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 142, smoothed-train-loss = 8.071712 | smoothed-test-loss = 8.101151 | rel_loss_delta = 0.0002496042198072434 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 142/300 [07:13<07:52,  2.99s/it]2026-10-14 00:34:15.252 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 143, smoothed-train-loss = 8.071736 | smoothed-test-loss = 8.101414 | rel_loss_delta = 2.925235163013909e-06 | loss_stagnation_count = 1

Epoch:  48%|████▊     | 143/300 [07:16<07:51,  3.00s/it]2026-10-14 00:34:18.262 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 144, smoothed-train-loss = 8.071392 | smoothed-test-loss = 8.098831 | rel_loss_delta = -4.264431502138475e-05 | loss_stagnation_count = 2

Epoch:  48%|████▊     | 144/300 [07:19<07:48,  3.01s/it]2026-10-14 00:34:21.149 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 145, smoothed-train-loss = 8.071338 | smoothed-test-loss = 8.099902 | rel_loss_delta = -6.670646727158491e-06 | loss_stagnation_count = 3

Epoch:  48%|████▊     | 145/300 [07:22<07:40,  2.97s/it]2026-10-14 00:34:24.191 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 146, smoothed-train-loss = 8.070349 | smoothed-test-loss = 8.097506 | rel_loss_delta = -0.00012247033379988998 | loss_stagnation_count = 0

Epoch:  49%|████▊     | 146/300 [07:25<07:40,  2.99s/it]2026-10-14 00:34:27.140 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 147, smoothed-train-loss = 8.070180 | smoothed-test-loss = 8.095620 | rel_loss_delta = -2.0988398499916662e-05 | loss_stagnation_count = 1

Epoch:  49%|████▉     | 147/300 [07:28<07:35,  2.98s/it]2026-10-14 00:34:30.163 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 148, smoothed-train-loss = 8.069155 | smoothed-test-loss = 8.088149 | rel_loss_delta = -0.00012707436947298392 | loss_stagnation_count = 0

Epoch:  49%|████▉     | 148/300 [07:31<07:34,  2.99s/it]2026-10-14 00:34:33.081 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 149, smoothed-train-loss = 8.074162 | smoothed-test-loss = 8.099636 | rel_loss_delta = 0.0006205227287268263 | loss_stagnation_count = 0

Epoch:  50%|████▉     | 149/300 [07:34<07:28,  2.97s/it]2026-10-14 00:34:36.094 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 150, smoothed-train-loss = 8.071233 | smoothed-test-loss = 8.103871 | rel_loss_delta = -0.00036274802344313223 | loss_stagnation_count = 0

Epoch:  50%|█████     | 150/300 [07:37<07:27,  2.98s/it]2026-10-14 00:34:39.151 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 151, smoothed-train-loss = 8.069393 | smoothed-test-loss = 8.101090 | rel_loss_delta = -0.00022788371107424 | loss_stagnation_count = 0

Epoch:  50%|█████     | 151/300 [07:40<07:27,  3.01s/it]2026-10-14 00:34:42.234 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 152, smoothed-train-loss = 8.067416 | smoothed-test-loss = 8.100050 | rel_loss_delta = -0.00024501494247175385 | loss_stagnation_count = 0

Epoch:  51%|█████     | 152/300 [07:43<07:28,  3.03s/it]2026-10-14 00:34:45.088 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 153, smoothed-train-loss = 8.072494 | smoothed-test-loss = 8.106313 | rel_loss_delta = 0.0006294203250161939 | loss_stagnation_count = 0

Epoch:  51%|█████     | 153/300 [07:46<07:17,  2.98s/it]2026-10-14 00:34:47.982 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 154, smoothed-train-loss = 8.070612 | smoothed-test-loss = 8.104403 | rel_loss_delta = -0.00023319442588393835 | loss_stagnation_count = 0

Epoch:  51%|█████▏    | 154/300 [07:48<07:10,  2.95s/it]2026-10-14 00:34:50.958 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 155, smoothed-train-loss = 8.072721 | smoothed-test-loss = 8.091300 | rel_loss_delta = 0.0002613819449456065 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 155/300 [07:51<07:09,  2.96s/it]2026-10-14 00:34:54.013 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 156, smoothed-train-loss = 8.068010 | smoothed-test-loss = 8.095972 | rel_loss_delta = -0.000583570371933147 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 156/300 [07:54<07:10,  2.99s/it]2026-10-14 00:34:56.909 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 157, smoothed-train-loss = 8.068787 | smoothed-test-loss = 8.102544 | rel_loss_delta = 9.634557220494388e-05 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 157/300 [07:57<07:03,  2.96s/it]2026-10-14 00:34:59.899 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 158, smoothed-train-loss = 8.073236 | smoothed-test-loss = 8.099402 | rel_loss_delta = 0.0005513184066348958 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 158/300 [08:00<07:01,  2.97s/it]2026-10-14 00:35:02.984 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 159, smoothed-train-loss = 8.070016 | smoothed-test-loss = 8.094118 | rel_loss_delta = -0.00039883950591428926 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 159/300 [08:03<07:03,  3.00s/it]2026-10-14 00:35:06.023 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 160, smoothed-train-loss = 8.068351 | smoothed-test-loss = 8.092734 | rel_loss_delta = -0.00020626468765993125 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 160/300 [08:06<07:02,  3.01s/it]2026-10-14 00:35:08.879 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 161, smoothed-train-loss = 8.067343 | smoothed-test-loss = 8.112089 | rel_loss_delta = -0.00012502470610846532 | loss_stagnation_count = 0

Epoch:  54%|█████▎    | 161/300 [08:09<06:52,  2.97s/it]2026-10-14 00:35:11.879 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 162, smoothed-train-loss = 8.067588 | smoothed-test-loss = 8.114178 | rel_loss_delta = 3.0356503665249192e-05 | loss_stagnation_count = 1

Epoch:  54%|█████▍    | 162/300 [08:12<06:50,  2.98s/it]2026-10-14 00:35:14.964 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 163, smoothed-train-loss = 8.069273 | smoothed-test-loss = 8.099804 | rel_loss_delta = 0.00020895210286439397 | loss_stagnation_count = 0

Epoch:  54%|█████▍    | 163/300 [08:15<06:52,  3.01s/it]2026-10-14 00:35:18.085 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 164, smoothed-train-loss = 8.068450 | smoothed-test-loss = 8.102845 | rel_loss_delta = -0.00010201893501693731 | loss_stagnation_count = 0

Epoch:  55%|█████▍    | 164/300 [08:19<06:53,  3.04s/it]2026-10-14 00:35:21.020 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 165, smoothed-train-loss = 8.073144 | smoothed-test-loss = 8.092372 | rel_loss_delta = 0.000581734045783237 | loss_stagnation_count = 0

Epoch:  55%|█████▌    | 165/300 [08:21<06:46,  3.01s/it]2026-10-14 00:35:24.059 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 166, smoothed-train-loss = 8.069219 | smoothed-test-loss = 8.096238 | rel_loss_delta = -0.00048617116146057514 | loss_stagnation_count = 0

Epoch:  55%|█████▌    | 166/300 [08:24<06:44,  3.02s/it]2026-10-14 00:35:27.104 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 167, smoothed-train-loss = 8.071016 | smoothed-test-loss = 8.101754 | rel_loss_delta = 0.00022271211046135221 | loss_stagnation_count = 0

Epoch:  56%|█████▌    | 167/300 [08:28<06:42,  3.03s/it]2026-10-14 00:35:30.036 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 168, smoothed-train-loss = 8.069271 | smoothed-test-loss = 8.097050 | rel_loss_delta = -0.00021623666495271046 | loss_stagnation_count = 0

Epoch:  56%|█████▌    | 168/300 [08:30<06:35,  3.00s/it]2026-10-14 00:35:32.852 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 169, smoothed-train-loss = 8.066684 | smoothed-test-loss = 8.127355 | rel_loss_delta = -0.0003206014080299286 | loss_stagnation_count = 0

Epoch:  56%|█████▋    | 169/300 [08:33<06:25,  2.94s/it]2026-10-14 00:35:35.764 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 170, smoothed-train-loss = 8.069322 | smoothed-test-loss = 8.092968 | rel_loss_delta = 0.0003270026728406278 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 170/300 [08:36<06:21,  2.93s/it]2026-10-14 00:35:38.800 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 171, smoothed-train-loss = 8.070170 | smoothed-test-loss = 8.111444 | rel_loss_delta = 0.00010515696378779716 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 171/300 [08:39<06:22,  2.96s/it]2026-10-14 00:35:41.777 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 172, smoothed-train-loss = 8.068680 | smoothed-test-loss = 8.100319 | rel_loss_delta = -0.00018461575637735537 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 172/300 [08:42<06:19,  2.97s/it]2026-10-14 00:35:44.648 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 173, smoothed-train-loss = 8.066979 | smoothed-test-loss = 8.122849 | rel_loss_delta = -0.00021085395040487396 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 173/300 [08:45<06:13,  2.94s/it]2026-10-14 00:35:47.718 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 174, smoothed-train-loss = 8.070100 | smoothed-test-loss = 8.117947 | rel_loss_delta = 0.0003869369861091504 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 174/300 [08:48<06:15,  2.98s/it]2026-10-14 00:35:50.798 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 175, smoothed-train-loss = 8.069777 | smoothed-test-loss = 8.114084 | rel_loss_delta = -4.00233081015572e-05 | loss_stagnation_count = 1

Epoch:  58%|█████▊    | 175/300 [08:51<06:16,  3.01s/it]2026-10-14 00:35:53.722 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 176, smoothed-train-loss = 8.068719 | smoothed-test-loss = 8.102732 | rel_loss_delta = -0.00013111480957199366 | loss_stagnation_count = 0

Epoch:  59%|█████▊    | 176/300 [08:54<06:09,  2.98s/it]2026-10-14 00:35:56.656 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 177, smoothed-train-loss = 8.067255 | smoothed-test-loss = 8.103109 | rel_loss_delta = -0.00018143609475975683 | loss_stagnation_count = 0

Epoch:  59%|█████▉    | 177/300 [08:57<06:05,  2.97s/it]2026-10-14 00:35:59.677 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 178, smoothed-train-loss = 8.071035 | smoothed-test-loss = 8.088172 | rel_loss_delta = 0.0004685331689187154 | loss_stagnation_count = 0

Epoch:  59%|█████▉    | 178/300 [09:00<06:04,  2.98s/it]2026-10-14 00:36:02.665 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 179, smoothed-train-loss = 8.067498 | smoothed-test-loss = 8.107635 | rel_loss_delta = -0.00043829087702872344 | loss_stagnation_count = 0

Epoch:  60%|█████▉    | 179/300 [09:03<06:01,  2.99s/it]2026-10-14 00:36:05.689 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 180, smoothed-train-loss = 8.068179 | smoothed-test-loss = 8.097592 | rel_loss_delta = 8.449316934345639e-05 | loss_stagnation_count = 0

Epoch:  60%|██████    | 180/300 [09:06<05:59,  3.00s/it]2026-10-14 00:36:08.515 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 181, smoothed-train-loss = 8.067380 | smoothed-test-loss = 8.102463 | rel_loss_delta = -9.902234174626004e-05 | loss_stagnation_count = 0

Epoch:  60%|██████    | 181/300 [09:09<05:50,  2.95s/it]2026-10-14 00:36:11.472 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 182, smoothed-train-loss = 8.067048 | smoothed-test-loss = 8.105166 | rel_loss_delta = -4.1167452415921275e-05 | loss_stagnation_count = 1

Epoch:  61%|██████    | 182/300 [09:12<05:47,  2.95s/it]2026-10-14 00:36:14.395 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 183, smoothed-train-loss = 8.067881 | smoothed-test-loss = 8.097284 | rel_loss_delta = 0.00010327283382870531 | loss_stagnation_count = 0

Epoch:  61%|██████    | 183/300 [09:15<05:44,  2.94s/it]2026-10-14 00:36:17.428 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 184, smoothed-train-loss = 8.068453 | smoothed-test-loss = 8.094605 | rel_loss_delta = 7.091794357443526e-05 | loss_stagnation_count = 0

Epoch:  61%|██████▏   | 184/300 [09:18<05:44,  2.97s/it]2026-10-14 00:36:20.323 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 185, smoothed-train-loss = 8.068476 | smoothed-test-loss = 8.096714 | rel_loss_delta = 2.7980817300945043e-06 | loss_stagnation_count = 1

Epoch:  62%|██████▏   | 185/300 [09:21<05:38,  2.95s/it]2026-10-14 00:36:23.351 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 186, smoothed-train-loss = 8.069124 | smoothed-test-loss = 8.102656 | rel_loss_delta = 8.026003345626179e-05 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 186/300 [09:24<05:38,  2.97s/it]2026-10-14 00:36:26.373 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 187, smoothed-train-loss = 8.067561 | smoothed-test-loss = 8.104732 | rel_loss_delta = -0.00019363838161978833 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 187/300 [09:27<05:37,  2.99s/it]2026-10-14 00:36:29.479 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 188, smoothed-train-loss = 8.067925 | smoothed-test-loss = 8.091271 | rel_loss_delta = 4.515580552395475e-05 | loss_stagnation_count = 1

Epoch:  63%|██████▎   | 188/300 [09:30<05:38,  3.02s/it]2026-10-14 00:36:32.445 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 189, smoothed-train-loss = 8.066280 | smoothed-test-loss = 8.100982 | rel_loss_delta = -0.00020394776844360473 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 189/300 [09:33<05:33,  3.01s/it]2026-10-14 00:36:35.381 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 190, smoothed-train-loss = 8.070162 | smoothed-test-loss = 8.119222 | rel_loss_delta = 0.0004812236103780357 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 190/300 [09:36<05:28,  2.98s/it]2026-10-14 00:36:38.396 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 191, smoothed-train-loss = 8.068555 | smoothed-test-loss = 8.105565 | rel_loss_delta = -0.0001990514930953435 | loss_stagnation_count = 0

Epoch:  64%|██████▎   | 191/300 [09:39<05:26,  2.99s/it]2026-10-14 00:36:41.502 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 192, smoothed-train-loss = 8.067282 | smoothed-test-loss = 8.113513 | rel_loss_delta = -0.0001577664091404586 | loss_stagnation_count = 0

Epoch:  64%|██████▍   | 192/300 [09:42<05:26,  3.03s/it]2026-10-14 00:36:44.378 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 193, smoothed-train-loss = 8.068221 | smoothed-test-loss = 8.099040 | rel_loss_delta = 0.00011631799616221748 | loss_stagnation_count = 0

Epoch:  64%|██████▍   | 193/300 [09:45<05:19,  2.98s/it]2026-10-14 00:36:48.124 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 194, smoothed-train-loss = 8.068250 | smoothed-test-loss = 8.101463 | rel_loss_delta = 3.6682422980236883e-06 | loss_stagnation_count = 1

Epoch:  65%|██████▍   | 194/300 [09:49<05:40,  3.21s/it]2026-10-14 00:36:52.906 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 195, smoothed-train-loss = 8.064984 | smoothed-test-loss = 8.099761 | rel_loss_delta = -0.00040484971694259794 | loss_stagnation_count = 0

Epoch:  65%|██████▌   | 195/300 [09:53<06:26,  3.68s/it]2026-10-14 00:36:56.077 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 196, smoothed-train-loss = 8.065918 | smoothed-test-loss = 8.100902 | rel_loss_delta = 0.0001158039808656196 | loss_stagnation_count = 0

Epoch:  65%|██████▌   | 196/300 [09:56<06:07,  3.53s/it]2026-10-14 00:36:58.926 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 197, smoothed-train-loss = 8.069008 | smoothed-test-loss = 8.107038 | rel_loss_delta = 0.00038317036908120945 | loss_stagnation_count = 0

Epoch:  66%|██████▌   | 197/300 [09:59<05:42,  3.32s/it]2026-10-14 00:37:01.887 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 198, smoothed-train-loss = 8.069734 | smoothed-test-loss = 8.111731 | rel_loss_delta = 8.991520173659159e-05 | loss_stagnation_count = 0

Epoch:  66%|██████▌   | 198/300 [10:02<05:28,  3.22s/it]2026-10-14 00:37:04.848 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 199, smoothed-train-loss = 8.069075 | smoothed-test-loss = 8.095304 | rel_loss_delta = -8.170293229845291e-05 | loss_stagnation_count = 0

Epoch:  66%|██████▋   | 199/300 [10:05<05:17,  3.14s/it]2026-10-14 00:37:07.833 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 200, smoothed-train-loss = 8.068392 | smoothed-test-loss = 8.094620 | rel_loss_delta = -8.462488869404096e-05 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 200/300 [10:08<05:09,  3.09s/it]2026-10-14 00:37:10.835 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 201, smoothed-train-loss = 8.066021 | smoothed-test-loss = 8.096703 | rel_loss_delta = -0.00029388357744378896 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 201/300 [10:11<05:03,  3.07s/it]2026-10-14 00:37:13.915 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 202, smoothed-train-loss = 8.068734 | smoothed-test-loss = 8.093670 | rel_loss_delta = 0.00033636755785174853 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 202/300 [10:14<05:00,  3.07s/it]2026-10-14 00:37:16.845 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 203, smoothed-train-loss = 8.067454 | smoothed-test-loss = 8.095338 | rel_loss_delta = -0.00015865804311388117 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 203/300 [10:17<04:53,  3.03s/it]2026-10-14 00:37:19.911 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 204, smoothed-train-loss = 8.067816 | smoothed-test-loss = 8.113670 | rel_loss_delta = 4.497065968277286e-05 | loss_stagnation_count = 1

Epoch:  68%|██████▊   | 204/300 [10:20<04:51,  3.04s/it]2026-10-14 00:37:23.000 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 205, smoothed-train-loss = 8.066661 | smoothed-test-loss = 8.108754 | rel_loss_delta = -0.00014318184402529235 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 205/300 [10:23<04:50,  3.05s/it]2026-10-14 00:37:26.121 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 206, smoothed-train-loss = 8.069474 | smoothed-test-loss = 8.092277 | rel_loss_delta = 0.00034869401670977937 | loss_stagnation_count = 0

Epoch:  69%|██████▊   | 206/300 [10:27<04:48,  3.07s/it]2026-10-14 00:37:29.079 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 207, smoothed-train-loss = 8.066024 | smoothed-test-loss = 8.094522 | rel_loss_delta = -0.000427521763575821 | loss_stagnation_count = 0

Epoch:  69%|██████▉   | 207/300 [10:29<04:42,  3.04s/it]2026-10-14 00:37:31.978 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 208, smoothed-train-loss = 8.065904 | smoothed-test-loss = 8.102689 | rel_loss_delta = -1.4932574270666585e-05 | loss_stagnation_count = 1

Epoch:  69%|██████▉   | 208/300 [10:32<04:35,  3.00s/it]2026-10-14 00:37:34.877 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 209, smoothed-train-loss = 8.067814 | smoothed-test-loss = 8.089297 | rel_loss_delta = 0.00023680307156536345 | loss_stagnation_count = 0

Epoch:  70%|██████▉   | 209/300 [10:35<04:30,  2.97s/it]2026-10-14 00:37:37.954 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 210, smoothed-train-loss = 8.067613 | smoothed-test-loss = 8.096497 | rel_loss_delta = -2.489481088547805e-05 | loss_stagnation_count = 1

Epoch:  70%|███████   | 210/300 [10:38<04:30,  3.00s/it]2026-10-14 00:37:41.010 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 211, smoothed-train-loss = 8.067092 | smoothed-test-loss = 8.098770 | rel_loss_delta = -6.452928990557091e-05 | loss_stagnation_count = 0

Epoch:  70%|███████   | 211/300 [10:41<04:28,  3.02s/it]2026-10-14 00:37:44.131 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 212, smoothed-train-loss = 8.067068 | smoothed-test-loss = 8.097915 | rel_loss_delta = -3.033023447125799e-06 | loss_stagnation_count = 1

Epoch:  71%|███████   | 212/300 [10:45<04:28,  3.05s/it]2026-10-14 00:37:46.992 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 213, smoothed-train-loss = 8.067520 | smoothed-test-loss = 8.100893 | rel_loss_delta = 5.608229503062256e-05 | loss_stagnation_count = 0

Epoch:  71%|███████   | 213/300 [10:47<04:20,  2.99s/it]2026-10-14 00:37:49.981 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 214, smoothed-train-loss = 8.067310 | smoothed-test-loss = 8.093484 | rel_loss_delta = -2.6096910623259468e-05 | loss_stagnation_count = 1

Epoch:  71%|███████▏  | 214/300 [10:50<04:17,  2.99s/it]2026-10-14 00:37:52.912 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 215, smoothed-train-loss = 8.067030 | smoothed-test-loss = 8.102314 | rel_loss_delta = -3.464974900434084e-05 | loss_stagnation_count = 2

Epoch:  72%|███████▏  | 215/300 [10:53<04:12,  2.97s/it]2026-10-14 00:37:55.808 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 216, smoothed-train-loss = 8.069436 | smoothed-test-loss = 8.111981 | rel_loss_delta = 0.00029821850131968503 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 216/300 [10:56<04:07,  2.95s/it]2026-10-14 00:37:58.693 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 217, smoothed-train-loss = 8.069670 | smoothed-test-loss = 8.092627 | rel_loss_delta = 2.896833485734671e-05 | loss_stagnation_count = 1

Epoch:  72%|███████▏  | 217/300 [10:59<04:03,  2.93s/it]2026-10-14 00:38:01.745 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 218, smoothed-train-loss = 8.066716 | smoothed-test-loss = 8.097715 | rel_loss_delta = -0.00036602136540620596 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 218/300 [11:02<04:03,  2.97s/it]2026-10-14 00:38:04.744 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 219, smoothed-train-loss = 8.067443 | smoothed-test-loss = 8.097648 | rel_loss_delta = 9.014387107088228e-05 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 219/300 [11:05<04:01,  2.98s/it]2026-10-14 00:38:07.731 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 220, smoothed-train-loss = 8.068589 | smoothed-test-loss = 8.107236 | rel_loss_delta = 0.00014200150336495505 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 220/300 [11:08<03:58,  2.98s/it]2026-10-14 00:38:10.686 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 221, smoothed-train-loss = 8.065773 | smoothed-test-loss = 8.089008 | rel_loss_delta = -0.00034893593971289065 | loss_stagnation_count = 0

Epoch:  74%|███████▎  | 221/300 [11:11<03:54,  2.97s/it]2026-10-14 00:38:13.749 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 222, smoothed-train-loss = 8.066491 | smoothed-test-loss = 8.095930 | rel_loss_delta = 8.899546406051834e-05 | loss_stagnation_count = 0

Epoch:  74%|███████▍  | 222/300 [11:14<03:53,  3.00s/it]2026-10-14 00:38:16.772 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 223, smoothed-train-loss = 8.065527 | smoothed-test-loss = 8.104191 | rel_loss_delta = -0.00011956303304798552 | loss_stagnation_count = 0

Epoch:  74%|███████▍  | 223/300 [11:17<03:51,  3.01s/it]2026-10-14 00:38:19.770 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 224, smoothed-train-loss = 8.063771 | smoothed-test-loss = 8.089838 | rel_loss_delta = -0.00021769417816640382 | loss_stagnation_count = 0

Epoch:  75%|███████▍  | 224/300 [11:20<03:48,  3.00s/it]2026-10-14 00:38:22.617 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 225, smoothed-train-loss = 8.065002 | smoothed-test-loss = 8.094786 | rel_loss_delta = 0.00015266217557351696 | loss_stagnation_count = 0

Epoch:  75%|███████▌  | 225/300 [11:23<03:41,  2.96s/it]2026-10-14 00:38:25.683 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 226, smoothed-train-loss = 8.066441 | smoothed-test-loss = 8.092594 | rel_loss_delta = 0.00017843447175946794 | loss_stagnation_count = 0

Epoch:  75%|███████▌  | 226/300 [11:26<03:41,  2.99s/it]2026-10-14 00:38:28.792 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 227, smoothed-train-loss = 8.064562 | smoothed-test-loss = 8.098868 | rel_loss_delta = -0.00023298494283793353 | loss_stagnation_count = 0

Epoch:  76%|███████▌  | 227/300 [11:29<03:40,  3.03s/it]2026-10-14 00:38:31.847 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 228, smoothed-train-loss = 8.066397 | smoothed-test-loss = 8.098930 | rel_loss_delta = 0.00022752286938128622 | loss_stagnation_count = 0

Epoch:  76%|███████▌  | 228/300 [11:32<03:38,  3.03s/it]2026-10-14 00:38:34.773 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 229, smoothed-train-loss = 8.066407 | smoothed-test-loss = 8.099619 | rel_loss_delta = 1.3164485890484829e-06 | loss_stagnation_count = 1

Epoch:  76%|███████▋  | 229/300 [11:35<03:33,  3.00s/it]2026-10-14 00:38:37.701 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 230, smoothed-train-loss = 8.065252 | smoothed-test-loss = 8.107915 | rel_loss_delta = -0.00014321249520194036 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 230/300 [11:38<03:28,  2.98s/it]2026-10-14 00:38:40.667 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 231, smoothed-train-loss = 8.065773 | smoothed-test-loss = 8.102715 | rel_loss_delta = 6.461199512800651e-05 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 231/300 [11:41<03:25,  2.98s/it]2026-10-14 00:38:43.805 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 232, smoothed-train-loss = 8.061943 | smoothed-test-loss = 8.101862 | rel_loss_delta = -0.00047486230284783963 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 232/300 [11:44<03:25,  3.02s/it]2026-10-14 00:38:46.675 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 233, smoothed-train-loss = 8.066770 | smoothed-test-loss = 8.111935 | rel_loss_delta = 0.0005987451231846414 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 233/300 [11:47<03:19,  2.98s/it]2026-10-14 00:38:49.627 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 234, smoothed-train-loss = 8.066863 | smoothed-test-loss = 8.105075 | rel_loss_delta = 1.1539269810118255e-05 | loss_stagnation_count = 1

Epoch:  78%|███████▊  | 234/300 [11:50<03:16,  2.97s/it]2026-10-14 00:38:52.539 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 235, smoothed-train-loss = 8.064437 | smoothed-test-loss = 8.098539 | rel_loss_delta = -0.00030079092394339683 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 235/300 [11:53<03:11,  2.95s/it]2026-10-14 00:38:55.466 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 236, smoothed-train-loss = 8.067250 | smoothed-test-loss = 8.095753 | rel_loss_delta = 0.0003488649381189819 | loss_stagnation_count = 0

Epoch:  79%|███████▊  | 236/300 [11:56<03:08,  2.95s/it]2026-10-14 00:38:58.380 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 237, smoothed-train-loss = 8.066119 | smoothed-test-loss = 8.087256 | rel_loss_delta = -0.0001402461895868336 | loss_stagnation_count = 0

Epoch:  79%|███████▉  | 237/300 [11:59<03:04,  2.94s/it]2026-10-14 00:39:01.330 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 238, smoothed-train-loss = 8.066836 | smoothed-test-loss = 8.102298 | rel_loss_delta = 8.89796077962249e-05 | loss_stagnation_count = 0

Epoch:  79%|███████▉  | 238/300 [12:02<03:02,  2.94s/it]2026-10-14 00:39:04.231 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 239, smoothed-train-loss = 8.065773 | smoothed-test-loss = 8.101517 | rel_loss_delta = -0.00013182051194395662 | loss_stagnation_count = 0

Epoch:  80%|███████▉  | 239/300 [12:05<02:58,  2.93s/it]2026-10-14 00:39:07.187 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 240, smoothed-train-loss = 8.065703 | smoothed-test-loss = 8.097420 | rel_loss_delta = -8.6630554728773e-06 | loss_stagnation_count = 1

Epoch:  80%|████████  | 240/300 [12:08<02:56,  2.94s/it]2026-10-14 00:39:10.073 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 241, smoothed-train-loss = 8.064473 | smoothed-test-loss = 8.111756 | rel_loss_delta = -0.0001525469819448792 | loss_stagnation_count = 0

Epoch:  80%|████████  | 241/300 [12:10<02:52,  2.92s/it]2026-10-14 00:39:13.325 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 242, smoothed-train-loss = 8.063954 | smoothed-test-loss = 8.096197 | rel_loss_delta = -6.429615784280131e-05 | loss_stagnation_count = 0

Epoch:  81%|████████  | 242/300 [12:14<02:55,  3.02s/it]2026-10-14 00:39:16.397 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 243, smoothed-train-loss = 8.067128 | smoothed-test-loss = 8.095181 | rel_loss_delta = 0.0003935902333230296 | loss_stagnation_count = 0

Epoch:  81%|████████  | 243/300 [12:17<02:53,  3.04s/it]2026-10-14 00:39:19.372 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 244, smoothed-train-loss = 8.067202 | smoothed-test-loss = 8.097450 | rel_loss_delta = 9.208982938370267e-06 | loss_stagnation_count = 1

Epoch:  81%|████████▏ | 244/300 [12:20<02:48,  3.02s/it]2026-10-14 00:39:22.261 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 245, smoothed-train-loss = 8.064972 | smoothed-test-loss = 8.094637 | rel_loss_delta = -0.00027649043892636255 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 245/300 [12:23<02:43,  2.98s/it]2026-10-14 00:39:25.232 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 246, smoothed-train-loss = 8.065678 | smoothed-test-loss = 8.096627 | rel_loss_delta = 8.750688402347421e-05 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 246/300 [12:26<02:40,  2.98s/it]2026-10-14 00:39:28.280 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 247, smoothed-train-loss = 8.065327 | smoothed-test-loss = 8.104490 | rel_loss_delta = -4.35186519534324e-05 | loss_stagnation_count = 1

Epoch:  82%|████████▏ | 247/300 [12:29<02:38,  3.00s/it]2026-10-14 00:39:31.273 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 248, smoothed-train-loss = 8.064764 | smoothed-test-loss = 8.105844 | rel_loss_delta = -6.974269801753621e-05 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 248/300 [12:32<02:35,  3.00s/it]2026-10-14 00:39:34.171 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 249, smoothed-train-loss = 8.064159 | smoothed-test-loss = 8.100742 | rel_loss_delta = -7.50106047695891e-05 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 249/300 [12:35<02:31,  2.97s/it]2026-10-14 00:39:37.199 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 250, smoothed-train-loss = 8.066811 | smoothed-test-loss = 8.111271 | rel_loss_delta = 0.0003288186884444421 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 250/300 [12:38<02:29,  2.99s/it]2026-10-14 00:39:40.153 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 251, smoothed-train-loss = 8.066663 | smoothed-test-loss = 8.094061 | rel_loss_delta = -1.8329745780301535e-05 | loss_stagnation_count = 1

Epoch:  84%|████████▎ | 251/300 [12:41<02:25,  2.98s/it]2026-10-14 00:39:43.160 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 252, smoothed-train-loss = 8.066190 | smoothed-test-loss = 8.104015 | rel_loss_delta = -5.860637964128574e-05 | loss_stagnation_count = 0

Epoch:  84%|████████▍ | 252/300 [12:44<02:23,  2.99s/it]2026-10-14 00:39:46.084 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 253, smoothed-train-loss = 8.066786 | smoothed-test-loss = 8.086775 | rel_loss_delta = 7.383557775646176e-05 | loss_stagnation_count = 0

Epoch:  84%|████████▍ | 253/300 [12:47<02:19,  2.97s/it]2026-10-14 00:39:49.207 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 254, smoothed-train-loss = 8.066489 | smoothed-test-loss = 8.089410 | rel_loss_delta = -3.679876950057079e-05 | loss_stagnation_count = 1

Epoch:  85%|████████▍ | 254/300 [12:50<02:18,  3.01s/it]2026-10-14 00:39:52.431 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 255, smoothed-train-loss = 8.064907 | smoothed-test-loss = 8.089791 | rel_loss_delta = -0.00019616913531060072 | loss_stagnation_count = 0

Epoch:  85%|████████▌ | 255/300 [12:53<02:18,  3.08s/it]2026-10-14 00:39:55.401 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 256, smoothed-train-loss = 8.063385 | smoothed-test-loss = 8.102422 | rel_loss_delta = -0.00018866520815018185 | loss_stagnation_count = 0

Epoch:  85%|████████▌ | 256/300 [12:56<02:13,  3.04s/it]2026-10-14 00:39:58.401 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 257, smoothed-train-loss = 8.064733 | smoothed-test-loss = 8.105199 | rel_loss_delta = 0.0001672201947705539 | loss_stagnation_count = 0

Epoch:  86%|████████▌ | 257/300 [12:59<02:10,  3.03s/it]2026-10-14 00:40:01.398 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 258, smoothed-train-loss = 8.065119 | smoothed-test-loss = 8.107825 | rel_loss_delta = 4.783626513830264e-05 | loss_stagnation_count = 1

Epoch:  86%|████████▌ | 258/300 [13:02<02:06,  3.02s/it]2026-10-14 00:40:04.481 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 259, smoothed-train-loss = 8.065526 | smoothed-test-loss = 8.103310 | rel_loss_delta = 5.042251896288831e-05 | loss_stagnation_count = 0

Epoch:  86%|████████▋ | 259/300 [13:05<02:04,  3.04s/it]2026-10-14 00:40:07.437 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 260, smoothed-train-loss = 8.065788 | smoothed-test-loss = 8.100355 | rel_loss_delta = 3.2574692754156045e-05 | loss_stagnation_count = 1

Epoch:  87%|████████▋ | 260/300 [13:08<02:00,  3.01s/it]2026-10-14 00:40:10.453 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 261, smoothed-train-loss = 8.064019 | smoothed-test-loss = 8.090721 | rel_loss_delta = -0.00021932380588402133 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 261/300 [13:11<01:57,  3.01s/it]2026-10-14 00:40:13.374 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 262, smoothed-train-loss = 8.063566 | smoothed-test-loss = 8.095141 | rel_loss_delta = -5.62605935889771e-05 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 262/300 [13:14<01:53,  2.99s/it]2026-10-14 00:40:16.398 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 263, smoothed-train-loss = 8.065966 | smoothed-test-loss = 8.094424 | rel_loss_delta = 0.0002977096397158807 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 263/300 [13:17<01:50,  3.00s/it]2026-10-14 00:40:19.580 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 264, smoothed-train-loss = 8.064001 | smoothed-test-loss = 8.090513 | rel_loss_delta = -0.00024372545378598096 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 264/300 [13:20<01:49,  3.05s/it]2026-10-14 00:40:22.486 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 265, smoothed-train-loss = 8.067209 | smoothed-test-loss = 8.104558 | rel_loss_delta = 0.0003979247813225065 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 265/300 [13:23<01:45,  3.01s/it]2026-10-14 00:40:25.447 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 266, smoothed-train-loss = 8.066261 | smoothed-test-loss = 8.096429 | rel_loss_delta = -0.00011757217630026524 | loss_stagnation_count = 0

Epoch:  89%|████████▊ | 266/300 [13:26<01:41,  2.99s/it]2026-10-14 00:40:28.429 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 267, smoothed-train-loss = 8.065703 | smoothed-test-loss = 8.084653 | rel_loss_delta = -6.919565204915334e-05 | loss_stagnation_count = 0

Epoch:  89%|████████▉ | 267/300 [13:29<01:38,  2.99s/it]2026-10-14 00:40:31.494 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 268, smoothed-train-loss = 8.064076 | smoothed-test-loss = 8.090560 | rel_loss_delta = -0.0002017332224094727 | loss_stagnation_count = 0

Epoch:  89%|████████▉ | 268/300 [13:32<01:36,  3.01s/it]2026-10-14 00:40:34.356 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 269, smoothed-train-loss = 8.063898 | smoothed-test-loss = 8.097284 | rel_loss_delta = -2.2049648986532997e-05 | loss_stagnation_count = 1

Epoch:  90%|████████▉ | 269/300 [13:35<01:32,  2.97s/it]2026-10-14 00:40:37.295 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 270, smoothed-train-loss = 8.066120 | smoothed-test-loss = 8.089873 | rel_loss_delta = 0.00027561745238034616 | loss_stagnation_count = 0

Epoch:  90%|█████████ | 270/300 [13:38<01:28,  2.96s/it]2026-10-14 00:40:40.283 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 271, smoothed-train-loss = 8.063815 | smoothed-test-loss = 8.100544 | rel_loss_delta = -0.00028583055111149145 | loss_stagnation_count = 0

Epoch:  90%|█████████ | 271/300 [13:41<01:26,  2.97s/it]2026-10-14 00:40:43.317 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 272, smoothed-train-loss = 8.067049 | smoothed-test-loss = 8.087978 | rel_loss_delta = 0.0004011166271351881 | loss_stagnation_count = 0

Epoch:  91%|█████████ | 272/300 [13:44<01:23,  2.99s/it]2026-10-14 00:40:46.284 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 273, smoothed-train-loss = 8.065068 | smoothed-test-loss = 8.091195 | rel_loss_delta = -0.00024565854759976664 | loss_stagnation_count = 0

Epoch:  91%|█████████ | 273/300 [13:47<01:20,  2.98s/it]2026-10-14 00:40:49.204 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 274, smoothed-train-loss = 8.064487 | smoothed-test-loss = 8.090317 | rel_loss_delta = -7.205318054396585e-05 | loss_stagnation_count = 0

Epoch:  91%|█████████▏| 274/300 [13:50<01:17,  2.96s/it]2026-10-14 00:40:52.173 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 275, smoothed-train-loss = 8.065090 | smoothed-test-loss = 8.095196 | rel_loss_delta = 7.489500286810184e-05 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 275/300 [13:53<01:14,  2.96s/it]2026-10-14 00:40:55.189 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 276, smoothed-train-loss = 8.064142 | smoothed-test-loss = 8.087820 | rel_loss_delta = -0.0001176137201570339 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 276/300 [13:56<01:11,  2.98s/it]2026-10-14 00:40:58.052 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 277, smoothed-train-loss = 8.063672 | smoothed-test-loss = 8.092536 | rel_loss_delta = -5.8269459000984547e-05 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 277/300 [13:58<01:07,  2.95s/it]2026-10-14 00:41:01.101 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 278, smoothed-train-loss = 8.063335 | smoothed-test-loss = 8.099038 | rel_loss_delta = -4.178139752458699e-05 | loss_stagnation_count = 1

Epoch:  93%|█████████▎| 278/300 [14:02<01:05,  2.98s/it]2026-10-14 00:41:04.106 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 279, smoothed-train-loss = 8.064892 | smoothed-test-loss = 8.090713 | rel_loss_delta = 0.00019310718725195508 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 279/300 [14:05<01:02,  2.98s/it]2026-10-14 00:41:07.224 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 280, smoothed-train-loss = 8.064236 | smoothed-test-loss = 8.096579 | rel_loss_delta = -8.13472755171628e-05 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 280/300 [14:08<01:00,  3.02s/it]2026-10-14 00:41:10.137 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 281, smoothed-train-loss = 8.062222 | smoothed-test-loss = 8.110799 | rel_loss_delta = -0.0002497124489272787 | loss_stagnation_count = 0

Epoch:  94%|█████████▎| 281/300 [14:11<00:56,  2.99s/it]2026-10-14 00:41:13.137 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 282, smoothed-train-loss = 8.065482 | smoothed-test-loss = 8.086893 | rel_loss_delta = 0.00040433758859944615 | loss_stagnation_count = 0

Epoch:  94%|█████████▍| 282/300 [14:14<00:53,  2.99s/it]2026-10-14 00:41:16.272 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 283, smoothed-train-loss = 8.063695 | smoothed-test-loss = 8.098830 | rel_loss_delta = -0.00022165368741725993 | loss_stagnation_count = 0

Epoch:  94%|█████████▍| 283/300 [14:17<00:51,  3.04s/it]2026-10-14 00:41:19.210 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 284, smoothed-train-loss = 8.064086 | smoothed-test-loss = 8.096626 | rel_loss_delta = 4.8602491450196774e-05 | loss_stagnation_count = 1

Epoch:  95%|█████████▍| 284/300 [14:20<00:48,  3.01s/it]2026-10-14 00:41:22.022 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 285, smoothed-train-loss = 8.063374 | smoothed-test-loss = 8.089220 | rel_loss_delta = -8.828703954984706e-05 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 285/300 [14:22<00:44,  2.95s/it]2026-10-14 00:41:25.170 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 286, smoothed-train-loss = 8.064157 | smoothed-test-loss = 8.097222 | rel_loss_delta = 9.708498703044972e-05 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 286/300 [14:26<00:42,  3.01s/it]2026-10-14 00:41:28.327 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 287, smoothed-train-loss = 8.062828 | smoothed-test-loss = 8.090046 | rel_loss_delta = -0.00016487707089472457 | loss_stagnation_count = 0

Epoch:  96%|█████████▌| 287/300 [14:29<00:39,  3.05s/it]2026-10-14 00:41:31.285 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 288, smoothed-train-loss = 8.063834 | smoothed-test-loss = 8.094701 | rel_loss_delta = 0.00012474568552704577 | loss_stagnation_count = 0

Epoch:  96%|█████████▌| 288/300 [14:32<00:36,  3.02s/it]2026-10-14 00:41:34.198 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 289, smoothed-train-loss = 8.064454 | smoothed-test-loss = 8.096167 | rel_loss_delta = 7.693961074025748e-05 | loss_stagnation_count = 0

Epoch:  96%|█████████▋| 289/300 [14:35<00:32,  2.99s/it]2026-10-14 00:41:37.149 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 290, smoothed-train-loss = 8.065615 | smoothed-test-loss = 8.100814 | rel_loss_delta = 0.0001439690101188092 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 290/300 [14:38<00:29,  2.98s/it]2026-10-14 00:41:40.219 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 291, smoothed-train-loss = 8.061499 | smoothed-test-loss = 8.093810 | rel_loss_delta = -0.0005103121149297909 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 291/300 [14:41<00:27,  3.01s/it]2026-10-14 00:41:43.313 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 292, smoothed-train-loss = 8.062999 | smoothed-test-loss = 8.111814 | rel_loss_delta = 0.00018611449386638903 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 292/300 [14:44<00:24,  3.03s/it]2026-10-14 00:41:46.227 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 293, smoothed-train-loss = 8.064344 | smoothed-test-loss = 8.093455 | rel_loss_delta = 0.00016670360207689244 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 293/300 [14:47<00:20,  3.00s/it]2026-10-14 00:41:49.172 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 294, smoothed-train-loss = 8.063606 | smoothed-test-loss = 8.095998 | rel_loss_delta = -9.141459791925364e-05 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 294/300 [14:50<00:17,  2.98s/it]2026-10-14 00:41:52.168 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 295, smoothed-train-loss = 8.062030 | smoothed-test-loss = 8.093547 | rel_loss_delta = -0.0001954292567234763 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 295/300 [14:53<00:14,  2.99s/it]2026-10-14 00:41:55.266 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 296, smoothed-train-loss = 8.065121 | smoothed-test-loss = 8.099141 | rel_loss_delta = 0.00038336508982968355 | loss_stagnation_count = 0

Epoch:  99%|█████████▊| 296/300 [14:56<00:12,  3.02s/it]2026-10-14 00:41:58.117 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 297, smoothed-train-loss = 8.063270 | smoothed-test-loss = 8.106063 | rel_loss_delta = -0.00022957357312905843 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 297/300 [14:59<00:08,  2.97s/it]2026-10-14 00:42:01.055 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 298, smoothed-train-loss = 8.064345 | smoothed-test-loss = 8.090376 | rel_loss_delta = 0.00013338180075011773 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 298/300 [15:01<00:05,  2.96s/it]2026-10-14 00:42:04.170 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 299, smoothed-train-loss = 8.064706 | smoothed-test-loss = 8.095027 | rel_loss_delta = 4.479147454182952e-05 | loss_stagnation_count = 1

Epoch: 100%|█████████▉| 299/300 [15:05<00:03,  3.01s/it]2026-10-14 00:42:07.115 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 300, smoothed-train-loss = 8.064010 | smoothed-test-loss = 8.093581 | rel_loss_delta = -8.633926257441672e-05 | loss_stagnation_count = 0

Epoch: 100%|██████████| 300/300 [15:08<00:00,  3.03s/it]
2026-10-14 00:42:07.120 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:8.093420028686523
2026-10-14 00:42:07.121 | INFO     | __main__:<module>:289 - Reflow iteration # 0 : Generating z0_train,z1_train for reflow iteration # 1

inference-batches:   0%|          | 0/123 [00:00<?, ?it/s]

Solving ODE | Method = euler:   0%|          | 0/1.0 [00:00<?, ?it/s]

Solving ODE | Method = euler:  25%|██▌       | 0.2522522509098053/1.0 [00:00<00:00,  2.52it/s]

Solving ODE | Method = euler:  64%|██████▍   | 0.6446446776390076/1.0 [00:00<00:00,  3.34it/s]
inference-batches:   1%|          | 1/123 [00:00<00:35,  3.41it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  3.40it/s]



Solving ODE | Method = euler:  41%|████▏     | 0.4144144058227539/1.0 [00:00<00:00,  4.14it/s]


Solving ODE | Method = euler:  85%|████████▌ | 0.8528528809547424/1.0 [00:00<00:00,  4.28it/s]
inference-batches:   2%|▏         | 2/123 [00:00<00:31,  3.87it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.27it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.4464464485645294/1.0 [00:00<00:00,  4.46it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9089089035987854/1.0 [00:00<00:00,  4.56it/s]
inference-batches:   2%|▏         | 3/123 [00:00<00:28,  4.14it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]


Solving ODE | Method = euler:  91%|█████████▏| 0.913913905620575/1.0 [00:00<00:00,  4.58it/s]  
inference-batches:   3%|▎         | 4/123 [00:00<00:27,  4.30it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.56it/s]


Solving ODE | Method = euler:  42%|████▏     | 0.4154154062271118/1.0 [00:00<00:00,  4.15it/s]

Solving ODE | Method = euler:  88%|████████▊ | 0.8758758902549744/1.0 [00:00<00:00,  4.42it/s]
inference-batches:   4%|▍         | 5/123 [00:01<00:27,  4.33it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.37it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8968968987464905/1.0 [00:00<00:00,  4.48it/s] 
inference-batches:   5%|▍         | 6/123 [00:01<00:26,  4.37it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.45it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.44044044613838196/1.0 [00:00<00:00,  4.40it/s]

Solving ODE | Method = euler:  89%|████████▉ | 0.890890896320343/1.0 [00:00<00:00,  4.46it/s]  
inference-batches:   6%|▌         | 7/123 [00:01<00:26,  4.38it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.41it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.4454454481601715/1.0 [00:00<00:00,  4.45it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.53it/s]
inference-batches:   7%|▋         | 8/123 [00:01<00:26,  4.42it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]

Solving ODE | Method = euler:  91%|█████████▏| 0.912912905216217/1.0 [00:00<00:00,  4.58it/s]  
inference-batches:   7%|▋         | 9/123 [00:02<00:25,  4.46it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.54it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9029029011726379/1.0 [00:00<00:00,  4.52it/s]
inference-batches:   8%|▊         | 10/123 [00:02<00:25,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.44044044613838196/1.0 [00:00<00:00,  4.40it/s]

Solving ODE | Method = euler:  89%|████████▉ | 0.8878878951072693/1.0 [00:00<00:00,  4.44it/s] 
inference-batches:   9%|▉         | 11/123 [00:02<00:25,  4.45it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.38it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.4464464485645294/1.0 [00:00<00:00,  4.46it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9059059023857117/1.0 [00:00<00:00,  4.53it/s]
inference-batches:  10%|▉         | 12/123 [00:02<00:24,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.43843844532966614/1.0 [00:00<00:00,  4.38it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8968968987464905/1.0 [00:00<00:00,  4.50it/s] 
inference-batches:  11%|█         | 13/123 [00:02<00:24,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.46it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.43843844532966614/1.0 [00:00<00:00,  4.37it/s]


Solving ODE | Method = euler:  89%|████████▉ | 0.8928928971290588/1.0 [00:00<00:00,  4.47it/s] 
inference-batches:  11%|█▏        | 14/123 [00:03<00:24,  4.46it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.45it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.52it/s] 
inference-batches:  12%|█▏        | 15/123 [00:03<00:24,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8978978991508484/1.0 [00:00<00:00,  4.49it/s]
inference-batches:  13%|█▎        | 16/123 [00:03<00:23,  4.48it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.4454454481601715/1.0 [00:00<00:00,  4.45it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8978978991508484/1.0 [00:00<00:00,  4.49it/s]
inference-batches:  14%|█▍        | 17/123 [00:03<00:23,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.4454454481601715/1.0 [00:00<00:00,  4.45it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9029029011726379/1.0 [00:00<00:00,  4.52it/s]
inference-batches:  15%|█▍        | 18/123 [00:04<00:23,  4.48it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.4434434473514557/1.0 [00:00<00:00,  4.43it/s]

Solving ODE | Method = euler:  89%|████████▉ | 0.8888888955116272/1.0 [00:00<00:00,  4.44it/s]
inference-batches:  15%|█▌        | 19/123 [00:04<00:23,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.44it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4524524509906769/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9079079031944275/1.0 [00:00<00:00,  4.54it/s]
inference-batches:  16%|█▋        | 20/123 [00:04<00:22,  4.49it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.45045045018196106/1.0 [00:00<00:00,  4.50it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.55it/s] 
inference-batches:  17%|█▋        | 21/123 [00:04<00:22,  4.50it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]



Solving ODE | Method = euler:  40%|████      | 0.402402400970459/1.0 [00:00<00:00,  4.01it/s]


Solving ODE | Method = euler:  85%|████████▌ | 0.8518518805503845/1.0 [00:00<00:00,  4.29it/s]
inference-batches:  18%|█▊        | 22/123 [00:05<00:22,  4.43it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.26it/s]


Solving ODE | Method = euler:  44%|████▎     | 0.4364364445209503/1.0 [00:00<00:00,  4.36it/s]

Solving ODE | Method = euler:  88%|████████▊ | 0.8838838934898376/1.0 [00:00<00:00,  4.43it/s]
inference-batches:  19%|█▊        | 23/123 [00:05<00:22,  4.42it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.38it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.45145145058631897/1.0 [00:00<00:00,  4.51it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9079079031944275/1.0 [00:00<00:00,  4.54it/s] 
inference-batches:  20%|█▉        | 24/123 [00:05<00:22,  4.45it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.45145145058631897/1.0 [00:00<00:00,  4.51it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9089089035987854/1.0 [00:00<00:00,  4.55it/s] 
inference-batches:  20%|██        | 25/123 [00:05<00:21,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.53it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.56it/s] 
inference-batches:  21%|██        | 26/123 [00:05<00:21,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8978978991508484/1.0 [00:00<00:00,  4.49it/s] 
inference-batches:  22%|██▏       | 27/123 [00:06<00:21,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.48it/s]



Solving ODE | Method = euler:  46%|████▌     | 0.4564564526081085/1.0 [00:00<00:00,  4.56it/s]


Solving ODE | Method = euler:  92%|█████████▏| 0.9209209084510803/1.0 [00:00<00:00,  4.60it/s]
inference-batches:  23%|██▎       | 28/123 [00:06<00:21,  4.51it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.57it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4524524509906769/1.0 [00:00<00:00,  4.52it/s]

Solving ODE | Method = euler:  91%|█████████▏| 0.913913905620575/1.0 [00:00<00:00,  4.58it/s] 
inference-batches:  24%|██▎       | 29/123 [00:06<00:20,  4.52it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.55it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.4454454481601715/1.0 [00:00<00:00,  4.45it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8958958983421326/1.0 [00:00<00:00,  4.48it/s]
inference-batches:  24%|██▍       | 30/123 [00:06<00:20,  4.51it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.46it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.4434434473514557/1.0 [00:00<00:00,  4.43it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8988988995552063/1.0 [00:00<00:00,  4.50it/s]
inference-batches:  25%|██▌       | 31/123 [00:07<00:20,  4.50it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.48it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4524524509906769/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  91%|█████████▏| 0.9149149060249329/1.0 [00:00<00:00,  4.58it/s]
inference-batches:  26%|██▌       | 32/123 [00:07<00:20,  4.52it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.56it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4534534513950348/1.0 [00:00<00:00,  4.53it/s]

Solving ODE | Method = euler:  91%|█████████▏| 0.913913905620575/1.0 [00:00<00:00,  4.57it/s] 
inference-batches:  27%|██▋       | 33/123 [00:07<00:19,  4.53it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.56it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4524524509906769/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.55it/s]
inference-batches:  28%|██▊       | 34/123 [00:07<00:19,  4.53it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]

Solving ODE | Method = euler:  91%|█████████▏| 0.913913905620575/1.0 [00:00<00:00,  4.58it/s]  
inference-batches:  28%|██▊       | 35/123 [00:07<00:19,  4.54it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.56it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9079079031944275/1.0 [00:00<00:00,  4.54it/s] 
inference-batches:  29%|██▉       | 36/123 [00:08<00:19,  4.53it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.52it/s] 
inference-batches:  30%|███       | 37/123 [00:08<00:18,  4.53it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4534534513950348/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  91%|█████████▏| 0.913913905620575/1.0 [00:00<00:00,  4.57it/s] 
inference-batches:  31%|███       | 38/123 [00:08<00:18,  4.54it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.55it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.43943944573402405/1.0 [00:00<00:00,  4.39it/s]

Solving ODE | Method = euler:  88%|████████▊ | 0.8788788914680481/1.0 [00:00<00:00,  4.31it/s] 
inference-batches:  32%|███▏      | 39/123 [00:08<00:20,  4.16it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  3.49it/s]



Solving ODE | Method = euler:  40%|████      | 0.4044044017791748/1.0 [00:00<00:00,  4.04it/s]


Solving ODE | Method = euler:  85%|████████▌ | 0.8508508801460266/1.0 [00:00<00:00,  4.29it/s]
inference-batches:  33%|███▎      | 40/123 [00:09<00:19,  4.20it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.27it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.4434434473514557/1.0 [00:00<00:00,  4.43it/s]

Solving ODE | Method = euler:  89%|████████▊ | 0.8868868947029114/1.0 [00:00<00:00,  4.33it/s]
inference-batches:  33%|███▎      | 41/123 [00:09<00:19,  4.24it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.34it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.56it/s] 
inference-batches:  34%|███▍      | 42/123 [00:09<00:18,  4.33it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  42%|████▏     | 0.424424409866333/1.0 [00:00<00:00,  4.24it/s]

Solving ODE | Method = euler:  87%|████████▋ | 0.8698698878288269/1.0 [00:00<00:00,  4.37it/s]
inference-batches:  35%|███▍      | 43/123 [00:09<00:18,  4.31it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.26it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.43843844532966614/1.0 [00:00<00:00,  4.38it/s]


Solving ODE | Method = euler:  89%|████████▉ | 0.8888888955116272/1.0 [00:00<00:00,  4.44it/s] 
inference-batches:  36%|███▌      | 44/123 [00:09<00:18,  4.32it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.34it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.4434434473514557/1.0 [00:00<00:00,  4.43it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.90190190076828/1.0 [00:00<00:00,  4.52it/s]  
inference-batches:  37%|███▋      | 45/123 [00:10<00:17,  4.36it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.44it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.44044044613838196/1.0 [00:00<00:00,  4.40it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8968968987464905/1.0 [00:00<00:00,  4.49it/s] 
inference-batches:  37%|███▋      | 46/123 [00:10<00:17,  4.39it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4534534513950348/1.0 [00:00<00:00,  4.53it/s]

Solving ODE | Method = euler:  92%|█████████▏| 0.9169169068336487/1.0 [00:00<00:00,  4.59it/s]
inference-batches:  38%|███▊      | 47/123 [00:10<00:17,  4.43it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]



Solving ODE | Method = euler:  41%|████      | 0.4114114046096802/1.0 [00:00<00:00,  4.11it/s]


Solving ODE | Method = euler:  86%|████████▌ | 0.8598598837852478/1.0 [00:00<00:00,  4.33it/s]
inference-batches:  39%|███▉      | 48/123 [00:10<00:17,  4.39it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.29it/s]


Solving ODE | Method = euler:  43%|████▎     | 0.4324324131011963/1.0 [00:00<00:00,  4.32it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8968968987464905/1.0 [00:00<00:00,  4.51it/s]
inference-batches:  40%|███▉      | 49/123 [00:11<00:16,  4.42it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4544544517993927/1.0 [00:00<00:00,  4.54it/s]


Solving ODE | Method = euler:  91%|█████████▏| 0.913913905620575/1.0 [00:00<00:00,  4.57it/s] 
inference-batches:  41%|████      | 50/123 [00:11<00:16,  4.46it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.54it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.45045045018196106/1.0 [00:00<00:00,  4.50it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9009009003639221/1.0 [00:00<00:00,  4.39it/s] 
inference-batches:  41%|████▏     | 51/123 [00:11<00:16,  4.44it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.40it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4544544517993927/1.0 [00:00<00:00,  4.54it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9109109044075012/1.0 [00:00<00:00,  4.55it/s]
inference-batches:  42%|████▏     | 52/123 [00:11<00:15,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.4454454481601715/1.0 [00:00<00:00,  4.45it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9049049019813538/1.0 [00:00<00:00,  4.53it/s]
inference-batches:  43%|████▎     | 53/123 [00:11<00:15,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4534534513950348/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  91%|█████████▏| 0.9149149060249329/1.0 [00:00<00:00,  4.58it/s]
inference-batches:  44%|████▍     | 54/123 [00:12<00:15,  4.50it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.56it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4534534513950348/1.0 [00:00<00:00,  4.53it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9109109044075012/1.0 [00:00<00:00,  4.55it/s]
inference-batches:  45%|████▍     | 55/123 [00:12<00:15,  4.51it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.44144144654273987/1.0 [00:00<00:00,  4.41it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8998998999595642/1.0 [00:00<00:00,  4.51it/s] 
inference-batches:  46%|████▌     | 56/123 [00:12<00:14,  4.50it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.48it/s]


Solving ODE | Method = euler:  44%|████▎     | 0.4374374449253082/1.0 [00:00<00:00,  4.37it/s]

Solving ODE | Method = euler:  87%|████████▋ | 0.8748748898506165/1.0 [00:00<00:00,  4.29it/s]
inference-batches:  46%|████▋     | 57/123 [00:12<00:14,  4.42it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.23it/s]



Solving ODE | Method = euler:  43%|████▎     | 0.4344344139099121/1.0 [00:00<00:00,  4.34it/s]


Solving ODE | Method = euler:  89%|████████▉ | 0.8928928971290588/1.0 [00:00<00:00,  4.48it/s]
inference-batches:  47%|████▋     | 58/123 [00:13<00:14,  4.43it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.45it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.45045045018196106/1.0 [00:00<00:00,  4.50it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9089089035987854/1.0 [00:00<00:00,  4.55it/s] 
inference-batches:  48%|████▊     | 59/123 [00:13<00:14,  4.46it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  89%|████████▉ | 0.8948948979377747/1.0 [00:00<00:00,  4.44it/s] 
inference-batches:  49%|████▉     | 60/123 [00:13<00:14,  4.45it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.43it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8988988995552063/1.0 [00:00<00:00,  4.49it/s] 
inference-batches:  50%|████▉     | 61/123 [00:13<00:13,  4.46it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.48it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9059059023857117/1.0 [00:00<00:00,  4.53it/s] 
inference-batches:  50%|█████     | 62/123 [00:14<00:13,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.44044044613838196/1.0 [00:00<00:00,  4.40it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9049049019813538/1.0 [00:00<00:00,  4.54it/s] 
inference-batches:  51%|█████     | 63/123 [00:14<00:13,  4.49it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.90190190076828/1.0 [00:00<00:00,  4.51it/s]   
inference-batches:  52%|█████▏    | 64/123 [00:14<00:13,  4.49it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.45145145058631897/1.0 [00:00<00:00,  4.51it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9049049019813538/1.0 [00:00<00:00,  4.52it/s] 
inference-batches:  53%|█████▎    | 65/123 [00:14<00:12,  4.50it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]



Solving ODE | Method = euler:  43%|████▎     | 0.42842841148376465/1.0 [00:00<00:00,  4.28it/s]


Solving ODE | Method = euler:  87%|████████▋ | 0.8748748898506165/1.0 [00:00<00:00,  4.39it/s] 
inference-batches:  54%|█████▎    | 66/123 [00:14<00:12,  4.45it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.35it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]

Solving ODE | Method = euler:  91%|█████████▏| 0.912912905216217/1.0 [00:00<00:00,  4.58it/s]  
inference-batches:  54%|█████▍    | 67/123 [00:15<00:12,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.54it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9119119048118591/1.0 [00:00<00:00,  4.56it/s] 
inference-batches:  55%|█████▌    | 68/123 [00:15<00:12,  4.49it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.44144144654273987/1.0 [00:00<00:00,  4.41it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.90190190076828/1.0 [00:00<00:00,  4.52it/s]   
inference-batches:  56%|█████▌    | 69/123 [00:15<00:12,  4.49it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8978978991508484/1.0 [00:00<00:00,  4.49it/s]
inference-batches:  57%|█████▋    | 70/123 [00:15<00:11,  4.49it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.44044044613838196/1.0 [00:00<00:00,  4.40it/s]

Solving ODE | Method = euler:  88%|████████▊ | 0.8808808922767639/1.0 [00:00<00:00,  4.22it/s] 
inference-batches:  58%|█████▊    | 71/123 [00:16<00:11,  4.42it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.27it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4544544517993927/1.0 [00:00<00:00,  4.54it/s]


Solving ODE | Method = euler:  92%|█████████▏| 0.9169169068336487/1.0 [00:00<00:00,  4.58it/s]
inference-batches:  59%|█████▊    | 72/123 [00:16<00:11,  4.46it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.56it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4534534513950348/1.0 [00:00<00:00,  4.53it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.55it/s]
inference-batches:  59%|█████▉    | 73/123 [00:16<00:11,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]



Solving ODE | Method = euler:  44%|████▎     | 0.4374374449253082/1.0 [00:00<00:00,  4.37it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8968968987464905/1.0 [00:00<00:00,  4.50it/s]
inference-batches:  60%|██████    | 74/123 [00:16<00:10,  4.48it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.52it/s] 
inference-batches:  61%|██████    | 75/123 [00:16<00:10,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4524524509906769/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  91%|█████████▏| 0.912912905216217/1.0 [00:00<00:00,  4.57it/s] 
inference-batches:  62%|██████▏   | 76/123 [00:17<00:10,  4.50it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.55it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4544544517993927/1.0 [00:00<00:00,  4.54it/s]

Solving ODE | Method = euler:  91%|█████████▏| 0.9149149060249329/1.0 [00:00<00:00,  4.57it/s]
inference-batches:  63%|██████▎   | 77/123 [00:17<00:10,  4.52it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.55it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.4454454481601715/1.0 [00:00<00:00,  4.45it/s]


Solving ODE | Method = euler:  89%|████████▉ | 0.890890896320343/1.0 [00:00<00:00,  4.25it/s] 
inference-batches:  63%|██████▎   | 78/123 [00:17<00:10,  4.45it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.28it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.47it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9089089035987854/1.0 [00:00<00:00,  4.55it/s] 
inference-batches:  64%|██████▍   | 79/123 [00:17<00:09,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9029029011726379/1.0 [00:00<00:00,  4.51it/s] 
inference-batches:  65%|██████▌   | 80/123 [00:18<00:09,  4.48it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.52it/s] 
inference-batches:  66%|██████▌   | 81/123 [00:18<00:09,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4424424469470978/1.0 [00:00<00:00,  4.42it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8958958983421326/1.0 [00:00<00:00,  4.48it/s]
inference-batches:  67%|██████▋   | 82/123 [00:18<00:09,  4.48it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.46it/s]


Solving ODE | Method = euler:  44%|████▎     | 0.4374374449253082/1.0 [00:00<00:00,  4.37it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8998998999595642/1.0 [00:00<00:00,  4.52it/s]
inference-batches:  67%|██████▋   | 83/123 [00:18<00:08,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]



Solving ODE | Method = euler:  42%|████▏     | 0.4214214086532593/1.0 [00:00<00:00,  4.21it/s]


Solving ODE | Method = euler:  88%|████████▊ | 0.8768768906593323/1.0 [00:00<00:00,  4.41it/s]
inference-batches:  68%|██████▊   | 84/123 [00:18<00:08,  4.45it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.38it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9069069027900696/1.0 [00:00<00:00,  4.54it/s] 
inference-batches:  69%|██████▉   | 85/123 [00:19<00:08,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9119119048118591/1.0 [00:00<00:00,  4.57it/s] 
inference-batches:  70%|██████▉   | 86/123 [00:19<00:08,  4.49it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]

Solving ODE | Method = euler:  89%|████████▉ | 0.8948948979377747/1.0 [00:00<00:00,  4.42it/s] 
inference-batches:  71%|███████   | 87/123 [00:19<00:08,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.42it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]


Solving ODE | Method = euler:  89%|████████▉ | 0.8888888955116272/1.0 [00:00<00:00,  4.37it/s]
inference-batches:  72%|███████▏  | 88/123 [00:19<00:07,  4.44it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.38it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.55it/s] 
inference-batches:  72%|███████▏  | 89/123 [00:20<00:07,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.53it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.43843844532966614/1.0 [00:00<00:00,  4.38it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.90190190076828/1.0 [00:00<00:00,  4.53it/s]   
inference-batches:  73%|███████▎  | 90/123 [00:20<00:07,  4.48it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  46%|████▌     | 0.4554554522037506/1.0 [00:00<00:00,  4.55it/s]

Solving ODE | Method = euler:  92%|█████████▏| 0.9199199080467224/1.0 [00:00<00:00,  4.60it/s]
inference-batches:  74%|███████▍  | 91/123 [00:20<00:07,  4.51it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.58it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4524524509906769/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9089089035987854/1.0 [00:00<00:00,  4.54it/s]
inference-batches:  75%|███████▍  | 92/123 [00:20<00:06,  4.51it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.53it/s]
inference-batches:  76%|███████▌  | 93/123 [00:20<00:06,  4.51it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.43843844532966614/1.0 [00:00<00:00,  4.38it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.54it/s] 
inference-batches:  76%|███████▋  | 94/123 [00:21<00:06,  4.51it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]


Solving ODE | Method = euler:  39%|███▉      | 0.38838839530944824/1.0 [00:00<00:00,  3.88it/s]

Solving ODE | Method = euler:  85%|████████▍ | 0.8488488793373108/1.0 [00:00<00:00,  4.30it/s] 
inference-batches:  77%|███████▋  | 95/123 [00:21<00:06,  4.44it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.27it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.90190190076828/1.0 [00:00<00:00,  4.52it/s]  
inference-batches:  78%|███████▊  | 96/123 [00:21<00:06,  4.45it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8988988995552063/1.0 [00:00<00:00,  4.49it/s] 
inference-batches:  79%|███████▉  | 97/123 [00:21<00:05,  4.46it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.4454454481601715/1.0 [00:00<00:00,  4.45it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9069069027900696/1.0 [00:00<00:00,  4.54it/s]
inference-batches:  80%|███████▉  | 98/123 [00:22<00:05,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.43843844532966614/1.0 [00:00<00:00,  4.38it/s]

Solving ODE | Method = euler:  89%|████████▉ | 0.8878878951072693/1.0 [00:00<00:00,  4.45it/s] 
inference-batches:  80%|████████  | 99/123 [00:22<00:05,  4.46it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.43it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.56it/s] 
inference-batches:  81%|████████▏ | 100/123 [00:22<00:05,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.43843844532966614/1.0 [00:00<00:00,  4.37it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8978978991508484/1.0 [00:00<00:00,  4.50it/s] 
inference-batches:  82%|████████▏ | 101/123 [00:22<00:04,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.46it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.43843844532966614/1.0 [00:00<00:00,  4.38it/s]


Solving ODE | Method = euler:  88%|████████▊ | 0.879879891872406/1.0 [00:00<00:00,  4.40it/s]  
inference-batches:  83%|████████▎ | 102/123 [00:22<00:04,  4.45it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.39it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9049049019813538/1.0 [00:00<00:00,  4.53it/s] 
inference-batches:  84%|████████▎ | 103/123 [00:23<00:04,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9079079031944275/1.0 [00:00<00:00,  4.54it/s] 
inference-batches:  85%|████████▍ | 104/123 [00:23<00:04,  4.48it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.4464464485645294/1.0 [00:00<00:00,  4.46it/s]

Solving ODE | Method = euler:  89%|████████▉ | 0.8938938975334167/1.0 [00:00<00:00,  4.47it/s]
inference-batches:  85%|████████▌ | 105/123 [00:23<00:04,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.45it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.44144144654273987/1.0 [00:00<00:00,  4.41it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9029029011726379/1.0 [00:00<00:00,  4.53it/s] 
inference-batches:  86%|████████▌ | 106/123 [00:23<00:03,  4.49it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.4424424469470978/1.0 [00:00<00:00,  4.42it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9029029011726379/1.0 [00:00<00:00,  4.53it/s]
inference-batches:  87%|████████▋ | 107/123 [00:24<00:03,  4.49it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]



Solving ODE | Method = euler:  42%|████▏     | 0.424424409866333/1.0 [00:00<00:00,  4.24it/s]


Solving ODE | Method = euler:  88%|████████▊ | 0.8848848938941956/1.0 [00:00<00:00,  4.45it/s]
inference-batches:  88%|████████▊ | 108/123 [00:24<00:03,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.41it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8998998999595642/1.0 [00:00<00:00,  4.50it/s]
inference-batches:  89%|████████▊ | 109/123 [00:24<00:03,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.4464464485645294/1.0 [00:00<00:00,  4.46it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.56it/s]
inference-batches:  89%|████████▉ | 110/123 [00:24<00:02,  4.49it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.43943944573402405/1.0 [00:00<00:00,  4.39it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8998998999595642/1.0 [00:00<00:00,  4.52it/s] 
inference-batches:  90%|█████████ | 111/123 [00:24<00:02,  4.49it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9009009003639221/1.0 [00:00<00:00,  4.51it/s]
inference-batches:  91%|█████████ | 112/123 [00:25<00:02,  4.49it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8968968987464905/1.0 [00:00<00:00,  4.49it/s]
inference-batches:  92%|█████████▏| 113/123 [00:25<00:02,  4.46it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.37it/s]



Solving ODE | Method = euler:  44%|████▎     | 0.43543541431427/1.0 [00:00<00:00,  4.35it/s]


Solving ODE | Method = euler:  89%|████████▉ | 0.8928928971290588/1.0 [00:00<00:00,  4.48it/s]
inference-batches:  93%|█████████▎| 114/123 [00:25<00:02,  4.46it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.46it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4524524509906769/1.0 [00:00<00:00,  4.52it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9119119048118591/1.0 [00:00<00:00,  4.56it/s]
inference-batches:  93%|█████████▎| 115/123 [00:25<00:01,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.54it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9069069027900696/1.0 [00:00<00:00,  4.54it/s] 
inference-batches:  94%|█████████▍| 116/123 [00:26<00:01,  4.50it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9059059023857117/1.0 [00:00<00:00,  4.53it/s] 
inference-batches:  95%|█████████▌| 117/123 [00:26<00:01,  4.50it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9009009003639221/1.0 [00:00<00:00,  4.50it/s] 
inference-batches:  96%|█████████▌| 118/123 [00:26<00:01,  4.49it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.4464464485645294/1.0 [00:00<00:00,  4.46it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9069069027900696/1.0 [00:00<00:00,  4.54it/s]
inference-batches:  97%|█████████▋| 119/123 [00:26<00:00,  4.50it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]



Solving ODE | Method = euler:  43%|████▎     | 0.4324324131011963/1.0 [00:00<00:00,  4.32it/s]


Solving ODE | Method = euler:  89%|████████▊ | 0.8868868947029114/1.0 [00:00<00:00,  4.45it/s]
inference-batches:  98%|█████████▊| 120/123 [00:26<00:00,  4.48it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.42it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.4424424469470978/1.0 [00:00<00:00,  4.42it/s]

Solving ODE | Method = euler:  88%|████████▊ | 0.8848848938941956/1.0 [00:00<00:00,  4.35it/s]
inference-batches:  98%|█████████▊| 121/123 [00:27<00:00,  4.44it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.35it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.90190190076828/1.0 [00:00<00:00,  4.51it/s]   
inference-batches:  99%|█████████▉| 122/123 [00:27<00:00,  4.46it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  53%|█████▎    | 0.5295295119285583/1.0 [00:00<00:00,  5.29it/s]
inference-batches: 100%|██████████| 123/123 [00:27<00:00,  4.46it/s]
Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  5.42it/s]
2026-10-14 00:42:34.726 | INFO     | __main__:<module>:300 - Reflow iteration # 1 : Generating z0_test,z1_test for reflow_iteration # 2

inference-batches:   0%|          | 0/49 [00:00<?, ?it/s]

Solving ODE | Method = euler:   0%|          | 0/1.0 [00:00<?, ?it/s]

Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.90190190076828/1.0 [00:00<00:00,  4.51it/s]   
inference-batches:   2%|▏         | 1/49 [00:00<00:10,  4.49it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.48it/s]



Solving ODE | Method = euler:  41%|████▏     | 0.4144144058227539/1.0 [00:00<00:00,  4.14it/s]


Solving ODE | Method = euler:  84%|████████▍ | 0.8438438177108765/1.0 [00:00<00:00,  4.23it/s]
inference-batches:   4%|▍         | 2/49 [00:00<00:10,  4.31it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.19it/s]


Solving ODE | Method = euler:  43%|████▎     | 0.42742741107940674/1.0 [00:00<00:00,  4.27it/s]

Solving ODE | Method = euler:  89%|████████▉ | 0.8898898959159851/1.0 [00:00<00:00,  4.48it/s] 
inference-batches:   6%|▌         | 3/49 [00:00<00:10,  4.37it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.44it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.44144144654273987/1.0 [00:00<00:00,  4.41it/s]


Solving ODE | Method = euler:  89%|████████▉ | 0.8878878951072693/1.0 [00:00<00:00,  4.44it/s] 
inference-batches:   8%|▊         | 4/49 [00:00<00:10,  4.39it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.43it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.43843844532966614/1.0 [00:00<00:00,  4.38it/s]

Solving ODE | Method = euler:  89%|████████▊ | 0.8858858942985535/1.0 [00:00<00:00,  4.43it/s] 
inference-batches:  10%|█         | 5/49 [00:01<00:10,  4.40it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.40it/s]



Solving ODE | Method = euler:  44%|████▎     | 0.4374374449253082/1.0 [00:00<00:00,  4.37it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8958958983421326/1.0 [00:00<00:00,  4.49it/s]
inference-batches:  12%|█▏        | 6/49 [00:01<00:09,  4.42it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.45it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9059059023857117/1.0 [00:00<00:00,  4.53it/s] 
inference-batches:  14%|█▍        | 7/49 [00:01<00:09,  4.45it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]


Solving ODE | Method = euler:  89%|████████▉ | 0.890890896320343/1.0 [00:00<00:00,  4.45it/s] 
inference-batches:  16%|█▋        | 8/49 [00:01<00:09,  4.44it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.44it/s]


Solving ODE | Method = euler:  43%|████▎     | 0.4334334135055542/1.0 [00:00<00:00,  4.33it/s]

Solving ODE | Method = euler:  88%|████████▊ | 0.8838838934898376/1.0 [00:00<00:00,  4.43it/s]
inference-batches:  18%|█▊        | 9/49 [00:02<00:09,  4.43it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.39it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.44144144654273987/1.0 [00:00<00:00,  4.41it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8978978991508484/1.0 [00:00<00:00,  4.50it/s] 
inference-batches:  20%|██        | 10/49 [00:02<00:08,  4.44it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.44144144654273987/1.0 [00:00<00:00,  4.41it/s]

Solving ODE | Method = euler:  89%|████████▉ | 0.890890896320343/1.0 [00:00<00:00,  4.46it/s]  
inference-batches:  22%|██▏       | 11/49 [00:02<00:08,  4.44it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.43it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4434434473514557/1.0 [00:00<00:00,  4.43it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8968968987464905/1.0 [00:00<00:00,  4.49it/s]
inference-batches:  24%|██▍       | 12/49 [00:02<00:08,  4.45it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.46it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.43943944573402405/1.0 [00:00<00:00,  4.39it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8988988995552063/1.0 [00:00<00:00,  4.51it/s] 
inference-batches:  27%|██▋       | 13/49 [00:02<00:08,  4.45it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.45it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8998998999595642/1.0 [00:00<00:00,  4.50it/s] 
inference-batches:  29%|██▊       | 14/49 [00:03<00:07,  4.46it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.53it/s] 
inference-batches:  31%|███       | 15/49 [00:03<00:07,  4.47it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.51it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.45145145058631897/1.0 [00:00<00:00,  4.51it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.55it/s] 
inference-batches:  33%|███▎      | 16/49 [00:03<00:07,  4.49it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.45045045018196106/1.0 [00:00<00:00,  4.50it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9009009003639221/1.0 [00:00<00:00,  4.39it/s] 
inference-batches:  35%|███▍      | 17/49 [00:03<00:07,  4.46it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.37it/s]



Solving ODE | Method = euler:  43%|████▎     | 0.4344344139099121/1.0 [00:00<00:00,  4.34it/s]


Solving ODE | Method = euler:  88%|████████▊ | 0.8848848938941956/1.0 [00:00<00:00,  4.43it/s]
inference-batches:  37%|███▋      | 18/49 [00:04<00:06,  4.44it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.41it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8968968987464905/1.0 [00:00<00:00,  4.49it/s]
inference-batches:  39%|███▉      | 19/49 [00:04<00:06,  4.45it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.4464464485645294/1.0 [00:00<00:00,  4.46it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.52it/s]
inference-batches:  41%|████      | 20/49 [00:04<00:06,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  44%|████▍     | 0.44044044613838196/1.0 [00:00<00:00,  4.40it/s]

Solving ODE | Method = euler:  89%|████████▉ | 0.8948948979377747/1.0 [00:00<00:00,  4.48it/s] 
inference-batches:  43%|████▎     | 21/49 [00:04<00:06,  4.46it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.46it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  90%|████████▉ | 0.8978978991508484/1.0 [00:00<00:00,  4.49it/s] 
inference-batches:  45%|████▍     | 22/49 [00:04<00:06,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.48it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.4464464485645294/1.0 [00:00<00:00,  4.46it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8978978991508484/1.0 [00:00<00:00,  4.49it/s]
inference-batches:  47%|████▋     | 23/49 [00:05<00:05,  4.46it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.45it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.44044044613838196/1.0 [00:00<00:00,  4.40it/s]


Solving ODE | Method = euler:  89%|████████▊ | 0.8868868947029114/1.0 [00:00<00:00,  4.43it/s] 
inference-batches:  49%|████▉     | 24/49 [00:05<00:05,  4.45it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.41it/s]


Solving ODE | Method = euler:  43%|████▎     | 0.42842841148376465/1.0 [00:00<00:00,  4.28it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8958958983421326/1.0 [00:00<00:00,  4.51it/s] 
inference-batches:  51%|█████     | 25/49 [00:05<00:05,  4.45it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.46it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4424424469470978/1.0 [00:00<00:00,  4.42it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9059059023857117/1.0 [00:00<00:00,  4.54it/s]
inference-batches:  53%|█████▎    | 26/49 [00:05<00:05,  4.47it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  41%|████      | 0.40840840339660645/1.0 [00:00<00:00,  4.08it/s]

Solving ODE | Method = euler:  86%|████████▌ | 0.8598598837852478/1.0 [00:00<00:00,  4.33it/s] 
inference-batches:  55%|█████▌    | 27/49 [00:06<00:04,  4.43it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.32it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.45145145058631897/1.0 [00:00<00:00,  4.51it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.55it/s] 
inference-batches:  57%|█████▋    | 28/49 [00:06<00:04,  4.46it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.4464464485645294/1.0 [00:00<00:00,  4.46it/s]

Solving ODE | Method = euler:  89%|████████▉ | 0.8928928971290588/1.0 [00:00<00:00,  4.45it/s]
inference-batches:  59%|█████▉    | 29/49 [00:06<00:04,  4.46it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.45it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.45045045018196106/1.0 [00:00<00:00,  4.50it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9009009003639221/1.0 [00:00<00:00,  4.50it/s] 
inference-batches:  61%|██████    | 30/49 [00:06<00:04,  4.46it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.45145145058631897/1.0 [00:00<00:00,  4.51it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.55it/s] 
inference-batches:  63%|██████▎   | 31/49 [00:06<00:04,  4.48it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44744744896888733/1.0 [00:00<00:00,  4.47it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.90190190076828/1.0 [00:00<00:00,  4.51it/s]   
inference-batches:  65%|██████▌   | 32/49 [00:07<00:03,  4.48it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4544544517993927/1.0 [00:00<00:00,  4.54it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9089089035987854/1.0 [00:00<00:00,  4.54it/s]
inference-batches:  67%|██████▋   | 33/49 [00:07<00:03,  4.50it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4524524509906769/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9109109044075012/1.0 [00:00<00:00,  4.56it/s]
inference-batches:  69%|██████▉   | 34/49 [00:07<00:03,  4.51it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.4454454481601715/1.0 [00:00<00:00,  4.45it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.52it/s]
inference-batches:  71%|███████▏  | 35/49 [00:07<00:03,  4.51it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.4464464485645294/1.0 [00:00<00:00,  4.46it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.53it/s]
inference-batches:  73%|███████▎  | 36/49 [00:08<00:02,  4.51it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4534534513950348/1.0 [00:00<00:00,  4.53it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9119119048118591/1.0 [00:00<00:00,  4.56it/s]
inference-batches:  76%|███████▌  | 37/49 [00:08<00:02,  4.52it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.54it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.4534534513950348/1.0 [00:00<00:00,  4.53it/s]


Solving ODE | Method = euler:  91%|█████████▏| 0.9149149060249329/1.0 [00:00<00:00,  4.58it/s]
inference-batches:  78%|███████▊  | 38/49 [00:08<00:02,  4.53it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.55it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4544544517993927/1.0 [00:00<00:00,  4.54it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.54it/s]
inference-batches:  80%|███████▉  | 39/49 [00:08<00:02,  4.53it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]



Solving ODE | Method = euler:  45%|████▌     | 0.45145145058631897/1.0 [00:00<00:00,  4.51it/s]


Solving ODE | Method = euler:  91%|█████████▏| 0.912912905216217/1.0 [00:00<00:00,  4.57it/s]  
inference-batches:  82%|████████▏ | 40/49 [00:08<00:01,  4.53it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.54it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.4464464485645294/1.0 [00:00<00:00,  4.45it/s]

Solving ODE | Method = euler:  90%|█████████ | 0.9029029011726379/1.0 [00:00<00:00,  4.52it/s]
inference-batches:  84%|████████▎ | 41/49 [00:09<00:01,  4.52it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.48it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44844844937324524/1.0 [00:00<00:00,  4.48it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9039039015769958/1.0 [00:00<00:00,  4.52it/s] 
inference-batches:  86%|████████▌ | 42/49 [00:09<00:01,  4.52it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.50it/s]


Solving ODE | Method = euler:  45%|████▍     | 0.4454454481601715/1.0 [00:00<00:00,  4.45it/s]

Solving ODE | Method = euler:  90%|████████▉ | 0.8988988995552063/1.0 [00:00<00:00,  4.50it/s]
inference-batches:  88%|████████▊ | 43/49 [00:09<00:01,  4.51it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.48it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  91%|█████████ | 0.9099099040031433/1.0 [00:00<00:00,  4.55it/s] 
inference-batches:  90%|████████▉ | 44/49 [00:09<00:01,  4.51it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.52it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.4534534513950348/1.0 [00:00<00:00,  4.53it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9119119048118591/1.0 [00:00<00:00,  4.56it/s]
inference-batches:  92%|█████████▏| 45/49 [00:10<00:00,  4.52it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.54it/s]



Solving ODE | Method = euler:  45%|████▍     | 0.44944944977760315/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9029029011726379/1.0 [00:00<00:00,  4.51it/s] 
inference-batches:  94%|█████████▍| 46/49 [00:10<00:00,  4.51it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  45%|████▌     | 0.45045045018196106/1.0 [00:00<00:00,  4.50it/s]

Solving ODE | Method = euler:  91%|█████████ | 0.9119119048118591/1.0 [00:00<00:00,  4.56it/s] 
inference-batches:  96%|█████████▌| 47/49 [00:10<00:00,  4.52it/s]


Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.54it/s]



Solving ODE | Method = euler:  44%|████▍     | 0.4444444477558136/1.0 [00:00<00:00,  4.44it/s]


Solving ODE | Method = euler:  90%|█████████ | 0.9029029011726379/1.0 [00:00<00:00,  4.52it/s]
inference-batches:  98%|█████████▊| 48/49 [00:10<00:00,  4.51it/s]

Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.49it/s]


Solving ODE | Method = euler:  48%|████▊     | 0.4764764606952667/1.0 [00:00<00:00,  4.76it/s]

Solving ODE | Method = euler:  96%|█████████▌| 0.9569569826126099/1.0 [00:00<00:00,  4.78it/s]
inference-batches: 100%|██████████| 49/49 [00:10<00:00,  4.48it/s]
Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [00:00<00:00,  4.76it/s]
reflow iterations:  50%|█████     | 1/2 [15:47<15:47, 947.57s/it]2026-10-14 00:42:45.668 | INFO     | __main__:<module>:266 - Reflow iteration: 1
2026-10-14 00:42:46.406 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:57 - @epoch = 0 (before training), loss_fn = MSELoss(), loss_value = 2.713577
2026-10-14 00:42:46.410 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:64 - MSELoss(test) before training:2.7135770320892334

Epoch:   0%|          | 0/300 [00:00<?, ?it/s]2026-10-14 00:42:49.365 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 1, smoothed-train-loss = 0.305680 | smoothed-test-loss = 0.093926 | rel_loss_delta = inf | loss_stagnation_count = 0

Epoch:   0%|          | 1/300 [00:02<14:43,  2.95s/it]2026-10-14 00:42:52.321 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 2, smoothed-train-loss = 0.093771 | smoothed-test-loss = 0.057519 | rel_loss_delta = inf | loss_stagnation_count = 0

Epoch:   1%|          | 2/300 [00:05<14:40,  2.96s/it]2026-10-14 00:42:55.298 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 3, smoothed-train-loss = 0.052202 | smoothed-test-loss = 0.043638 | rel_loss_delta = -0.44329775041595815 | loss_stagnation_count = 0

Epoch:   1%|          | 3/300 [00:08<14:40,  2.97s/it]2026-10-14 00:42:58.234 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 4, smoothed-train-loss = 0.039898 | smoothed-test-loss = 0.035371 | rel_loss_delta = -0.2357112104797507 | loss_stagnation_count = 0

Epoch:   1%|▏         | 4/300 [00:11<14:34,  2.95s/it]2026-10-14 00:43:01.194 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 5, smoothed-train-loss = 0.033952 | smoothed-test-loss = 0.031530 | rel_loss_delta = -0.14901157500259163 | loss_stagnation_count = 0

Epoch:   2%|▏         | 5/300 [00:14<14:31,  2.96s/it]2026-10-14 00:43:04.059 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 6, smoothed-train-loss = 0.030501 | smoothed-test-loss = 0.027982 | rel_loss_delta = -0.10165683764814995 | loss_stagnation_count = 0

Epoch:   2%|▏         | 6/300 [00:17<14:19,  2.93s/it]2026-10-14 00:43:07.008 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 7, smoothed-train-loss = 0.028104 | smoothed-test-loss = 0.026276 | rel_loss_delta = -0.07857158646202583 | loss_stagnation_count = 0

Epoch:   2%|▏         | 7/300 [00:20<14:19,  2.93s/it]2026-10-14 00:43:10.068 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 8, smoothed-train-loss = 0.026502 | smoothed-test-loss = 0.025029 | rel_loss_delta = -0.057029081364503235 | loss_stagnation_count = 0

Epoch:   3%|▎         | 8/300 [00:23<14:28,  2.97s/it]2026-10-14 00:43:12.981 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 9, smoothed-train-loss = 0.025435 | smoothed-test-loss = 0.023808 | rel_loss_delta = -0.04024892523877124 | loss_stagnation_count = 0

Epoch:   3%|▎         | 9/300 [00:26<14:19,  2.95s/it]2026-10-14 00:43:15.851 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 10, smoothed-train-loss = 0.024499 | smoothed-test-loss = 0.023168 | rel_loss_delta = -0.03680950690881745 | loss_stagnation_count = 0

Epoch:   3%|▎         | 10/300 [00:29<14:09,  2.93s/it]2026-10-14 00:43:18.968 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 11, smoothed-train-loss = 0.023975 | smoothed-test-loss = 0.022602 | rel_loss_delta = -0.021369755663948532 | loss_stagnation_count = 0

Epoch:   4%|▎         | 11/300 [00:32<14:22,  2.99s/it]2026-10-14 00:43:22.011 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 12, smoothed-train-loss = 0.023332 | smoothed-test-loss = 0.022262 | rel_loss_delta = -0.02682826222399049 | loss_stagnation_count = 0

Epoch:   4%|▍         | 12/300 [00:35<14:24,  3.00s/it]2026-10-14 00:43:25.079 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 13, smoothed-train-loss = 0.022930 | smoothed-test-loss = 0.022473 | rel_loss_delta = -0.017249864432887396 | loss_stagnation_count = 0

Epoch:   4%|▍         | 13/300 [00:38<14:27,  3.02s/it]2026-10-14 00:43:28.001 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 14, smoothed-train-loss = 0.022585 | smoothed-test-loss = 0.022222 | rel_loss_delta = -0.015028740545605612 | loss_stagnation_count = 0

Epoch:   5%|▍         | 14/300 [00:41<14:15,  2.99s/it]2026-10-14 00:43:31.053 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 15, smoothed-train-loss = 0.022302 | smoothed-test-loss = 0.021628 | rel_loss_delta = -0.012545607508096817 | loss_stagnation_count = 0

Epoch:   5%|▌         | 15/300 [00:44<14:17,  3.01s/it]2026-10-14 00:43:34.071 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 16, smoothed-train-loss = 0.022064 | smoothed-test-loss = 0.021570 | rel_loss_delta = -0.010673358656395713 | loss_stagnation_count = 0

Epoch:   5%|▌         | 16/300 [00:47<14:15,  3.01s/it]2026-10-14 00:43:37.073 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 17, smoothed-train-loss = 0.021782 | smoothed-test-loss = 0.020628 | rel_loss_delta = -0.01274464237520166 | loss_stagnation_count = 0

Epoch:   6%|▌         | 17/300 [00:50<14:11,  3.01s/it]2026-10-14 00:43:40.161 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 18, smoothed-train-loss = 0.021793 | smoothed-test-loss = 0.020814 | rel_loss_delta = 0.0004980547725651014 | loss_stagnation_count = 0

Epoch:   6%|▌         | 18/300 [00:53<14:15,  3.03s/it]2026-10-14 00:43:43.234 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 19, smoothed-train-loss = 0.021551 | smoothed-test-loss = 0.021345 | rel_loss_delta = -0.01112159534452042 | loss_stagnation_count = 0

Epoch:   6%|▋         | 19/300 [00:56<14:15,  3.05s/it]2026-10-14 00:43:46.299 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 20, smoothed-train-loss = 0.021445 | smoothed-test-loss = 0.021076 | rel_loss_delta = -0.004904945499316116 | loss_stagnation_count = 0

Epoch:   7%|▋         | 20/300 [00:59<14:14,  3.05s/it]2026-10-14 00:43:49.280 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 21, smoothed-train-loss = 0.021453 | smoothed-test-loss = 0.020771 | rel_loss_delta = 0.00037383815284651987 | loss_stagnation_count = 0

Epoch:   7%|▋         | 21/300 [01:02<14:05,  3.03s/it]2026-10-14 00:43:52.149 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 22, smoothed-train-loss = 0.021321 | smoothed-test-loss = 0.020664 | rel_loss_delta = -0.006166080672812593 | loss_stagnation_count = 0

Epoch:   7%|▋         | 22/300 [01:05<13:48,  2.98s/it]2026-10-14 00:43:55.304 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 23, smoothed-train-loss = 0.021239 | smoothed-test-loss = 0.020270 | rel_loss_delta = -0.0038289639439955973 | loss_stagnation_count = 0

Epoch:   8%|▊         | 23/300 [01:08<14:00,  3.03s/it]2026-10-14 00:43:58.301 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 24, smoothed-train-loss = 0.021227 | smoothed-test-loss = 0.021471 | rel_loss_delta = -0.0005563495587046108 | loss_stagnation_count = 0

Epoch:   8%|▊         | 24/300 [01:11<13:54,  3.02s/it]2026-10-14 00:44:01.365 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 25, smoothed-train-loss = 0.020939 | smoothed-test-loss = 0.020553 | rel_loss_delta = -0.013607798604471628 | loss_stagnation_count = 0

Epoch:   8%|▊         | 25/300 [01:14<13:54,  3.04s/it]2026-10-14 00:44:04.244 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 26, smoothed-train-loss = 0.021032 | smoothed-test-loss = 0.020709 | rel_loss_delta = 0.004445611794914827 | loss_stagnation_count = 0

Epoch:   9%|▊         | 26/300 [01:17<13:38,  2.99s/it]2026-10-14 00:44:07.189 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 27, smoothed-train-loss = 0.020871 | smoothed-test-loss = 0.020385 | rel_loss_delta = -0.0076319998707080035 | loss_stagnation_count = 0

Epoch:   9%|▉         | 27/300 [01:20<13:32,  2.98s/it]2026-10-14 00:44:10.216 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 28, smoothed-train-loss = 0.020925 | smoothed-test-loss = 0.020124 | rel_loss_delta = 0.0025634919267271115 | loss_stagnation_count = 0

Epoch:   9%|▉         | 28/300 [01:23<13:33,  2.99s/it]2026-10-14 00:44:13.309 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 29, smoothed-train-loss = 0.020945 | smoothed-test-loss = 0.019984 | rel_loss_delta = 0.0009725997644457039 | loss_stagnation_count = 0

Epoch:  10%|▉         | 29/300 [01:26<13:38,  3.02s/it]2026-10-14 00:44:16.230 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 30, smoothed-train-loss = 0.020873 | smoothed-test-loss = 0.021219 | rel_loss_delta = -0.003437551143779828 | loss_stagnation_count = 0

Epoch:  10%|█         | 30/300 [01:29<13:27,  2.99s/it]2026-10-14 00:44:19.186 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 31, smoothed-train-loss = 0.021014 | smoothed-test-loss = 0.021048 | rel_loss_delta = 0.0067631828512708745 | loss_stagnation_count = 0

Epoch:  10%|█         | 31/300 [01:32<13:21,  2.98s/it]2026-10-14 00:44:22.226 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 32, smoothed-train-loss = 0.020802 | smoothed-test-loss = 0.020330 | rel_loss_delta = -0.01008541623021419 | loss_stagnation_count = 0

Epoch:  11%|█         | 32/300 [01:35<13:23,  3.00s/it]2026-10-14 00:44:25.270 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 33, smoothed-train-loss = 0.020769 | smoothed-test-loss = 0.019554 | rel_loss_delta = -0.001619827343852737 | loss_stagnation_count = 0

Epoch:  11%|█         | 33/300 [01:38<13:24,  3.01s/it]2026-10-14 00:44:28.120 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 34, smoothed-train-loss = 0.020848 | smoothed-test-loss = 0.020298 | rel_loss_delta = 0.0038326315114455244 | loss_stagnation_count = 0

Epoch:  11%|█▏        | 34/300 [01:41<13:08,  2.96s/it]2026-10-14 00:44:31.088 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 35, smoothed-train-loss = 0.020890 | smoothed-test-loss = 0.019581 | rel_loss_delta = 0.002012975073391317 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 35/300 [01:44<13:05,  2.96s/it]2026-10-14 00:44:34.058 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 36, smoothed-train-loss = 0.020770 | smoothed-test-loss = 0.019814 | rel_loss_delta = -0.005741239268488923 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 36/300 [01:47<13:03,  2.97s/it]2026-10-14 00:44:37.023 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 37, smoothed-train-loss = 0.020556 | smoothed-test-loss = 0.019793 | rel_loss_delta = -0.01033301567299516 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 37/300 [01:50<13:00,  2.97s/it]2026-10-14 00:44:39.948 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 38, smoothed-train-loss = 0.020649 | smoothed-test-loss = 0.019544 | rel_loss_delta = 0.0045605677259599985 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 38/300 [01:53<12:53,  2.95s/it]2026-10-14 00:44:42.929 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 39, smoothed-train-loss = 0.020544 | smoothed-test-loss = 0.019823 | rel_loss_delta = -0.005115754388344276 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 39/300 [01:56<12:53,  2.96s/it]2026-10-14 00:44:45.914 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 40, smoothed-train-loss = 0.020542 | smoothed-test-loss = 0.019934 | rel_loss_delta = -7.810231918319658e-05 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 40/300 [01:59<12:51,  2.97s/it]2026-10-14 00:44:48.961 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 41, smoothed-train-loss = 0.020647 | smoothed-test-loss = 0.020388 | rel_loss_delta = 0.005089715989452458 | loss_stagnation_count = 0

Epoch:  14%|█▎        | 41/300 [02:02<12:55,  2.99s/it]2026-10-14 00:44:51.874 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 42, smoothed-train-loss = 0.020544 | smoothed-test-loss = 0.019655 | rel_loss_delta = -0.004971782911834777 | loss_stagnation_count = 0

Epoch:  14%|█▍        | 42/300 [02:05<12:45,  2.97s/it]2026-10-14 00:44:54.981 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 43, smoothed-train-loss = 0.020551 | smoothed-test-loss = 0.019568 | rel_loss_delta = 0.00032036172566609993 | loss_stagnation_count = 0

Epoch:  14%|█▍        | 43/300 [02:08<12:53,  3.01s/it]2026-10-14 00:44:58.062 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 44, smoothed-train-loss = 0.020449 | smoothed-test-loss = 0.019424 | rel_loss_delta = -0.004934667469956058 | loss_stagnation_count = 0

Epoch:  15%|█▍        | 44/300 [02:11<12:56,  3.03s/it]2026-10-14 00:45:00.993 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 45, smoothed-train-loss = 0.020333 | smoothed-test-loss = 0.019110 | rel_loss_delta = -0.005690186234361573 | loss_stagnation_count = 0

Epoch:  15%|█▌        | 45/300 [02:14<12:45,  3.00s/it]2026-10-14 00:45:04.009 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 46, smoothed-train-loss = 0.020481 | smoothed-test-loss = 0.019467 | rel_loss_delta = 0.007289446465771177 | loss_stagnation_count = 0

Epoch:  15%|█▌        | 46/300 [02:17<12:43,  3.01s/it]2026-10-14 00:45:07.054 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 47, smoothed-train-loss = 0.020537 | smoothed-test-loss = 0.020164 | rel_loss_delta = 0.0027327839226332262 | loss_stagnation_count = 0

Epoch:  16%|█▌        | 47/300 [02:20<12:43,  3.02s/it]2026-10-14 00:45:10.124 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 48, smoothed-train-loss = 0.020249 | smoothed-test-loss = 0.020311 | rel_loss_delta = -0.014015084885255836 | loss_stagnation_count = 0

Epoch:  16%|█▌        | 48/300 [02:23<12:44,  3.03s/it]2026-10-14 00:45:13.139 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 49, smoothed-train-loss = 0.020514 | smoothed-test-loss = 0.019848 | rel_loss_delta = 0.013075043815775312 | loss_stagnation_count = 0

Epoch:  16%|█▋        | 49/300 [02:26<12:39,  3.03s/it]2026-10-14 00:45:15.981 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 50, smoothed-train-loss = 0.020491 | smoothed-test-loss = 0.020500 | rel_loss_delta = -0.0011151631828865251 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 50/300 [02:29<12:22,  2.97s/it]2026-10-14 00:45:18.918 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 51, smoothed-train-loss = 0.020361 | smoothed-test-loss = 0.020640 | rel_loss_delta = -0.006357939883436476 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 51/300 [02:32<12:17,  2.96s/it]2026-10-14 00:45:21.895 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 52, smoothed-train-loss = 0.020262 | smoothed-test-loss = 0.019345 | rel_loss_delta = -0.00485123804506615 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 52/300 [02:35<12:15,  2.97s/it]2026-10-14 00:45:24.918 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 53, smoothed-train-loss = 0.020276 | smoothed-test-loss = 0.021068 | rel_loss_delta = 0.0007097020255609271 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 53/300 [02:38<12:16,  2.98s/it]2026-10-14 00:45:27.917 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 54, smoothed-train-loss = 0.020221 | smoothed-test-loss = 0.020693 | rel_loss_delta = -0.002733353998148449 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 54/300 [02:41<12:15,  2.99s/it]2026-10-14 00:45:30.898 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 55, smoothed-train-loss = 0.020515 | smoothed-test-loss = 0.019833 | rel_loss_delta = 0.01453209550427632 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 55/300 [02:44<12:11,  2.99s/it]2026-10-14 00:45:33.910 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 56, smoothed-train-loss = 0.020271 | smoothed-test-loss = 0.019250 | rel_loss_delta = -0.011890225036229319 | loss_stagnation_count = 0

Epoch:  19%|█▊        | 56/300 [02:47<12:10,  2.99s/it]2026-10-14 00:45:36.872 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 57, smoothed-train-loss = 0.020243 | smoothed-test-loss = 0.019260 | rel_loss_delta = -0.0013920305742375501 | loss_stagnation_count = 0

Epoch:  19%|█▉        | 57/300 [02:50<12:05,  2.98s/it]2026-10-14 00:45:39.859 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 58, smoothed-train-loss = 0.020197 | smoothed-test-loss = 0.019019 | rel_loss_delta = -0.0022750035318954975 | loss_stagnation_count = 0

Epoch:  19%|█▉        | 58/300 [02:53<12:02,  2.98s/it]2026-10-14 00:45:42.981 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 59, smoothed-train-loss = 0.020086 | smoothed-test-loss = 0.019346 | rel_loss_delta = -0.005461754715783643 | loss_stagnation_count = 0

Epoch:  20%|█▉        | 59/300 [02:56<12:09,  3.03s/it]2026-10-14 00:45:45.929 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 60, smoothed-train-loss = 0.020166 | smoothed-test-loss = 0.019541 | rel_loss_delta = 0.003954471055254297 | loss_stagnation_count = 0

Epoch:  20%|██        | 60/300 [02:59<12:00,  3.00s/it]2026-10-14 00:45:48.840 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 61, smoothed-train-loss = 0.020033 | smoothed-test-loss = 0.019770 | rel_loss_delta = -0.0065645154319502 | loss_stagnation_count = 0

Epoch:  20%|██        | 61/300 [03:02<11:51,  2.98s/it]2026-10-14 00:45:51.791 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 62, smoothed-train-loss = 0.020224 | smoothed-test-loss = 0.020245 | rel_loss_delta = 0.009519236868107944 | loss_stagnation_count = 0

Epoch:  21%|██        | 62/300 [03:05<11:46,  2.97s/it]2026-10-14 00:45:54.871 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 63, smoothed-train-loss = 0.020085 | smoothed-test-loss = 0.020081 | rel_loss_delta = -0.006893961627204307 | loss_stagnation_count = 0

Epoch:  21%|██        | 63/300 [03:08<11:51,  3.00s/it]2026-10-14 00:45:57.809 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 64, smoothed-train-loss = 0.020265 | smoothed-test-loss = 0.019396 | rel_loss_delta = 0.008986129830980714 | loss_stagnation_count = 0

Epoch:  21%|██▏       | 64/300 [03:11<11:43,  2.98s/it]2026-10-14 00:46:00.748 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 65, smoothed-train-loss = 0.019933 | smoothed-test-loss = 0.020032 | rel_loss_delta = -0.01639354158870315 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 65/300 [03:14<11:37,  2.97s/it]2026-10-14 00:46:03.556 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 66, smoothed-train-loss = 0.019998 | smoothed-test-loss = 0.019587 | rel_loss_delta = 0.003252015525873807 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 66/300 [03:17<11:23,  2.92s/it]2026-10-14 00:46:06.547 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 67, smoothed-train-loss = 0.019995 | smoothed-test-loss = 0.019015 | rel_loss_delta = -0.00012166579226788396 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 67/300 [03:20<11:25,  2.94s/it]2026-10-14 00:46:09.647 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 68, smoothed-train-loss = 0.020002 | smoothed-test-loss = 0.019843 | rel_loss_delta = 0.00032483104285724486 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 68/300 [03:23<11:33,  2.99s/it]2026-10-14 00:46:12.726 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 69, smoothed-train-loss = 0.020007 | smoothed-test-loss = 0.019405 | rel_loss_delta = 0.0002807447901299798 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 69/300 [03:26<11:36,  3.02s/it]2026-10-14 00:46:15.674 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 70, smoothed-train-loss = 0.019869 | smoothed-test-loss = 0.020900 | rel_loss_delta = -0.006938147142286511 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 70/300 [03:29<11:29,  3.00s/it]2026-10-14 00:46:18.638 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 71, smoothed-train-loss = 0.019986 | smoothed-test-loss = 0.019094 | rel_loss_delta = 0.005889417077107887 | loss_stagnation_count = 0

Epoch:  24%|██▎       | 71/300 [03:32<11:23,  2.99s/it]2026-10-14 00:46:21.735 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 72, smoothed-train-loss = 0.019910 | smoothed-test-loss = 0.019841 | rel_loss_delta = -0.0038038021542007258 | loss_stagnation_count = 0

Epoch:  24%|██▍       | 72/300 [03:35<11:28,  3.02s/it]2026-10-14 00:46:24.717 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 73, smoothed-train-loss = 0.019930 | smoothed-test-loss = 0.021027 | rel_loss_delta = 0.0010105962896196556 | loss_stagnation_count = 0

Epoch:  24%|██▍       | 73/300 [03:38<11:22,  3.01s/it]2026-10-14 00:46:27.683 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 74, smoothed-train-loss = 0.019923 | smoothed-test-loss = 0.019562 | rel_loss_delta = -0.0003335458601635716 | loss_stagnation_count = 0

Epoch:  25%|██▍       | 74/300 [03:41<11:16,  3.00s/it]2026-10-14 00:46:30.721 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 75, smoothed-train-loss = 0.019954 | smoothed-test-loss = 0.018858 | rel_loss_delta = 0.0015591370176314464 | loss_stagnation_count = 0

Epoch:  25%|██▌       | 75/300 [03:44<11:16,  3.01s/it]2026-10-14 00:46:33.755 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 76, smoothed-train-loss = 0.019732 | smoothed-test-loss = 0.018803 | rel_loss_delta = -0.011129120156624392 | loss_stagnation_count = 0

Epoch:  25%|██▌       | 76/300 [03:47<11:15,  3.02s/it]2026-10-14 00:46:36.790 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 77, smoothed-train-loss = 0.019822 | smoothed-test-loss = 0.022061 | rel_loss_delta = 0.004558297757088329 | loss_stagnation_count = 0

Epoch:  26%|██▌       | 77/300 [03:50<11:13,  3.02s/it]2026-10-14 00:46:39.737 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 78, smoothed-train-loss = 0.019880 | smoothed-test-loss = 0.019408 | rel_loss_delta = 0.0029259331663701376 | loss_stagnation_count = 0

Epoch:  26%|██▌       | 78/300 [03:53<11:05,  3.00s/it]2026-10-14 00:46:42.747 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 79, smoothed-train-loss = 0.019823 | smoothed-test-loss = 0.020545 | rel_loss_delta = -0.002864484454654951 | loss_stagnation_count = 0

Epoch:  26%|██▋       | 79/300 [03:56<11:03,  3.00s/it]2026-10-14 00:46:45.710 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 80, smoothed-train-loss = 0.019770 | smoothed-test-loss = 0.018717 | rel_loss_delta = -0.0026599411798502184 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 80/300 [03:59<10:57,  2.99s/it]2026-10-14 00:46:48.743 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 81, smoothed-train-loss = 0.019831 | smoothed-test-loss = 0.018793 | rel_loss_delta = 0.003075253469453441 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 81/300 [04:02<10:57,  3.00s/it]2026-10-14 00:46:51.628 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 82, smoothed-train-loss = 0.019706 | smoothed-test-loss = 0.019122 | rel_loss_delta = -0.006298769963131645 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 82/300 [04:05<10:46,  2.97s/it]2026-10-14 00:46:54.790 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 83, smoothed-train-loss = 0.019685 | smoothed-test-loss = 0.019165 | rel_loss_delta = -0.0010694444211318949 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 83/300 [04:08<10:56,  3.03s/it]2026-10-14 00:46:57.839 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 84, smoothed-train-loss = 0.019843 | smoothed-test-loss = 0.018684 | rel_loss_delta = 0.008018154710474292 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 84/300 [04:11<10:55,  3.03s/it]2026-10-14 00:47:00.943 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 85, smoothed-train-loss = 0.019842 | smoothed-test-loss = 0.018776 | rel_loss_delta = -3.810169455434769e-05 | loss_stagnation_count = 1

Epoch:  28%|██▊       | 85/300 [04:14<10:56,  3.05s/it]2026-10-14 00:47:03.784 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 86, smoothed-train-loss = 0.019652 | smoothed-test-loss = 0.019307 | rel_loss_delta = -0.009608496285814736 | loss_stagnation_count = 0

Epoch:  29%|██▊       | 86/300 [04:17<10:39,  2.99s/it]2026-10-14 00:47:06.819 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 87, smoothed-train-loss = 0.019702 | smoothed-test-loss = 0.019844 | rel_loss_delta = 0.002552199779369456 | loss_stagnation_count = 0

Epoch:  29%|██▉       | 87/300 [04:20<10:39,  3.00s/it]2026-10-14 00:47:09.893 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 88, smoothed-train-loss = 0.019710 | smoothed-test-loss = 0.019438 | rel_loss_delta = 0.0003992126647751566 | loss_stagnation_count = 0

Epoch:  29%|██▉       | 88/300 [04:23<10:41,  3.02s/it]2026-10-14 00:47:12.919 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 89, smoothed-train-loss = 0.019660 | smoothed-test-loss = 0.020223 | rel_loss_delta = -0.0025173039362418105 | loss_stagnation_count = 0

Epoch:  30%|██▉       | 89/300 [04:26<10:38,  3.03s/it]2026-10-14 00:47:15.822 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 90, smoothed-train-loss = 0.019799 | smoothed-test-loss = 0.019219 | rel_loss_delta = 0.007050606382924862 | loss_stagnation_count = 0

Epoch:  30%|███       | 90/300 [04:29<10:27,  2.99s/it]2026-10-14 00:47:18.823 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 91, smoothed-train-loss = 0.019623 | smoothed-test-loss = 0.018956 | rel_loss_delta = -0.008881490537837872 | loss_stagnation_count = 0

Epoch:  30%|███       | 91/300 [04:32<10:25,  2.99s/it]2026-10-14 00:47:21.821 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 92, smoothed-train-loss = 0.019792 | smoothed-test-loss = 0.019069 | rel_loss_delta = 0.008644930566007772 | loss_stagnation_count = 0

Epoch:  31%|███       | 92/300 [04:35<10:22,  2.99s/it]2026-10-14 00:47:24.986 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 93, smoothed-train-loss = 0.019758 | smoothed-test-loss = 0.018380 | rel_loss_delta = -0.0017158696122049376 | loss_stagnation_count = 0

Epoch:  31%|███       | 93/300 [04:38<10:30,  3.05s/it]2026-10-14 00:47:27.876 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 94, smoothed-train-loss = 0.019688 | smoothed-test-loss = 0.018992 | rel_loss_delta = -0.003543622725261302 | loss_stagnation_count = 0

Epoch:  31%|███▏      | 94/300 [04:41<10:17,  3.00s/it]2026-10-14 00:47:30.896 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 95, smoothed-train-loss = 0.019663 | smoothed-test-loss = 0.019508 | rel_loss_delta = -0.0013092214841389342 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 95/300 [04:44<10:16,  3.01s/it]2026-10-14 00:47:33.811 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 96, smoothed-train-loss = 0.019635 | smoothed-test-loss = 0.019616 | rel_loss_delta = -0.0014141312178519373 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 96/300 [04:47<10:07,  2.98s/it]2026-10-14 00:47:36.779 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 97, smoothed-train-loss = 0.019604 | smoothed-test-loss = 0.019203 | rel_loss_delta = -0.001543158926842363 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 97/300 [04:50<10:03,  2.98s/it]2026-10-14 00:47:39.823 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 98, smoothed-train-loss = 0.019604 | smoothed-test-loss = 0.019469 | rel_loss_delta = -1.564035450805719e-05 | loss_stagnation_count = 1

Epoch:  33%|███▎      | 98/300 [04:53<10:05,  3.00s/it]2026-10-14 00:47:42.814 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 99, smoothed-train-loss = 0.019674 | smoothed-test-loss = 0.019254 | rel_loss_delta = 0.0035824221202016364 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 99/300 [04:56<10:01,  2.99s/it]2026-10-14 00:47:45.792 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 100, smoothed-train-loss = 0.019623 | smoothed-test-loss = 0.019777 | rel_loss_delta = -0.0026003451885265576 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 100/300 [04:59<09:57,  2.99s/it]2026-10-14 00:47:48.787 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 101, smoothed-train-loss = 0.019641 | smoothed-test-loss = 0.018627 | rel_loss_delta = 0.000917918115672575 | loss_stagnation_count = 0

Epoch:  34%|███▎      | 101/300 [05:02<09:55,  2.99s/it]2026-10-14 00:47:51.665 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 102, smoothed-train-loss = 0.019629 | smoothed-test-loss = 0.018339 | rel_loss_delta = -0.0006032744391331467 | loss_stagnation_count = 0

Epoch:  34%|███▍      | 102/300 [05:05<09:45,  2.96s/it]2026-10-14 00:47:54.629 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 103, smoothed-train-loss = 0.019532 | smoothed-test-loss = 0.018812 | rel_loss_delta = -0.004956038116885952 | loss_stagnation_count = 0

Epoch:  34%|███▍      | 103/300 [05:08<09:42,  2.96s/it]2026-10-14 00:47:57.669 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 104, smoothed-train-loss = 0.019683 | smoothed-test-loss = 0.018932 | rel_loss_delta = 0.007717461470376894 | loss_stagnation_count = 0

Epoch:  35%|███▍      | 104/300 [05:11<09:44,  2.98s/it]2026-10-14 00:48:00.649 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 105, smoothed-train-loss = 0.019583 | smoothed-test-loss = 0.019717 | rel_loss_delta = -0.005082716229386594 | loss_stagnation_count = 0

Epoch:  35%|███▌      | 105/300 [05:14<09:41,  2.98s/it]2026-10-14 00:48:03.535 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 106, smoothed-train-loss = 0.019483 | smoothed-test-loss = 0.018607 | rel_loss_delta = -0.005082682620328082 | loss_stagnation_count = 0

Epoch:  35%|███▌      | 106/300 [05:17<09:33,  2.95s/it]2026-10-14 00:48:06.524 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 107, smoothed-train-loss = 0.019690 | smoothed-test-loss = 0.019086 | rel_loss_delta = 0.010606591680155555 | loss_stagnation_count = 0

Epoch:  36%|███▌      | 107/300 [05:20<09:32,  2.96s/it]2026-10-14 00:48:09.525 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 108, smoothed-train-loss = 0.019494 | smoothed-test-loss = 0.018482 | rel_loss_delta = -0.009961282529577864 | loss_stagnation_count = 0

Epoch:  36%|███▌      | 108/300 [05:23<09:31,  2.98s/it]2026-10-14 00:48:12.549 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 109, smoothed-train-loss = 0.019408 | smoothed-test-loss = 0.019108 | rel_loss_delta = -0.004412394657903236 | loss_stagnation_count = 0

Epoch:  36%|███▋      | 109/300 [05:26<09:31,  2.99s/it]2026-10-14 00:48:15.459 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 110, smoothed-train-loss = 0.019522 | smoothed-test-loss = 0.018385 | rel_loss_delta = 0.005860085721699397 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 110/300 [05:29<09:23,  2.97s/it]2026-10-14 00:48:18.431 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 111, smoothed-train-loss = 0.019458 | smoothed-test-loss = 0.019073 | rel_loss_delta = -0.003238894253242867 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 111/300 [05:32<09:20,  2.97s/it]2026-10-14 00:48:21.449 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 112, smoothed-train-loss = 0.019604 | smoothed-test-loss = 0.019617 | rel_loss_delta = 0.007506809333858364 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 112/300 [05:35<09:20,  2.98s/it]2026-10-14 00:48:24.652 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 113, smoothed-train-loss = 0.019523 | smoothed-test-loss = 0.018610 | rel_loss_delta = -0.004158062959366091 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 113/300 [05:38<09:30,  3.05s/it]2026-10-14 00:48:27.591 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 114, smoothed-train-loss = 0.019528 | smoothed-test-loss = 0.019065 | rel_loss_delta = 0.00024019791974785645 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 114/300 [05:41<09:20,  3.02s/it]2026-10-14 00:48:30.636 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 115, smoothed-train-loss = 0.019249 | smoothed-test-loss = 0.019252 | rel_loss_delta = -0.014255816370148172 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 115/300 [05:44<09:19,  3.02s/it]2026-10-14 00:48:33.611 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 116, smoothed-train-loss = 0.019302 | smoothed-test-loss = 0.018321 | rel_loss_delta = 0.002740260793230404 | loss_stagnation_count = 0

Epoch:  39%|███▊      | 116/300 [05:47<09:13,  3.01s/it]2026-10-14 00:48:36.551 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 117, smoothed-train-loss = 0.019406 | smoothed-test-loss = 0.019240 | rel_loss_delta = 0.0053807460483759105 | loss_stagnation_count = 0

Epoch:  39%|███▉      | 117/300 [05:50<09:06,  2.99s/it]2026-10-14 00:48:39.460 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 118, smoothed-train-loss = 0.019453 | smoothed-test-loss = 0.018218 | rel_loss_delta = 0.0024344387563363505 | loss_stagnation_count = 0

Epoch:  39%|███▉      | 118/300 [05:53<08:59,  2.96s/it]2026-10-14 00:48:42.564 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 119, smoothed-train-loss = 0.019224 | smoothed-test-loss = 0.018907 | rel_loss_delta = -0.011754878182130654 | loss_stagnation_count = 0

Epoch:  40%|███▉      | 119/300 [05:56<09:04,  3.01s/it]2026-10-14 00:48:45.530 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 120, smoothed-train-loss = 0.019414 | smoothed-test-loss = 0.018480 | rel_loss_delta = 0.009889134491925078 | loss_stagnation_count = 0

Epoch:  40%|████      | 120/300 [05:59<08:59,  2.99s/it]2026-10-14 00:48:48.500 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 121, smoothed-train-loss = 0.019242 | smoothed-test-loss = 0.018401 | rel_loss_delta = -0.008898489183538954 | loss_stagnation_count = 0

Epoch:  40%|████      | 121/300 [06:02<08:54,  2.99s/it]2026-10-14 00:48:51.481 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 122, smoothed-train-loss = 0.019440 | smoothed-test-loss = 0.018244 | rel_loss_delta = 0.010286441170055107 | loss_stagnation_count = 0

Epoch:  41%|████      | 122/300 [06:05<08:51,  2.99s/it]2026-10-14 00:48:54.556 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 123, smoothed-train-loss = 0.019178 | smoothed-test-loss = 0.018905 | rel_loss_delta = -0.013468875367354409 | loss_stagnation_count = 0

Epoch:  41%|████      | 123/300 [06:08<08:53,  3.01s/it]2026-10-14 00:48:57.477 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 124, smoothed-train-loss = 0.019393 | smoothed-test-loss = 0.020222 | rel_loss_delta = 0.011218619417271893 | loss_stagnation_count = 0

Epoch:  41%|████▏     | 124/300 [06:11<08:45,  2.98s/it]2026-10-14 00:49:00.412 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 125, smoothed-train-loss = 0.019305 | smoothed-test-loss = 0.018453 | rel_loss_delta = -0.0045267642988815115 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 125/300 [06:14<08:39,  2.97s/it]2026-10-14 00:49:03.326 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 126, smoothed-train-loss = 0.019140 | smoothed-test-loss = 0.018542 | rel_loss_delta = -0.0085504229864846 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 126/300 [06:16<08:33,  2.95s/it]2026-10-14 00:49:06.500 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 127, smoothed-train-loss = 0.019231 | smoothed-test-loss = 0.019203 | rel_loss_delta = 0.004766084737710685 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 127/300 [06:20<08:42,  3.02s/it]2026-10-14 00:49:09.529 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 128, smoothed-train-loss = 0.019227 | smoothed-test-loss = 0.019520 | rel_loss_delta = -0.0002103966896080156 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 128/300 [06:23<08:39,  3.02s/it]2026-10-14 00:49:12.610 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 129, smoothed-train-loss = 0.019320 | smoothed-test-loss = 0.019249 | rel_loss_delta = 0.004818852016925797 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 129/300 [06:26<08:39,  3.04s/it]2026-10-14 00:49:15.731 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 130, smoothed-train-loss = 0.019124 | smoothed-test-loss = 0.018184 | rel_loss_delta = -0.01012358439654329 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 130/300 [06:29<08:40,  3.06s/it]2026-10-14 00:49:19.001 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 131, smoothed-train-loss = 0.019400 | smoothed-test-loss = 0.019967 | rel_loss_delta = 0.014429542283121046 | loss_stagnation_count = 0

Epoch:  44%|████▎     | 131/300 [06:32<08:48,  3.13s/it]2026-10-14 00:49:22.179 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 132, smoothed-train-loss = 0.019095 | smoothed-test-loss = 0.018555 | rel_loss_delta = -0.015735287543396967 | loss_stagnation_count = 0

Epoch:  44%|████▍     | 132/300 [06:35<08:47,  3.14s/it]2026-10-14 00:49:25.251 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 133, smoothed-train-loss = 0.019162 | smoothed-test-loss = 0.018819 | rel_loss_delta = 0.0035226973396811595 | loss_stagnation_count = 0

Epoch:  44%|████▍     | 133/300 [06:38<08:41,  3.12s/it]2026-10-14 00:49:28.294 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 134, smoothed-train-loss = 0.019314 | smoothed-test-loss = 0.018249 | rel_loss_delta = 0.007936930310851865 | loss_stagnation_count = 0

Epoch:  45%|████▍     | 134/300 [06:41<08:34,  3.10s/it]2026-10-14 00:49:31.623 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 135, smoothed-train-loss = 0.019016 | smoothed-test-loss = 0.018300 | rel_loss_delta = -0.015425788753402487 | loss_stagnation_count = 0

Epoch:  45%|████▌     | 135/300 [06:45<08:42,  3.17s/it]2026-10-14 00:49:35.668 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 136, smoothed-train-loss = 0.019057 | smoothed-test-loss = 0.018049 | rel_loss_delta = 0.0021576590633791658 | loss_stagnation_count = 0

Epoch:  45%|████▌     | 136/300 [06:49<09:22,  3.43s/it]2026-10-14 00:49:39.560 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 137, smoothed-train-loss = 0.019269 | smoothed-test-loss = 0.018385 | rel_loss_delta = 0.011091831433616162 | loss_stagnation_count = 0

Epoch:  46%|████▌     | 137/300 [06:53<09:41,  3.57s/it]2026-10-14 00:49:42.745 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 138, smoothed-train-loss = 0.019199 | smoothed-test-loss = 0.017989 | rel_loss_delta = -0.0036273522141578884 | loss_stagnation_count = 0

Epoch:  46%|████▌     | 138/300 [06:56<09:19,  3.45s/it]2026-10-14 00:49:46.310 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 139, smoothed-train-loss = 0.019154 | smoothed-test-loss = 0.018320 | rel_loss_delta = -0.002324995347381658 | loss_stagnation_count = 0

Epoch:  46%|████▋     | 139/300 [06:59<09:21,  3.49s/it]2026-10-14 00:49:49.538 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 140, smoothed-train-loss = 0.019084 | smoothed-test-loss = 0.018401 | rel_loss_delta = -0.003660756539617329 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 140/300 [07:03<09:05,  3.41s/it]2026-10-14 00:49:53.246 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 141, smoothed-train-loss = 0.019138 | smoothed-test-loss = 0.019805 | rel_loss_delta = 0.0028348872463960736 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 141/300 [07:06<09:16,  3.50s/it]2026-10-14 00:49:56.363 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 142, smoothed-train-loss = 0.019226 | smoothed-test-loss = 0.018271 | rel_loss_delta = 0.004567471021711329 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 142/300 [07:09<08:54,  3.38s/it]2026-10-14 00:49:59.674 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 143, smoothed-train-loss = 0.019083 | smoothed-test-loss = 0.018364 | rel_loss_delta = -0.007402601381306378 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 143/300 [07:13<08:47,  3.36s/it]2026-10-14 00:50:02.801 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 144, smoothed-train-loss = 0.019215 | smoothed-test-loss = 0.019790 | rel_loss_delta = 0.006907806531747935 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 144/300 [07:16<08:33,  3.29s/it]2026-10-14 00:50:05.845 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 145, smoothed-train-loss = 0.019099 | smoothed-test-loss = 0.018334 | rel_loss_delta = -0.006022174049890881 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 145/300 [07:19<08:18,  3.22s/it]2026-10-14 00:50:08.877 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 146, smoothed-train-loss = 0.018995 | smoothed-test-loss = 0.018257 | rel_loss_delta = -0.005446685730701028 | loss_stagnation_count = 0

Epoch:  49%|████▊     | 146/300 [07:22<08:06,  3.16s/it]2026-10-14 00:50:12.319 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 147, smoothed-train-loss = 0.018998 | smoothed-test-loss = 0.018115 | rel_loss_delta = 0.00012437530212148237 | loss_stagnation_count = 0

Epoch:  49%|████▉     | 147/300 [07:25<08:16,  3.25s/it]2026-10-14 00:50:15.564 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 148, smoothed-train-loss = 0.019198 | smoothed-test-loss = 0.019294 | rel_loss_delta = 0.010512574993970611 | loss_stagnation_count = 0

Epoch:  49%|████▉     | 148/300 [07:29<08:13,  3.25s/it]2026-10-14 00:50:18.934 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 149, smoothed-train-loss = 0.019029 | smoothed-test-loss = 0.019156 | rel_loss_delta = -0.00879667134027135 | loss_stagnation_count = 0

Epoch:  50%|████▉     | 149/300 [07:32<08:15,  3.28s/it]2026-10-14 00:50:21.979 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 150, smoothed-train-loss = 0.019216 | smoothed-test-loss = 0.018117 | rel_loss_delta = 0.009863025450802507 | loss_stagnation_count = 0

Epoch:  50%|█████     | 150/300 [07:35<08:01,  3.21s/it]2026-10-14 00:50:25.043 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 151, smoothed-train-loss = 0.018973 | smoothed-test-loss = 0.018549 | rel_loss_delta = -0.012646754841384392 | loss_stagnation_count = 0

Epoch:  50%|█████     | 151/300 [07:38<07:51,  3.17s/it]2026-10-14 00:50:28.290 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 152, smoothed-train-loss = 0.019158 | smoothed-test-loss = 0.019612 | rel_loss_delta = 0.009737294465793246 | loss_stagnation_count = 0

Epoch:  51%|█████     | 152/300 [07:41<07:52,  3.19s/it]2026-10-14 00:50:31.664 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 153, smoothed-train-loss = 0.018915 | smoothed-test-loss = 0.018330 | rel_loss_delta = -0.012695647704269158 | loss_stagnation_count = 0

Epoch:  51%|█████     | 153/300 [07:45<07:57,  3.25s/it]2026-10-14 00:50:34.825 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 154, smoothed-train-loss = 0.018978 | smoothed-test-loss = 0.019408 | rel_loss_delta = 0.0033153205340538078 | loss_stagnation_count = 0

Epoch:  51%|█████▏    | 154/300 [07:48<07:50,  3.22s/it]2026-10-14 00:50:38.091 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 155, smoothed-train-loss = 0.019014 | smoothed-test-loss = 0.018544 | rel_loss_delta = 0.0018957812346051606 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 155/300 [07:51<07:48,  3.23s/it]2026-10-14 00:50:41.255 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 156, smoothed-train-loss = 0.019164 | smoothed-test-loss = 0.018124 | rel_loss_delta = 0.007902229381389498 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 156/300 [07:54<07:42,  3.21s/it]2026-10-14 00:50:44.505 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 157, smoothed-train-loss = 0.018968 | smoothed-test-loss = 0.018128 | rel_loss_delta = -0.010200772081562738 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 157/300 [07:58<07:41,  3.22s/it]2026-10-14 00:50:47.604 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 158, smoothed-train-loss = 0.019039 | smoothed-test-loss = 0.018654 | rel_loss_delta = 0.0037520189962128477 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 158/300 [08:01<07:32,  3.19s/it]2026-10-14 00:50:50.719 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 159, smoothed-train-loss = 0.019119 | smoothed-test-loss = 0.018175 | rel_loss_delta = 0.004202979215819217 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 159/300 [08:04<07:26,  3.17s/it]2026-10-14 00:50:54.223 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 160, smoothed-train-loss = 0.019044 | smoothed-test-loss = 0.018767 | rel_loss_delta = -0.003954093467935981 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 160/300 [08:07<07:37,  3.27s/it]2026-10-14 00:50:57.574 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 161, smoothed-train-loss = 0.019032 | smoothed-test-loss = 0.018709 | rel_loss_delta = -0.0006247966561437593 | loss_stagnation_count = 0

Epoch:  54%|█████▎    | 161/300 [08:11<07:37,  3.29s/it]2026-10-14 00:51:00.526 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 162, smoothed-train-loss = 0.018957 | smoothed-test-loss = 0.017848 | rel_loss_delta = -0.003929606587425739 | loss_stagnation_count = 0

Epoch:  54%|█████▍    | 162/300 [08:14<07:20,  3.19s/it]2026-10-14 00:51:03.533 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 163, smoothed-train-loss = 0.019010 | smoothed-test-loss = 0.020348 | rel_loss_delta = 0.002772935105150063 | loss_stagnation_count = 0

Epoch:  54%|█████▍    | 163/300 [08:17<07:09,  3.14s/it]2026-10-14 00:51:06.469 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 164, smoothed-train-loss = 0.018955 | smoothed-test-loss = 0.018919 | rel_loss_delta = -0.0028635779096904523 | loss_stagnation_count = 0

Epoch:  55%|█████▍    | 164/300 [08:20<06:58,  3.08s/it]2026-10-14 00:51:09.581 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 165, smoothed-train-loss = 0.019288 | smoothed-test-loss = 0.018236 | rel_loss_delta = 0.01753147471484468 | loss_stagnation_count = 0

Epoch:  55%|█████▌    | 165/300 [08:23<06:56,  3.09s/it]2026-10-14 00:51:12.424 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 166, smoothed-train-loss = 0.018926 | smoothed-test-loss = 0.018102 | rel_loss_delta = -0.01874157906506615 | loss_stagnation_count = 0

Epoch:  55%|█████▌    | 166/300 [08:26<06:43,  3.01s/it]2026-10-14 00:51:15.351 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 167, smoothed-train-loss = 0.018963 | smoothed-test-loss = 0.018482 | rel_loss_delta = 0.0019677441460360175 | loss_stagnation_count = 0

Epoch:  56%|█████▌    | 167/300 [08:28<06:37,  2.99s/it]2026-10-14 00:51:18.540 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 168, smoothed-train-loss = 0.018976 | smoothed-test-loss = 0.018379 | rel_loss_delta = 0.0006671448816598914 | loss_stagnation_count = 0

Epoch:  56%|█████▌    | 168/300 [08:32<06:42,  3.05s/it]2026-10-14 00:51:21.646 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 169, smoothed-train-loss = 0.019087 | smoothed-test-loss = 0.017951 | rel_loss_delta = 0.005865524900100339 | loss_stagnation_count = 0

Epoch:  56%|█████▋    | 169/300 [08:35<06:41,  3.07s/it]2026-10-14 00:51:24.557 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 170, smoothed-train-loss = 0.018892 | smoothed-test-loss = 0.018441 | rel_loss_delta = -0.010239180738641888 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 170/300 [08:38<06:32,  3.02s/it]2026-10-14 00:51:27.641 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 171, smoothed-train-loss = 0.018972 | smoothed-test-loss = 0.019231 | rel_loss_delta = 0.004242454741355659 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 171/300 [08:41<06:31,  3.04s/it]2026-10-14 00:51:30.697 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 172, smoothed-train-loss = 0.018963 | smoothed-test-loss = 0.017949 | rel_loss_delta = -0.0005039031070418597 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 172/300 [08:44<06:29,  3.04s/it]2026-10-14 00:51:33.805 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 173, smoothed-train-loss = 0.018941 | smoothed-test-loss = 0.018916 | rel_loss_delta = -0.0011474296085548182 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 173/300 [08:47<06:29,  3.06s/it]2026-10-14 00:51:36.895 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 174, smoothed-train-loss = 0.018995 | smoothed-test-loss = 0.018622 | rel_loss_delta = 0.0028613323350086327 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 174/300 [08:50<06:26,  3.07s/it]2026-10-14 00:51:40.025 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 175, smoothed-train-loss = 0.018925 | smoothed-test-loss = 0.019271 | rel_loss_delta = -0.0036932944268188153 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 175/300 [08:53<06:26,  3.09s/it]2026-10-14 00:51:43.252 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 176, smoothed-train-loss = 0.018904 | smoothed-test-loss = 0.018309 | rel_loss_delta = -0.0011184345095104336 | loss_stagnation_count = 0

Epoch:  59%|█████▊    | 176/300 [08:56<06:28,  3.13s/it]2026-10-14 00:51:46.213 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 177, smoothed-train-loss = 0.018956 | smoothed-test-loss = 0.018089 | rel_loss_delta = 0.0027739025653964566 | loss_stagnation_count = 0

Epoch:  59%|█████▉    | 177/300 [08:59<06:18,  3.08s/it]2026-10-14 00:51:49.242 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 178, smoothed-train-loss = 0.018963 | smoothed-test-loss = 0.018958 | rel_loss_delta = 0.0003600636635559866 | loss_stagnation_count = 0

Epoch:  59%|█████▉    | 178/300 [09:02<06:13,  3.06s/it]2026-10-14 00:51:52.465 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 179, smoothed-train-loss = 0.018919 | smoothed-test-loss = 0.018187 | rel_loss_delta = -0.0023097966660908438 | loss_stagnation_count = 0

Epoch:  60%|█████▉    | 179/300 [09:06<06:16,  3.11s/it]2026-10-14 00:51:55.742 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 180, smoothed-train-loss = 0.018974 | smoothed-test-loss = 0.018382 | rel_loss_delta = 0.0029026079235452097 | loss_stagnation_count = 0

Epoch:  60%|██████    | 180/300 [09:09<06:19,  3.16s/it]2026-10-14 00:51:59.259 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 181, smoothed-train-loss = 0.019002 | smoothed-test-loss = 0.019546 | rel_loss_delta = 0.0014938850892348805 | loss_stagnation_count = 0

Epoch:  60%|██████    | 181/300 [09:12<06:28,  3.27s/it]2026-10-14 00:52:02.362 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 182, smoothed-train-loss = 0.018927 | smoothed-test-loss = 0.017746 | rel_loss_delta = -0.0039567500794818805 | loss_stagnation_count = 0

Epoch:  61%|██████    | 182/300 [09:15<06:19,  3.22s/it]2026-10-14 00:52:05.716 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 183, smoothed-train-loss = 0.018925 | smoothed-test-loss = 0.017951 | rel_loss_delta = -0.00012298189913395466 | loss_stagnation_count = 0

Epoch:  61%|██████    | 183/300 [09:19<06:21,  3.26s/it]2026-10-14 00:52:08.841 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 184, smoothed-train-loss = 0.018818 | smoothed-test-loss = 0.018031 | rel_loss_delta = -0.005621098086180512 | loss_stagnation_count = 0

Epoch:  61%|██████▏   | 184/300 [09:22<06:13,  3.22s/it]2026-10-14 00:52:11.994 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 185, smoothed-train-loss = 0.018956 | smoothed-test-loss = 0.018941 | rel_loss_delta = 0.0073303413347545815 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 185/300 [09:25<06:07,  3.20s/it]2026-10-14 00:52:15.143 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 186, smoothed-train-loss = 0.018853 | smoothed-test-loss = 0.017832 | rel_loss_delta = -0.00545023808327188 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 186/300 [09:28<06:02,  3.18s/it]2026-10-14 00:52:18.130 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 187, smoothed-train-loss = 0.019097 | smoothed-test-loss = 0.018537 | rel_loss_delta = 0.01291166870177481 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 187/300 [09:31<05:53,  3.12s/it]2026-10-14 00:52:21.329 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 188, smoothed-train-loss = 0.018891 | smoothed-test-loss = 0.017711 | rel_loss_delta = -0.010774054213307708 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 188/300 [09:34<05:52,  3.15s/it]2026-10-14 00:52:24.701 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 189, smoothed-train-loss = 0.018811 | smoothed-test-loss = 0.017714 | rel_loss_delta = -0.00424717852197834 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 189/300 [09:38<05:56,  3.21s/it]2026-10-14 00:52:27.916 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 190, smoothed-train-loss = 0.018966 | smoothed-test-loss = 0.020448 | rel_loss_delta = 0.008266419846000943 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 190/300 [09:41<05:53,  3.21s/it]2026-10-14 00:52:31.001 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 191, smoothed-train-loss = 0.018764 | smoothed-test-loss = 0.018308 | rel_loss_delta = -0.01064678284043849 | loss_stagnation_count = 0

Epoch:  64%|██████▎   | 191/300 [09:44<05:46,  3.18s/it]2026-10-14 00:52:34.230 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 192, smoothed-train-loss = 0.018946 | smoothed-test-loss = 0.018437 | rel_loss_delta = 0.00968285932982505 | loss_stagnation_count = 0

Epoch:  64%|██████▍   | 192/300 [09:47<05:44,  3.19s/it]2026-10-14 00:52:37.470 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 193, smoothed-train-loss = 0.018847 | smoothed-test-loss = 0.017946 | rel_loss_delta = -0.0051943986502343805 | loss_stagnation_count = 0

Epoch:  64%|██████▍   | 193/300 [09:51<05:43,  3.21s/it]2026-10-14 00:52:40.464 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 194, smoothed-train-loss = 0.018981 | smoothed-test-loss = 0.018564 | rel_loss_delta = 0.007108825892982044 | loss_stagnation_count = 0

Epoch:  65%|██████▍   | 194/300 [09:54<05:33,  3.14s/it]2026-10-14 00:52:43.423 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 195, smoothed-train-loss = 0.018881 | smoothed-test-loss = 0.019007 | rel_loss_delta = -0.0052692882102279475 | loss_stagnation_count = 0

Epoch:  65%|██████▌   | 195/300 [09:57<05:24,  3.09s/it]2026-10-14 00:52:46.416 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 196, smoothed-train-loss = 0.018962 | smoothed-test-loss = 0.018787 | rel_loss_delta = 0.004252455592502093 | loss_stagnation_count = 0

Epoch:  65%|██████▌   | 196/300 [10:00<05:18,  3.06s/it]2026-10-14 00:52:49.677 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 197, smoothed-train-loss = 0.018928 | smoothed-test-loss = 0.018372 | rel_loss_delta = -0.0017959359482087765 | loss_stagnation_count = 0

Epoch:  66%|██████▌   | 197/300 [10:03<05:21,  3.12s/it]2026-10-14 00:52:52.878 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 198, smoothed-train-loss = 0.018849 | smoothed-test-loss = 0.018597 | rel_loss_delta = -0.004141117772410321 | loss_stagnation_count = 0

Epoch:  66%|██████▌   | 198/300 [10:06<05:20,  3.14s/it]2026-10-14 00:52:56.069 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 199, smoothed-train-loss = 0.018933 | smoothed-test-loss = 0.017632 | rel_loss_delta = 0.004471276289870796 | loss_stagnation_count = 0

Epoch:  66%|██████▋   | 199/300 [10:09<05:18,  3.16s/it]2026-10-14 00:52:59.147 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 200, smoothed-train-loss = 0.018821 | smoothed-test-loss = 0.017669 | rel_loss_delta = -0.005932821776901059 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 200/300 [10:12<05:13,  3.13s/it]2026-10-14 00:53:02.079 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 201, smoothed-train-loss = 0.018861 | smoothed-test-loss = 0.017656 | rel_loss_delta = 0.0021179203155770603 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 201/300 [10:15<05:04,  3.07s/it]2026-10-14 00:53:05.033 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 202, smoothed-train-loss = 0.018800 | smoothed-test-loss = 0.018024 | rel_loss_delta = -0.003213939797949229 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 202/300 [10:18<04:57,  3.04s/it]2026-10-14 00:53:08.187 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 203, smoothed-train-loss = 0.018897 | smoothed-test-loss = 0.018508 | rel_loss_delta = 0.00513587190667794 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 203/300 [10:21<04:58,  3.07s/it]2026-10-14 00:53:11.473 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 204, smoothed-train-loss = 0.018879 | smoothed-test-loss = 0.018680 | rel_loss_delta = -0.000960964181263189 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 204/300 [10:25<05:01,  3.14s/it]2026-10-14 00:53:14.752 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 205, smoothed-train-loss = 0.018850 | smoothed-test-loss = 0.018739 | rel_loss_delta = -0.0015148435160802444 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 205/300 [10:28<05:02,  3.18s/it]2026-10-14 00:53:17.839 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 206, smoothed-train-loss = 0.018827 | smoothed-test-loss = 0.018079 | rel_loss_delta = -0.0012244450674573355 | loss_stagnation_count = 0

Epoch:  69%|██████▊   | 206/300 [10:31<04:56,  3.15s/it]2026-10-14 00:53:20.842 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 207, smoothed-train-loss = 0.018804 | smoothed-test-loss = 0.018168 | rel_loss_delta = -0.0012492346388552626 | loss_stagnation_count = 0

Epoch:  69%|██████▉   | 207/300 [10:34<04:48,  3.11s/it]2026-10-14 00:53:24.062 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 208, smoothed-train-loss = 0.018934 | smoothed-test-loss = 0.017697 | rel_loss_delta = 0.00694046686507561 | loss_stagnation_count = 0

Epoch:  69%|██████▉   | 208/300 [10:37<04:48,  3.14s/it]2026-10-14 00:53:27.121 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 209, smoothed-train-loss = 0.018869 | smoothed-test-loss = 0.017923 | rel_loss_delta = -0.0034421181327544117 | loss_stagnation_count = 0

Epoch:  70%|██████▉   | 209/300 [10:40<04:43,  3.12s/it]2026-10-14 00:53:29.943 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 210, smoothed-train-loss = 0.018871 | smoothed-test-loss = 0.018255 | rel_loss_delta = 0.00013267257022489957 | loss_stagnation_count = 0

Epoch:  70%|███████   | 210/300 [10:43<04:32,  3.03s/it]2026-10-14 00:53:33.027 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 211, smoothed-train-loss = 0.018840 | smoothed-test-loss = 0.018291 | rel_loss_delta = -0.0016899853647837878 | loss_stagnation_count = 0

Epoch:  70%|███████   | 211/300 [10:46<04:30,  3.04s/it]2026-10-14 00:53:35.993 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 212, smoothed-train-loss = 0.018845 | smoothed-test-loss = 0.019484 | rel_loss_delta = 0.0002966616597511214 | loss_stagnation_count = 0

Epoch:  71%|███████   | 212/300 [10:49<04:25,  3.02s/it]2026-10-14 00:53:38.953 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 213, smoothed-train-loss = 0.018891 | smoothed-test-loss = 0.018392 | rel_loss_delta = 0.002426153833775562 | loss_stagnation_count = 0

Epoch:  71%|███████   | 213/300 [10:52<04:21,  3.00s/it]2026-10-14 00:53:41.802 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 214, smoothed-train-loss = 0.018908 | smoothed-test-loss = 0.018612 | rel_loss_delta = 0.0009220161622191281 | loss_stagnation_count = 0

Epoch:  71%|███████▏  | 214/300 [10:55<04:14,  2.96s/it]2026-10-14 00:53:44.906 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 215, smoothed-train-loss = 0.018793 | smoothed-test-loss = 0.018060 | rel_loss_delta = -0.00610607460805116 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 215/300 [10:58<04:15,  3.00s/it]2026-10-14 00:53:47.882 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 216, smoothed-train-loss = 0.018973 | smoothed-test-loss = 0.018039 | rel_loss_delta = 0.009603936905820998 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 216/300 [11:01<04:11,  2.99s/it]2026-10-14 00:53:50.932 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 217, smoothed-train-loss = 0.018779 | smoothed-test-loss = 0.017701 | rel_loss_delta = -0.01021895233796567 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 217/300 [11:04<04:09,  3.01s/it]2026-10-14 00:53:53.772 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 218, smoothed-train-loss = 0.018836 | smoothed-test-loss = 0.018537 | rel_loss_delta = 0.003004422699974984 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 218/300 [11:07<04:02,  2.96s/it]2026-10-14 00:53:56.799 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 219, smoothed-train-loss = 0.018809 | smoothed-test-loss = 0.019327 | rel_loss_delta = -0.001433215551679383 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 219/300 [11:10<04:01,  2.98s/it]2026-10-14 00:53:59.829 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 220, smoothed-train-loss = 0.018858 | smoothed-test-loss = 0.018135 | rel_loss_delta = 0.002623044678686733 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 220/300 [11:13<03:59,  2.99s/it]2026-10-14 00:54:02.831 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 221, smoothed-train-loss = 0.018761 | smoothed-test-loss = 0.018529 | rel_loss_delta = -0.0051772524938339184 | loss_stagnation_count = 0

Epoch:  74%|███████▎  | 221/300 [11:16<03:56,  3.00s/it]2026-10-14 00:54:05.914 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 222, smoothed-train-loss = 0.018901 | smoothed-test-loss = 0.018379 | rel_loss_delta = 0.007489693212819903 | loss_stagnation_count = 0

Epoch:  74%|███████▍  | 222/300 [11:19<03:55,  3.02s/it]2026-10-14 00:54:08.926 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 223, smoothed-train-loss = 0.018778 | smoothed-test-loss = 0.018139 | rel_loss_delta = -0.006533799700772413 | loss_stagnation_count = 0

Epoch:  74%|███████▍  | 223/300 [11:22<03:52,  3.02s/it]2026-10-14 00:54:12.255 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 224, smoothed-train-loss = 0.018883 | smoothed-test-loss = 0.018832 | rel_loss_delta = 0.005639080133185407 | loss_stagnation_count = 0

Epoch:  75%|███████▍  | 224/300 [11:25<03:56,  3.11s/it]2026-10-14 00:54:15.249 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 225, smoothed-train-loss = 0.018800 | smoothed-test-loss = 0.018779 | rel_loss_delta = -0.004393478562003586 | loss_stagnation_count = 0

Epoch:  75%|███████▌  | 225/300 [11:28<03:50,  3.08s/it]2026-10-14 00:54:18.441 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 226, smoothed-train-loss = 0.018882 | smoothed-test-loss = 0.018137 | rel_loss_delta = 0.004318048443285366 | loss_stagnation_count = 0

Epoch:  75%|███████▌  | 226/300 [11:32<03:50,  3.11s/it]2026-10-14 00:54:21.496 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 227, smoothed-train-loss = 0.018835 | smoothed-test-loss = 0.018020 | rel_loss_delta = -0.0024874964559073584 | loss_stagnation_count = 0

Epoch:  76%|███████▌  | 227/300 [11:35<03:45,  3.09s/it]2026-10-14 00:54:24.483 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 228, smoothed-train-loss = 0.018876 | smoothed-test-loss = 0.017882 | rel_loss_delta = 0.002198774572463464 | loss_stagnation_count = 0

Epoch:  76%|███████▌  | 228/300 [11:38<03:40,  3.06s/it]2026-10-14 00:54:27.432 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 229, smoothed-train-loss = 0.018809 | smoothed-test-loss = 0.018092 | rel_loss_delta = -0.0035702981395258184 | loss_stagnation_count = 0

Epoch:  76%|███████▋  | 229/300 [11:41<03:34,  3.03s/it]2026-10-14 00:54:30.307 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 230, smoothed-train-loss = 0.018864 | smoothed-test-loss = 0.018137 | rel_loss_delta = 0.0029631363941441326 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 230/300 [11:43<03:28,  2.98s/it]2026-10-14 00:54:33.255 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 231, smoothed-train-loss = 0.018770 | smoothed-test-loss = 0.017996 | rel_loss_delta = -0.005011092176600914 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 231/300 [11:46<03:25,  2.97s/it]2026-10-14 00:54:36.183 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 232, smoothed-train-loss = 0.018922 | smoothed-test-loss = 0.017615 | rel_loss_delta = 0.00809662685756936 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 232/300 [11:49<03:21,  2.96s/it]2026-10-14 00:54:39.229 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 233, smoothed-train-loss = 0.018763 | smoothed-test-loss = 0.018378 | rel_loss_delta = -0.008379214384337436 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 233/300 [11:52<03:19,  2.99s/it]2026-10-14 00:54:42.125 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 234, smoothed-train-loss = 0.018789 | smoothed-test-loss = 0.018147 | rel_loss_delta = 0.0013837231022302684 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 234/300 [11:55<03:15,  2.96s/it]2026-10-14 00:54:45.474 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 235, smoothed-train-loss = 0.018931 | smoothed-test-loss = 0.018497 | rel_loss_delta = 0.007565433330193772 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 235/300 [11:59<03:19,  3.08s/it]2026-10-14 00:54:48.544 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 236, smoothed-train-loss = 0.018859 | smoothed-test-loss = 0.019023 | rel_loss_delta = -0.0038409329355561547 | loss_stagnation_count = 0

Epoch:  79%|███████▊  | 236/300 [12:02<03:16,  3.07s/it]2026-10-14 00:54:51.577 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 237, smoothed-train-loss = 0.018893 | smoothed-test-loss = 0.018051 | rel_loss_delta = 0.0018279462878511637 | loss_stagnation_count = 0

Epoch:  79%|███████▉  | 237/300 [12:05<03:12,  3.06s/it]2026-10-14 00:54:54.460 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 238, smoothed-train-loss = 0.018833 | smoothed-test-loss = 0.018454 | rel_loss_delta = -0.003187291925502622 | loss_stagnation_count = 0

Epoch:  79%|███████▉  | 238/300 [12:08<03:06,  3.01s/it]2026-10-14 00:54:57.474 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 239, smoothed-train-loss = 0.018724 | smoothed-test-loss = 0.018571 | rel_loss_delta = -0.005780292149527815 | loss_stagnation_count = 0

Epoch:  80%|███████▉  | 239/300 [12:11<03:03,  3.01s/it]2026-10-14 00:55:00.544 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 240, smoothed-train-loss = 0.018888 | smoothed-test-loss = 0.017894 | rel_loss_delta = 0.008774523918180531 | loss_stagnation_count = 0

Epoch:  80%|████████  | 240/300 [12:14<03:01,  3.03s/it]2026-10-14 00:55:03.882 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 241, smoothed-train-loss = 0.018769 | smoothed-test-loss = 0.017831 | rel_loss_delta = -0.006303503196215625 | loss_stagnation_count = 0

Epoch:  80%|████████  | 241/300 [12:17<03:04,  3.12s/it]2026-10-14 00:55:06.838 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 242, smoothed-train-loss = 0.018915 | smoothed-test-loss = 0.020347 | rel_loss_delta = 0.0077378393611850695 | loss_stagnation_count = 0

Epoch:  81%|████████  | 242/300 [12:20<02:58,  3.07s/it]2026-10-14 00:55:09.823 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 243, smoothed-train-loss = 0.018747 | smoothed-test-loss = 0.019032 | rel_loss_delta = -0.008876547136387495 | loss_stagnation_count = 0

Epoch:  81%|████████  | 243/300 [12:23<02:53,  3.05s/it]2026-10-14 00:55:12.949 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 244, smoothed-train-loss = 0.018752 | smoothed-test-loss = 0.018052 | rel_loss_delta = 0.0002832187664527211 | loss_stagnation_count = 0

Epoch:  81%|████████▏ | 244/300 [12:26<02:51,  3.07s/it]2026-10-14 00:55:15.992 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 245, smoothed-train-loss = 0.018737 | smoothed-test-loss = 0.018226 | rel_loss_delta = -0.0007959407198322363 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 245/300 [12:29<02:48,  3.06s/it]2026-10-14 00:55:19.138 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 246, smoothed-train-loss = 0.018827 | smoothed-test-loss = 0.018148 | rel_loss_delta = 0.004791135309729317 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 246/300 [12:32<02:46,  3.09s/it]2026-10-14 00:55:22.372 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 247, smoothed-train-loss = 0.018919 | smoothed-test-loss = 0.017606 | rel_loss_delta = 0.004892543606379867 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 247/300 [12:35<02:45,  3.13s/it]2026-10-14 00:55:25.397 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 248, smoothed-train-loss = 0.018818 | smoothed-test-loss = 0.017970 | rel_loss_delta = -0.005315520235177211 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 248/300 [12:38<02:41,  3.10s/it]2026-10-14 00:55:28.463 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 249, smoothed-train-loss = 0.018710 | smoothed-test-loss = 0.017868 | rel_loss_delta = -0.005749728690176506 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 249/300 [12:42<02:37,  3.09s/it]2026-10-14 00:55:32.021 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 250, smoothed-train-loss = 0.018815 | smoothed-test-loss = 0.018499 | rel_loss_delta = 0.0055836253203508015 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 250/300 [12:45<02:41,  3.23s/it]2026-10-14 00:55:35.154 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 251, smoothed-train-loss = 0.018929 | smoothed-test-loss = 0.018069 | rel_loss_delta = 0.006087424928068704 | loss_stagnation_count = 0

Epoch:  84%|████████▎ | 251/300 [12:48<02:36,  3.20s/it]2026-10-14 00:55:38.406 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 252, smoothed-train-loss = 0.018690 | smoothed-test-loss = 0.018271 | rel_loss_delta = -0.012653904548597583 | loss_stagnation_count = 0

Epoch:  84%|████████▍ | 252/300 [12:51<02:34,  3.22s/it]2026-10-14 00:55:41.564 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 253, smoothed-train-loss = 0.018977 | smoothed-test-loss = 0.017589 | rel_loss_delta = 0.01539858715255849 | loss_stagnation_count = 0

Epoch:  84%|████████▍ | 253/300 [12:55<02:30,  3.20s/it]2026-10-14 00:55:44.398 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 254, smoothed-train-loss = 0.018629 | smoothed-test-loss = 0.018134 | rel_loss_delta = -0.01835836482969022 | loss_stagnation_count = 0

Epoch:  85%|████████▍ | 254/300 [12:57<02:22,  3.09s/it]2026-10-14 00:55:47.361 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 255, smoothed-train-loss = 0.018797 | smoothed-test-loss = 0.019345 | rel_loss_delta = 0.00900478722464419 | loss_stagnation_count = 0

Epoch:  85%|████████▌ | 255/300 [13:00<02:17,  3.05s/it]2026-10-14 00:55:50.441 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 256, smoothed-train-loss = 0.018897 | smoothed-test-loss = 0.018071 | rel_loss_delta = 0.005356239934700727 | loss_stagnation_count = 0

Epoch:  85%|████████▌ | 256/300 [13:04<02:14,  3.06s/it]2026-10-14 00:55:53.500 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 257, smoothed-train-loss = 0.018810 | smoothed-test-loss = 0.017511 | rel_loss_delta = -0.004645182772489556 | loss_stagnation_count = 0

Epoch:  86%|████████▌ | 257/300 [13:07<02:11,  3.06s/it]2026-10-14 00:55:56.398 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 258, smoothed-train-loss = 0.018897 | smoothed-test-loss = 0.018350 | rel_loss_delta = 0.004616035125943352 | loss_stagnation_count = 0

Epoch:  86%|████████▌ | 258/300 [13:09<02:06,  3.01s/it]2026-10-14 00:55:59.306 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 259, smoothed-train-loss = 0.018791 | smoothed-test-loss = 0.019631 | rel_loss_delta = -0.005575142472374024 | loss_stagnation_count = 0

Epoch:  86%|████████▋ | 259/300 [13:12<02:02,  2.98s/it]2026-10-14 00:56:02.336 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 260, smoothed-train-loss = 0.018843 | smoothed-test-loss = 0.017978 | rel_loss_delta = 0.002755376217263298 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 260/300 [13:15<01:59,  3.00s/it]2026-10-14 00:56:05.322 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 261, smoothed-train-loss = 0.018793 | smoothed-test-loss = 0.018976 | rel_loss_delta = -0.002647560196794632 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 261/300 [13:18<01:56,  2.99s/it]2026-10-14 00:56:08.250 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 262, smoothed-train-loss = 0.018832 | smoothed-test-loss = 0.017991 | rel_loss_delta = 0.002069470933894483 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 262/300 [13:21<01:52,  2.97s/it]2026-10-14 00:56:11.355 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 263, smoothed-train-loss = 0.018763 | smoothed-test-loss = 0.018273 | rel_loss_delta = -0.0036869746714926284 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 263/300 [13:24<01:51,  3.01s/it]2026-10-14 00:56:14.378 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 264, smoothed-train-loss = 0.018810 | smoothed-test-loss = 0.018743 | rel_loss_delta = 0.00251017126046922 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 264/300 [13:27<01:48,  3.02s/it]2026-10-14 00:56:17.397 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 265, smoothed-train-loss = 0.018742 | smoothed-test-loss = 0.018090 | rel_loss_delta = -0.0035884554616437522 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 265/300 [13:30<01:45,  3.02s/it]2026-10-14 00:56:20.511 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 266, smoothed-train-loss = 0.018840 | smoothed-test-loss = 0.018927 | rel_loss_delta = 0.005239257235031903 | loss_stagnation_count = 0

Epoch:  89%|████████▊ | 266/300 [13:34<01:43,  3.05s/it]2026-10-14 00:56:23.898 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 267, smoothed-train-loss = 0.018781 | smoothed-test-loss = 0.017972 | rel_loss_delta = -0.0031556166222903555 | loss_stagnation_count = 0

Epoch:  89%|████████▉ | 267/300 [13:37<01:43,  3.15s/it]2026-10-14 00:56:27.275 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 268, smoothed-train-loss = 0.018809 | smoothed-test-loss = 0.017696 | rel_loss_delta = 0.0014849032661544636 | loss_stagnation_count = 0

Epoch:  89%|████████▉ | 268/300 [13:40<01:42,  3.22s/it]2026-10-14 00:56:30.435 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 269, smoothed-train-loss = 0.018694 | smoothed-test-loss = 0.018985 | rel_loss_delta = -0.006083119353446958 | loss_stagnation_count = 0

Epoch:  90%|████████▉ | 269/300 [13:44<01:39,  3.20s/it]2026-10-14 00:56:33.345 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 270, smoothed-train-loss = 0.018758 | smoothed-test-loss = 0.019237 | rel_loss_delta = 0.0034014525039776766 | loss_stagnation_count = 0

Epoch:  90%|█████████ | 270/300 [13:46<01:33,  3.11s/it]2026-10-14 00:56:36.480 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 271, smoothed-train-loss = 0.018741 | smoothed-test-loss = 0.017762 | rel_loss_delta = -0.0009119763214513313 | loss_stagnation_count = 0

Epoch:  90%|█████████ | 271/300 [13:50<01:30,  3.12s/it]2026-10-14 00:56:39.469 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 272, smoothed-train-loss = 0.018842 | smoothed-test-loss = 0.017693 | rel_loss_delta = 0.005386323932945463 | loss_stagnation_count = 0

Epoch:  91%|█████████ | 272/300 [13:53<01:26,  3.08s/it]2026-10-14 00:56:42.431 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 273, smoothed-train-loss = 0.018803 | smoothed-test-loss = 0.018381 | rel_loss_delta = -0.0020590049109586823 | loss_stagnation_count = 0

Epoch:  91%|█████████ | 273/300 [13:56<01:22,  3.04s/it]2026-10-14 00:56:45.476 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 274, smoothed-train-loss = 0.018777 | smoothed-test-loss = 0.019082 | rel_loss_delta = -0.001361883148556274 | loss_stagnation_count = 0

Epoch:  91%|█████████▏| 274/300 [13:59<01:19,  3.04s/it]2026-10-14 00:56:48.491 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 275, smoothed-train-loss = 0.018833 | smoothed-test-loss = 0.020208 | rel_loss_delta = 0.002986024096484418 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 275/300 [14:02<01:15,  3.04s/it]2026-10-14 00:56:51.444 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 276, smoothed-train-loss = 0.018688 | smoothed-test-loss = 0.018244 | rel_loss_delta = -0.007711248183104713 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 276/300 [14:05<01:12,  3.01s/it]2026-10-14 00:56:54.549 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 277, smoothed-train-loss = 0.018722 | smoothed-test-loss = 0.018122 | rel_loss_delta = 0.0018010894947789804 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 277/300 [14:08<01:09,  3.04s/it]2026-10-14 00:56:57.451 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 278, smoothed-train-loss = 0.018851 | smoothed-test-loss = 0.018189 | rel_loss_delta = 0.006871526270436207 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 278/300 [14:11<01:05,  3.00s/it]2026-10-14 00:57:00.475 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 279, smoothed-train-loss = 0.018767 | smoothed-test-loss = 0.017965 | rel_loss_delta = -0.004437281716679283 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 279/300 [14:14<01:03,  3.01s/it]2026-10-14 00:57:03.479 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 280, smoothed-train-loss = 0.018866 | smoothed-test-loss = 0.018164 | rel_loss_delta = 0.0053001600159375 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 280/300 [14:17<01:00,  3.01s/it]2026-10-14 00:57:06.610 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 281, smoothed-train-loss = 0.018724 | smoothed-test-loss = 0.019833 | rel_loss_delta = -0.007520147022365901 | loss_stagnation_count = 0

Epoch:  94%|█████████▎| 281/300 [14:20<00:57,  3.04s/it]2026-10-14 00:57:09.496 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 282, smoothed-train-loss = 0.018787 | smoothed-test-loss = 0.018239 | rel_loss_delta = 0.0033543868727694526 | loss_stagnation_count = 0

Epoch:  94%|█████████▍| 282/300 [14:23<00:53,  3.00s/it]2026-10-14 00:57:12.483 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 283, smoothed-train-loss = 0.018735 | smoothed-test-loss = 0.019102 | rel_loss_delta = -0.002773453659406619 | loss_stagnation_count = 0

Epoch:  94%|█████████▍| 283/300 [14:26<00:50,  2.99s/it]2026-10-14 00:57:15.453 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 284, smoothed-train-loss = 0.018682 | smoothed-test-loss = 0.019113 | rel_loss_delta = -0.002838106932244064 | loss_stagnation_count = 0

Epoch:  95%|█████████▍| 284/300 [14:29<00:47,  2.99s/it]2026-10-14 00:57:18.499 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 285, smoothed-train-loss = 0.018884 | smoothed-test-loss = 0.018147 | rel_loss_delta = 0.010804564391763593 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 285/300 [14:32<00:45,  3.00s/it]2026-10-14 00:57:21.383 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 286, smoothed-train-loss = 0.018792 | smoothed-test-loss = 0.017801 | rel_loss_delta = -0.0048595626532420685 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 286/300 [14:34<00:41,  2.97s/it]2026-10-14 00:57:24.460 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 287, smoothed-train-loss = 0.018808 | smoothed-test-loss = 0.017738 | rel_loss_delta = 0.0008670413835527919 | loss_stagnation_count = 0

Epoch:  96%|█████████▌| 287/300 [14:38<00:39,  3.00s/it]2026-10-14 00:57:27.481 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 288, smoothed-train-loss = 0.018715 | smoothed-test-loss = 0.018100 | rel_loss_delta = -0.004966657895973947 | loss_stagnation_count = 0

Epoch:  96%|█████████▌| 288/300 [14:41<00:36,  3.01s/it]2026-10-14 00:57:30.576 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 289, smoothed-train-loss = 0.018723 | smoothed-test-loss = 0.017947 | rel_loss_delta = 0.00044552308790279726 | loss_stagnation_count = 0

Epoch:  96%|█████████▋| 289/300 [14:44<00:33,  3.03s/it]2026-10-14 00:57:33.486 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 290, smoothed-train-loss = 0.018822 | smoothed-test-loss = 0.017840 | rel_loss_delta = 0.00524879214463413 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 290/300 [14:47<00:29,  3.00s/it]2026-10-14 00:57:36.540 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 291, smoothed-train-loss = 0.018811 | smoothed-test-loss = 0.017712 | rel_loss_delta = -0.0005790827942275857 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 291/300 [14:50<00:27,  3.01s/it]2026-10-14 00:57:39.469 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 292, smoothed-train-loss = 0.018635 | smoothed-test-loss = 0.018733 | rel_loss_delta = -0.009331038546195337 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 292/300 [14:53<00:23,  2.99s/it]2026-10-14 00:57:42.426 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 293, smoothed-train-loss = 0.018694 | smoothed-test-loss = 0.017621 | rel_loss_delta = 0.003166645129463351 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 293/300 [14:56<00:20,  2.98s/it]2026-10-14 00:57:45.345 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 294, smoothed-train-loss = 0.018780 | smoothed-test-loss = 0.018038 | rel_loss_delta = 0.004604669072994726 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 294/300 [14:58<00:17,  2.96s/it]2026-10-14 00:57:48.311 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 295, smoothed-train-loss = 0.018721 | smoothed-test-loss = 0.017956 | rel_loss_delta = -0.0031435296127791325 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 295/300 [15:01<00:14,  2.96s/it]2026-10-14 00:57:51.338 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 296, smoothed-train-loss = 0.018648 | smoothed-test-loss = 0.017989 | rel_loss_delta = -0.00391713391270584 | loss_stagnation_count = 0

Epoch:  99%|█████████▊| 296/300 [15:04<00:11,  2.98s/it]2026-10-14 00:57:54.255 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 297, smoothed-train-loss = 0.018678 | smoothed-test-loss = 0.018377 | rel_loss_delta = 0.001602685295828527 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 297/300 [15:07<00:08,  2.96s/it]2026-10-14 00:57:57.187 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 298, smoothed-train-loss = 0.018742 | smoothed-test-loss = 0.019498 | rel_loss_delta = 0.003433955426246693 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 298/300 [15:10<00:05,  2.95s/it]2026-10-14 00:58:00.133 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 299, smoothed-train-loss = 0.018823 | smoothed-test-loss = 0.019040 | rel_loss_delta = 0.004302898399649817 | loss_stagnation_count = 0

Epoch: 100%|█████████▉| 299/300 [15:13<00:02,  2.95s/it]2026-10-14 00:58:03.109 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 300, smoothed-train-loss = 0.018647 | smoothed-test-loss = 0.017934 | rel_loss_delta = -0.009318794118722811 | loss_stagnation_count = 0

Epoch: 100%|██████████| 300/300 [15:16<00:00,  3.06s/it]
2026-10-14 00:58:03.114 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:0.017810601741075516
2026-10-14 00:58:03.115 | INFO     | __main__:<module>:282 - Given num_reflow_iterations = 1 and current reflow_iteration = 1, no further reflows are needed, exiting the reflow loop
reflow iterations:  50%|█████     | 1/2 [31:05<31:05, 1865.01s/it]
2026-10-14 00:58:03.120 | INFO     | __main__:<module>:340 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_4_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-14T00:26:57.171654.pkl
2026-10-14 00:58:03.120 | INFO     | __main__:<module>:341 - vt training process finished successfully.

Process finished with exit code 
```
# Inference

## (A) 1-RecFlow (0-Reflow)

### (i) euler
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
  Eval Results         : {0.0027896204963326456: {'real_generated': {'mean': 0.24769385159015656, 'stddev': 0.20144550536741368}, 'real_real': {'mean': 0.37529077231884, 'stddev': 0.1565505794271428}, 'generated_generated': {'mean': 0.34362403154373167, 'stddev': 0.14960283973794358}}, 0.0069740512408316135: {'real_generated': {'mean': 0.22509007006883622, 'stddev': 0.12226900704206085}, 'real_real': {'mean': 0.0880642767995596, 'stddev': 0.04301452855798539}, 'generated_generated': {'mean': 0.22344349920749665, 'stddev': 0.15510546437394118}}, 0.01: {'real_generated': {'mean': 0.3025777816772461, 'stddev': 0.22254257786974102}, 'real_real': {'mean': 0.22101373374462127, 'stddev': 0.14723877010191072}, 'generated_generated': {'mean': 0.15360376238822937, 'stddev': 0.10586595704385159}}, 0.013948102481663227: {'real_generated': {'mean': 0.28432684391736984, 'stddev': 0.1415634314573419}, 'real_real': {'mean': 0.11321609541773796, 'stddev': 0.05204483390610657}, 'generated_generated': {'mean': 0.18329050540924072, 'stddev': 0.0946885816567789}}, 0.02: {'real_generated': {'mean': 0.19063099138438702, 'stddev': 0.23292413025715858}, 'real_real': {'mean': 0.20394887775182724, 'stddev': 0.18940954070999308}, 'generated_generated': {'mean': 0.17182657569646836, 'stddev': 0.17124943099303178}}, 0.027896204963326454: {'real_generated': {'mean': 0.3477310657501221, 'stddev': 0.14345208267386397}, 'real_real': {'mean': 0.32092337012290956, 'stddev': 0.20727079697533393}, 'generated_generated': {'mean': 0.09964880086481571, 'stddev': 0.11545340157352948}}, 0.05: {'real_generated': {'mean': 0.28092335164546967, 'stddev': 0.24728789971411871}, 'real_real': {'mean': 0.40856285095214845, 'stddev': 0.2386643064402809}, 'generated_generated': {'mean': 0.3273811757564545, 'stddev': 0.09090176944583114}}, 0.05579240992665291: {'real_generated': {'mean': 0.25336326062679293, 'stddev': 0.1766136143176679}, 'real_real': {'mean': 0.1475096419453621, 'stddev': 0.1601464622702066}, 'generated_generated': {'mean': 0.18417986780405043, 'stddev': 0.11740585052518439}}, 0.1: {'real_generated': {'mean': 0.3210266351699829, 'stddev': 0.19480724680709258}, 'real_real': {'mean': 0.28835850656032563, 'stddev': 0.16537213283982577}, 'generated_generated': {'mean': 0.08573752157390117, 'stddev': 0.07713877951503026}}, 0.2: {'real_generated': {'mean': 0.3672471195459366, 'stddev': 0.24825700476448803}, 'real_real': {'mean': 0.18512920290231705, 'stddev': 0.09760537712645806}, 'generated_generated': {'mean': 0.2821045607328415, 'stddev': 0.1745596855249914}}, 0.5: {'real_generated': {'mean': 0.20916447937488555, 'stddev': 0.11176985022046744}, 'real_real': {'mean': 0.21669614166021348, 'stddev': 0.13592720934492888}, 'generated_generated': {'mean': 0.14014641158282756, 'stddev': 0.22608671612574988}}, 1.0: {'real_generated': {'mean': 0.30830729603767393, 'stddev': 0.08283535789226679}, 'real_real': {'mean': 0.27751975432038306, 'stddev': 0.18784069736921274}, 'generated_generated': {'mean': 0.2886589985340834, 'stddev': 0.18065725524470788}}, 2.0: {'real_generated': {'mean': 0.1893328994512558, 'stddev': 0.1519928245622434}, 'real_real': {'mean': 0.07429639846086503, 'stddev': 0.04944096950445489}, 'generated_generated': {'mean': 0.1292909488081932, 'stddev': 0.06738911923225954}}, 5.0: {'real_generated': {'mean': 0.04256027527153492, 'stddev': 0.01727426629508039}, 'real_real': {'mean': 0.087179109826684, 'stddev': 0.08637635566345361}, 'generated_generated': {'mean': 0.036655713245272635, 'stddev': 0.04309893548180586}}, 10.0: {'real_generated': {'mean': 0.012473854422569274, 'stddev': 0.007345669422563532}, 'real_real': {'mean': 0.011895246151834727, 'stddev': 0.010408107455431876}, 'generated_generated': {'mean': 0.0028015172574669123, 'stddev': 0.0012386554002845156}}}
==================================================
```
### (ii) rk4
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : rk4
  VT Trajectory Model  : global
  Num Reflows          : 0
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 21.36s
  Evaluations (t)      : 1000
  Eval Results         : {0.0027896204963326456: {'real_generated': {'mean': 0.4253720760345459, 'stddev': 0.2760469476498024}, 'real_real': {'mean': 0.37529077231884, 'stddev': 0.1565505794271428}, 'generated_generated': {'mean': 0.3543209180235863, 'stddev': 0.33684455672636365}}, 0.0069740512408316135: {'real_generated': {'mean': 0.3046700656414032, 'stddev': 0.1050164086748954}, 'real_real': {'mean': 0.0880642767995596, 'stddev': 0.04301452855798539}, 'generated_generated': {'mean': 0.24616990610957146, 'stddev': 0.18728606595981498}}, 0.01: {'real_generated': {'mean': 0.3435270994901657, 'stddev': 0.3580199444867617}, 'real_real': {'mean': 0.22101373374462127, 'stddev': 0.14723877010191072}, 'generated_generated': {'mean': 0.25345969423651693, 'stddev': 0.19438822188049665}}, 0.013948102481663227: {'real_generated': {'mean': 0.17815234437584876, 'stddev': 0.14841281487993463}, 'real_real': {'mean': 0.11321609541773796, 'stddev': 0.05204483390610657}, 'generated_generated': {'mean': 0.21705230623483657, 'stddev': 0.22020946649289674}}, 0.02: {'real_generated': {'mean': 0.18600718826055526, 'stddev': 0.11014967388673011}, 'real_real': {'mean': 0.20394887775182724, 'stddev': 0.18940954070999308}, 'generated_generated': {'mean': 0.0998033419251442, 'stddev': 0.04560659769003486}}, 0.027896204963326454: {'real_generated': {'mean': 0.4268298625946045, 'stddev': 0.1870124615524691}, 'real_real': {'mean': 0.32092337012290956, 'stddev': 0.20727079697533393}, 'generated_generated': {'mean': 0.11998284310102462, 'stddev': 0.09659779630455158}}, 0.05: {'real_generated': {'mean': 0.21323385313153267, 'stddev': 0.12511215238118242}, 'real_real': {'mean': 0.40856285095214845, 'stddev': 0.2386643064402809}, 'generated_generated': {'mean': 0.22243601083755493, 'stddev': 0.06560508252348696}}, 0.05579240992665291: {'real_generated': {'mean': 0.201075379550457, 'stddev': 0.13356743643581756}, 'real_real': {'mean': 0.1475096419453621, 'stddev': 0.1601464622702066}, 'generated_generated': {'mean': 0.15561259649693965, 'stddev': 0.16397121965055259}}, 0.1: {'real_generated': {'mean': 0.29793293625116346, 'stddev': 0.23931637547517767}, 'real_real': {'mean': 0.28835850656032563, 'stddev': 0.16537213283982577}, 'generated_generated': {'mean': 0.20566374361515044, 'stddev': 0.10001628531064173}}, 0.2: {'real_generated': {'mean': 0.26149348318576815, 'stddev': 0.08053365666179518}, 'real_real': {'mean': 0.18512920290231705, 'stddev': 0.09760537712645806}, 'generated_generated': {'mean': 0.20755565464496611, 'stddev': 0.12773175226988892}}, 0.5: {'real_generated': {'mean': 0.11332650482654572, 'stddev': 0.09231415778960057}, 'real_real': {'mean': 0.21669614166021348, 'stddev': 0.13592720934492888}, 'generated_generated': {'mean': 0.17024298757314682, 'stddev': 0.14546910544420313}}, 1.0: {'real_generated': {'mean': 0.19513882249593734, 'stddev': 0.1263472807373275}, 'real_real': {'mean': 0.27751975432038306, 'stddev': 0.18784069736921274}, 'generated_generated': {'mean': 0.23678955882787706, 'stddev': 0.14849395176242852}}, 2.0: {'real_generated': {'mean': 0.1927891418337822, 'stddev': 0.12053547473088855}, 'real_real': {'mean': 0.07429639846086503, 'stddev': 0.04944096950445489}, 'generated_generated': {'mean': 0.07624651193618774, 'stddev': 0.04175517586678}}, 5.0: {'real_generated': {'mean': 0.05997608006000519, 'stddev': 0.03104780800896224}, 'real_real': {'mean': 0.087179109826684, 'stddev': 0.08637635566345361}, 'generated_generated': {'mean': 0.10123682916164398, 'stddev': 0.04026111153979153}}, 10.0: {'real_generated': {'mean': 0.019344598054885864, 'stddev': 0.010205693197831128}, 'real_real': {'mean': 0.011895246151834727, 'stddev': 0.010408107455431876}, 'generated_generated': {'mean': 0.011335631972178817, 'stddev': 0.01695946045526527}}}
==================================================
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_4_2026-10-15T13:21:25.618263_trajectory_model_global_ode_solver_method_rk4.png
```
### (iii) dopri8
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : dopri8
  VT Trajectory Model  : global
  Num Reflows          : 0
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 62.35s
  Evaluations (t)      : 1000
  Eval Results         : {0.0027896204963326456: {'real_generated': {'mean': 0.25066138803958893, 'stddev': 0.12821408751036537}, 'real_real': {'mean': 0.37529077231884, 'stddev': 0.1565505794271428}, 'generated_generated': {'mean': 0.23222781270742415, 'stddev': 0.1910469314098145}}, 0.0069740512408316135: {'real_generated': {'mean': 0.3154006749391556, 'stddev': 0.16201451881155932}, 'real_real': {'mean': 0.0880642767995596, 'stddev': 0.04301452855798539}, 'generated_generated': {'mean': 0.08660072404891253, 'stddev': 0.10138245615645557}}, 0.01: {'real_generated': {'mean': 0.34411257803440093, 'stddev': 0.2079238727482096}, 'real_real': {'mean': 0.22101373374462127, 'stddev': 0.14723877010191072}, 'generated_generated': {'mean': 0.21273805052042008, 'stddev': 0.16735567791878822}}, 0.013948102481663227: {'real_generated': {'mean': 0.2585406184196472, 'stddev': 0.2770953710284411}, 'real_real': {'mean': 0.11321609541773796, 'stddev': 0.05204483390610657}, 'generated_generated': {'mean': 0.1890079692006111, 'stddev': 0.20586880744827765}}, 0.02: {'real_generated': {'mean': 0.26335152983665466, 'stddev': 0.16194491946986192}, 'real_real': {'mean': 0.20394887775182724, 'stddev': 0.18940954070999308}, 'generated_generated': {'mean': 0.1800018411129713, 'stddev': 0.10918787047815644}}, 0.027896204963326454: {'real_generated': {'mean': 0.15969469770789146, 'stddev': 0.06847863537205401}, 'real_real': {'mean': 0.32092337012290956, 'stddev': 0.20727079697533393}, 'generated_generated': {'mean': 0.25587166398763656, 'stddev': 0.16546590283314916}}, 0.05: {'real_generated': {'mean': 0.424286824464798, 'stddev': 0.2553737891072619}, 'real_real': {'mean': 0.40856285095214845, 'stddev': 0.2386643064402809}, 'generated_generated': {'mean': 0.2953356236219406, 'stddev': 0.15757461467415515}}, 0.05579240992665291: {'real_generated': {'mean': 0.3497161567211151, 'stddev': 0.4137026471043318}, 'real_real': {'mean': 0.1475096419453621, 'stddev': 0.1601464622702066}, 'generated_generated': {'mean': 0.3183047153055668, 'stddev': 0.2240537483273517}}, 0.1: {'real_generated': {'mean': 0.3366167962551117, 'stddev': 0.16277905028095055}, 'real_real': {'mean': 0.28835850656032563, 'stddev': 0.16537213283982577}, 'generated_generated': {'mean': 0.1811687245965004, 'stddev': 0.148139362450737}}, 0.2: {'real_generated': {'mean': 0.20478385984897612, 'stddev': 0.09385493969109851}, 'real_real': {'mean': 0.18512920290231705, 'stddev': 0.09760537712645806}, 'generated_generated': {'mean': 0.1772482253611088, 'stddev': 0.14176563591114605}}, 0.5: {'real_generated': {'mean': 0.2388467624783516, 'stddev': 0.09824748493339736}, 'real_real': {'mean': 0.21669614166021348, 'stddev': 0.13592720934492888}, 'generated_generated': {'mean': 0.15850446820259095, 'stddev': 0.07535211209953599}}, 1.0: {'real_generated': {'mean': 0.2047012209892273, 'stddev': 0.11575025839901322}, 'real_real': {'mean': 0.27751975432038306, 'stddev': 0.18784069736921274}, 'generated_generated': {'mean': 0.24648718796670438, 'stddev': 0.2494008971947051}}, 2.0: {'real_generated': {'mean': 0.1348670482635498, 'stddev': 0.07025761229964829}, 'real_real': {'mean': 0.07429639846086503, 'stddev': 0.04944096950445489}, 'generated_generated': {'mean': 0.26578415036201475, 'stddev': 0.14422884781462916}}, 5.0: {'real_generated': {'mean': 0.03234882671386004, 'stddev': 0.017461810328724146}, 'real_real': {'mean': 0.087179109826684, 'stddev': 0.08637635566345361}, 'generated_generated': {'mean': 0.04724878296256065, 'stddev': 0.027761270704558225}}, 10.0: {'real_generated': {'mean': 0.0248381819576025, 'stddev': 0.011631919610289008}, 'real_real': {'mean': 0.011895246151834727, 'stddev': 0.010408107455431876}, 'generated_generated': {'mean': 0.023069003690034152, 'stddev': 0.029416912724489746}}}
==================================================
/home/mbaddar/Documents/mbaddar/ttflow/ttflow/viz/viz.py:227: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
2026-10-15 14:11:42.518 | INFO     | __main__:<module>:338 - Inference and evaluation process finished successfully!
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_4_2026-10-15T14:03:56.938688_trajectory_model_global_ode_solver_method_dopri8.png
```

## (B) 2-RecFlow (1-Reflow)
### - `odesolve=euler`

```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : euler
  VT Trajectory Model  : global
  Num Reflows          : 1
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 5.55s
  Evaluations (t)      : 1000
  Eval Results         : {0.0027896204963326456: {'real_generated': {'mean': 1.4463842391967774, 'stddev': 0.525917384381303}, 'real_real': {'mean': 0.37529077231884, 'stddev': 0.1565505794271428}, 'generated_generated': {'mean': 0.18950481563806534, 'stddev': 0.10668610414420028}}, 0.0069740512408316135: {'real_generated': {'mean': 1.3155102252960205, 'stddev': 0.7854231315442262}, 'real_real': {'mean': 0.0880642767995596, 'stddev': 0.04301452855798539}, 'generated_generated': {'mean': 0.24513495415449144, 'stddev': 0.2568158386074465}}, 0.01: {'real_generated': {'mean': 1.4155438423156739, 'stddev': 0.274282162159392}, 'real_real': {'mean': 0.22101373374462127, 'stddev': 0.14723877010191072}, 'generated_generated': {'mean': 0.1707210263237357, 'stddev': 0.14681047311862724}}, 0.013948102481663227: {'real_generated': {'mean': 1.042065155506134, 'stddev': 0.5813507012339543}, 'real_real': {'mean': 0.11321609541773796, 'stddev': 0.05204483390610657}, 'generated_generated': {'mean': 0.25033764690160754, 'stddev': 0.13418588554594713}}, 0.02: {'real_generated': {'mean': 1.1208451747894288, 'stddev': 0.43164501850574144}, 'real_real': {'mean': 0.20394887775182724, 'stddev': 0.18940954070999308}, 'generated_generated': {'mean': 0.21251765340566636, 'stddev': 0.18537428161599845}}, 0.027896204963326454: {'real_generated': {'mean': 0.9959906965494156, 'stddev': 0.910005767898864}, 'real_real': {'mean': 0.32092337012290956, 'stddev': 0.20727079697533393}, 'generated_generated': {'mean': 0.2361626371741295, 'stddev': 0.13746689442615312}}, 0.05: {'real_generated': {'mean': 1.5133301258087157, 'stddev': 0.3923259703038088}, 'real_real': {'mean': 0.40856285095214845, 'stddev': 0.2386643064402809}, 'generated_generated': {'mean': 0.24117148984223605, 'stddev': 0.2649796972529103}}, 0.05579240992665291: {'real_generated': {'mean': 1.1923799514770508, 'stddev': 0.1168056787708431}, 'real_real': {'mean': 0.1475096419453621, 'stddev': 0.1601464622702066}, 'generated_generated': {'mean': 0.15611255019903184, 'stddev': 0.07270414512683707}}, 0.1: {'real_generated': {'mean': 1.2594483852386475, 'stddev': 0.5195853830933398}, 'real_real': {'mean': 0.28835850656032563, 'stddev': 0.16537213283982577}, 'generated_generated': {'mean': 0.30750015676021575, 'stddev': 0.13056806183710398}}, 0.2: {'real_generated': {'mean': 0.9713327884674072, 'stddev': 0.5843554717136391}, 'real_real': {'mean': 0.18512920290231705, 'stddev': 0.09760537712645806}, 'generated_generated': {'mean': 0.2465551346540451, 'stddev': 0.1902576968393311}}, 0.5: {'real_generated': {'mean': 1.0802013397216796, 'stddev': 0.12770315719912853}, 'real_real': {'mean': 0.21669614166021348, 'stddev': 0.13592720934492888}, 'generated_generated': {'mean': 0.214663908444345, 'stddev': 0.2732078259382706}}, 1.0: {'real_generated': {'mean': 1.2772693514823914, 'stddev': 0.3908604450530645}, 'real_real': {'mean': 0.27751975432038306, 'stddev': 0.18784069736921274}, 'generated_generated': {'mean': 0.24091835767030717, 'stddev': 0.21568425065981908}}, 2.0: {'real_generated': {'mean': 1.0733427047729491, 'stddev': 0.4482921121031875}, 'real_real': {'mean': 0.07429639846086503, 'stddev': 0.04944096950445489}, 'generated_generated': {'mean': 0.22550526186823844, 'stddev': 0.12340873284461923}}, 5.0: {'real_generated': {'mean': 0.3711400806903839, 'stddev': 0.09480804201333955}, 'real_real': {'mean': 0.087179109826684, 'stddev': 0.08637635566345361}, 'generated_generated': {'mean': 0.08546705953776837, 'stddev': 0.05534992783676292}}, 10.0: {'real_generated': {'mean': 0.07322697043418884, 'stddev': 0.04039265224463941}, 'real_real': {'mean': 0.011895246151834727, 'stddev': 0.010408107455431876}, 'generated_generated': {'mean': 0.03018131535500288, 'stddev': 0.02172561620266237}}}
==================================================
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_4_2026-10-15T14:21:56.443314_trajectory_model_global_ode_solver_method_euler.png
```
### `odesolver=rk4`
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : rk4
  VT Trajectory Model  : global
  Num Reflows          : 1
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 21.15s
  Evaluations (t)      : 1000
  Eval Results         : {0.0027896204963326456: {'real_generated': {'mean': 1.7048184871673584, 'stddev': 0.49786114370720214}, 'real_real': {'mean': 0.37529077231884, 'stddev': 0.1565505794271428}, 'generated_generated': {'mean': 0.1487441688776016, 'stddev': 0.06788309662582198}}, 0.0069740512408316135: {'real_generated': {'mean': 2.0679450035095215, 'stddev': 0.7398366003052999}, 'real_real': {'mean': 0.0880642767995596, 'stddev': 0.04301452855798539}, 'generated_generated': {'mean': 0.42705883681774137, 'stddev': 0.3323353503880368}}, 0.01: {'real_generated': {'mean': 1.1714638113975524, 'stddev': 0.616614933904458}, 'real_real': {'mean': 0.22101373374462127, 'stddev': 0.14723877010191072}, 'generated_generated': {'mean': 0.40831343829631805, 'stddev': 0.3005491499481929}}, 0.013948102481663227: {'real_generated': {'mean': 1.313367986679077, 'stddev': 0.6035818114247408}, 'real_real': {'mean': 0.11321609541773796, 'stddev': 0.05204483390610657}, 'generated_generated': {'mean': 0.24713580906391144, 'stddev': 0.15338408289593863}}, 0.02: {'real_generated': {'mean': 1.4532615184783935, 'stddev': 0.5008484741501297}, 'real_real': {'mean': 0.20394887775182724, 'stddev': 0.18940954070999308}, 'generated_generated': {'mean': 0.18771330192685126, 'stddev': 0.1685597578550481}}, 0.027896204963326454: {'real_generated': {'mean': 1.6063739776611328, 'stddev': 0.7569522926528264}, 'real_real': {'mean': 0.32092337012290956, 'stddev': 0.20727079697533393}, 'generated_generated': {'mean': 0.0956646203994751, 'stddev': 0.05272721363323867}}, 0.05: {'real_generated': {'mean': 1.68206467628479, 'stddev': 0.7426553961235038}, 'real_real': {'mean': 0.40856285095214845, 'stddev': 0.2386643064402809}, 'generated_generated': {'mean': 0.1630458950996399, 'stddev': 0.19780821557084743}}, 0.05579240992665291: {'real_generated': {'mean': 1.8126560688018798, 'stddev': 0.2548673740826864}, 'real_real': {'mean': 0.1475096419453621, 'stddev': 0.1601464622702066}, 'generated_generated': {'mean': 0.11067120805382728, 'stddev': 0.061896045567326315}}, 0.1: {'real_generated': {'mean': 1.660011625289917, 'stddev': 0.6417730541822452}, 'real_real': {'mean': 0.28835850656032563, 'stddev': 0.16537213283982577}, 'generated_generated': {'mean': 0.16269410997629166, 'stddev': 0.15181174410247864}}, 0.2: {'real_generated': {'mean': 1.5778504848480224, 'stddev': 0.4217316800238657}, 'real_real': {'mean': 0.18512920290231705, 'stddev': 0.09760537712645806}, 'generated_generated': {'mean': 0.20696276128292085, 'stddev': 0.15280817001361363}}, 0.5: {'real_generated': {'mean': 1.688841199874878, 'stddev': 0.6285696686341101}, 'real_real': {'mean': 0.21669614166021348, 'stddev': 0.13592720934492888}, 'generated_generated': {'mean': 0.1496398478746414, 'stddev': 0.11057641855099526}}, 1.0: {'real_generated': {'mean': 1.5982754945755004, 'stddev': 0.3464141471658072}, 'real_real': {'mean': 0.27751975432038306, 'stddev': 0.18784069736921274}, 'generated_generated': {'mean': 0.14414641223847866, 'stddev': 0.11089103540677285}}, 2.0: {'real_generated': {'mean': 1.5993579626083374, 'stddev': 0.7355099820990112}, 'real_real': {'mean': 0.07429639846086503, 'stddev': 0.04944096950445489}, 'generated_generated': {'mean': 0.1753278762102127, 'stddev': 0.12697279926783336}}, 5.0: {'real_generated': {'mean': 0.5668506741523742, 'stddev': 0.22410194893587732}, 'real_real': {'mean': 0.087179109826684, 'stddev': 0.08637635566345361}, 'generated_generated': {'mean': 0.06440297290682792, 'stddev': 0.03465591083783514}}, 10.0: {'real_generated': {'mean': 0.1206980049610138, 'stddev': 0.0647279935149728}, 'real_real': {'mean': 0.011895246151834727, 'stddev': 0.010408107455431876}, 'generated_generated': {'mean': 0.006874358374625444, 'stddev': 0.0036253893229315844}}}
==================================================
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_4_timestamp_2026-10-15T14:52:36.583020_trajectory_model_global_ode_solver_method_rk4_num_reflows_1.png
```
### `odesolver=dopri8`
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : dopri8
  VT Trajectory Model  : global
  Num Reflows          : 1
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 29.69s
  Evaluations (t)      : 1000
  Eval Results         : {0.0027896204963326456: {'real_generated': {'mean': 1.4112922430038453, 'stddev': 0.6573139709168861}, 'real_real': {'mean': 0.37529077231884, 'stddev': 0.1565505794271428}, 'generated_generated': {'mean': 0.2338704437017441, 'stddev': 0.14203757744786383}}, 0.0069740512408316135: {'real_generated': {'mean': 1.2084116816520691, 'stddev': 0.6590581699845754}, 'real_real': {'mean': 0.0880642767995596, 'stddev': 0.04301452855798539}, 'generated_generated': {'mean': 0.2572970539331436, 'stddev': 0.23866024017103912}}, 0.01: {'real_generated': {'mean': 1.482796549797058, 'stddev': 0.4137033011622539}, 'real_real': {'mean': 0.22101373374462127, 'stddev': 0.14723877010191072}, 'generated_generated': {'mean': 0.13478426039218902, 'stddev': 0.07521366967772726}}, 0.013948102481663227: {'real_generated': {'mean': 0.972235095500946, 'stddev': 0.24335965131847495}, 'real_real': {'mean': 0.11321609541773796, 'stddev': 0.05204483390610657}, 'generated_generated': {'mean': 0.2336980313062668, 'stddev': 0.0844660626010945}}, 0.02: {'real_generated': {'mean': 1.422971820831299, 'stddev': 0.8387889759236572}, 'real_real': {'mean': 0.20394887775182724, 'stddev': 0.18940954070999308}, 'generated_generated': {'mean': 0.19423171430826186, 'stddev': 0.15215290255872002}}, 0.027896204963326454: {'real_generated': {'mean': 1.1338194906711578, 'stddev': 0.8258902293185029}, 'real_real': {'mean': 0.32092337012290956, 'stddev': 0.20727079697533393}, 'generated_generated': {'mean': 0.4282551884651184, 'stddev': 0.4142194946654176}}, 0.05: {'real_generated': {'mean': 1.3738637566566467, 'stddev': 0.6716519096284542}, 'real_real': {'mean': 0.40856285095214845, 'stddev': 0.2386643064402809}, 'generated_generated': {'mean': 0.23653304129838942, 'stddev': 0.23995770928392585}}, 0.05579240992665291: {'real_generated': {'mean': 1.3264006137847901, 'stddev': 0.46080894434981884}, 'real_real': {'mean': 0.1475096419453621, 'stddev': 0.1601464622702066}, 'generated_generated': {'mean': 0.19605351388454437, 'stddev': 0.07873442313537128}}, 0.1: {'real_generated': {'mean': 1.682159161567688, 'stddev': 0.30893105196911347}, 'real_real': {'mean': 0.28835850656032563, 'stddev': 0.16537213283982577}, 'generated_generated': {'mean': 0.3071569323539734, 'stddev': 0.09519428400402158}}, 0.2: {'real_generated': {'mean': 1.2118119180202485, 'stddev': 0.8803697825073346}, 'real_real': {'mean': 0.18512920290231705, 'stddev': 0.09760537712645806}, 'generated_generated': {'mean': 0.20761520117521287, 'stddev': 0.13441028957606518}}, 0.5: {'real_generated': {'mean': 1.700590229034424, 'stddev': 0.39987883743055375}, 'real_real': {'mean': 0.21669614166021348, 'stddev': 0.13592720934492888}, 'generated_generated': {'mean': 0.181376214325428, 'stddev': 0.12571801934625346}}, 1.0: {'real_generated': {'mean': 1.2653594970703126, 'stddev': 0.4138680502804642}, 'real_real': {'mean': 0.27751975432038306, 'stddev': 0.18784069736921274}, 'generated_generated': {'mean': 0.31335727125406265, 'stddev': 0.20226052316782703}}, 2.0: {'real_generated': {'mean': 1.1878146409988404, 'stddev': 0.2759783661550201}, 'real_real': {'mean': 0.07429639846086503, 'stddev': 0.04944096950445489}, 'generated_generated': {'mean': 0.1526479348540306, 'stddev': 0.1442633546543879}}, 5.0: {'real_generated': {'mean': 0.3422751724720001, 'stddev': 0.11409343899939701}, 'real_real': {'mean': 0.087179109826684, 'stddev': 0.08637635566345361}, 'generated_generated': {'mean': 0.034956113807857035, 'stddev': 0.025237377497768883}}, 10.0: {'real_generated': {'mean': 0.06419523358345032, 'stddev': 0.030608854055861754}, 'real_real': {'mean': 0.011895246151834727, 'stddev': 0.010408107455431876}, 'generated_generated': {'mean': 0.015423434041440486, 'stddev': 0.020384020825796368}}}
==================================================
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_4_timestamp_2026-10-15T15:09:33.123300_trajectory_model_global_ode_solver_method_dopri8_num_reflows_1.png
```