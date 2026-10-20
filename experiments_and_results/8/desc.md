
# Conclusion 
Uniform prior is bad visually 
# Desc 
* Using Uniform Prior - all prev experiments use Gaussian one
* args 
```log
2026-10-19 22:21:00.926 | INFO     | __main__:<module>:161 - args = Namespace(data_snapshot_filepath='data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_data_dim_2_n_train_test_eval_500000_200000_200000.pkl', use_gpu=True, vt_model='nn', vt_opt='adam', trajectory_model='global', ode_solver_method='dopri8', rectification_linearity='linear', reflow_iterations=0, nn_epochs=300, nn_hidden_dim=128, batch_size=2048, nn_depth=2, lr=0.001, nn_reg_lambda=0.001, patience=10, min_rel_loss_delta=5e-05, dry_run=False, prior='uniform', logs_dir='logs', output_model_dir='models/experimental')
 
```
# Model Paths
## 1-RecFLow
```log
2026-10-19 22:35:43.753 | INFO     | __main__:<module>:355 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-19T22:21:00.926949.pkl
```


# Model Training Log

## 1-RecFlow
```log
/home/mbaddar/Documents/mbaddar/ttflow/.venv/bin/python /home/mbaddar/Documents/mbaddar/ttflow/ttflow/training/nn/vt_trainer.py --data-snapshot-filepath data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_data_dim_2_n_train_test_eval_500000_200000_200000.pkl --vt-model nn --vt-opt adam --trajectory-model global --reflow-iterations 0 --nn-epochs 300 --reflow-iterations 0 --ode-solver-method dopri8 --prior uniform 
2026-10-19 22:21:00.926 | INFO     | __main__:<module>:161 - args = Namespace(data_snapshot_filepath='data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_data_dim_2_n_train_test_eval_500000_200000_200000.pkl', use_gpu=True, vt_model='nn', vt_opt='adam', trajectory_model='global', ode_solver_method='dopri8', rectification_linearity='linear', reflow_iterations=0, nn_epochs=300, nn_hidden_dim=128, batch_size=2048, nn_depth=2, lr=0.001, nn_reg_lambda=0.001, patience=10, min_rel_loss_delta=5e-05, dry_run=False, prior='uniform', logs_dir='logs', output_model_dir='models/experimental')
2026-10-19 22:21:00.927 | INFO     | __main__:<module>:181 - is GPU/CUDA available = True
2026-10-19 22:21:00.927 | INFO     | __main__:<module>:184 - device = cuda
2026-10-19 22:21:01.015 | INFO     | __main__:<module>:189 - Using data snapshot: data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_data_dim_2_n_train_test_eval_500000_200000_200000.pkl
2026-10-19 22:21:01.015 | INFO     | ttflow.utils:check_system_memory_and_swap:56 - ram = 30.96, swap = 40.0 GB
2026-10-19 22:21:01.031 | INFO     | __main__:<module>:214 - data pkl size = 0.01 GB
2026-10-19 22:21:01.048 | INFO     | __main__:<module>:236 - Generating training data for the global trajectory training process..
2026-10-19 22:21:01.050 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-19 22:21:01.821 | INFO     | __main__:<module>:256 - ***********************************************************Training Model Specs:
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
    - Reflow Iterations:        0
**********************************************************
reflow iterations:   0%|          | 0/1 [00:00<?, ?it/s]2026-10-19 22:21:01.821 | INFO     | __main__:<module>:281 - Reflow iteration: 0
2026-10-19 22:21:02.736 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:57 - @epoch = 0 (before training), loss_fn = MSELoss(), loss_value = 51.003950
2026-10-19 22:21:02.741 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:64 - MSELoss(test) before training:51.00394821166992

Epoch:   0%|          | 0/300 [00:00<?, ?it/s]2026-10-19 22:21:06.077 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 1, smoothed-train-loss = 9.798835 | smoothed-test-loss = 3.303702 | rel_loss_delta = inf | loss_stagnation_count = 0

Epoch:   0%|          | 1/300 [00:03<16:37,  3.34s/it]2026-10-19 22:21:09.057 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 2, smoothed-train-loss = 3.787786 | smoothed-test-loss = 3.067030 | rel_loss_delta = inf | loss_stagnation_count = 0

Epoch:   1%|          | 2/300 [00:06<15:31,  3.13s/it]2026-10-19 22:21:11.985 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 3, smoothed-train-loss = 3.122304 | smoothed-test-loss = 3.021821 | rel_loss_delta = -0.17569147655485368 | loss_stagnation_count = 0

Epoch:   1%|          | 3/300 [00:09<15:01,  3.04s/it]2026-10-19 22:21:14.804 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 4, smoothed-train-loss = 3.044598 | smoothed-test-loss = 3.019684 | rel_loss_delta = -0.024887469118537217 | loss_stagnation_count = 0

Epoch:   1%|▏         | 4/300 [00:12<14:33,  2.95s/it]2026-10-19 22:21:17.705 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 5, smoothed-train-loss = 3.029209 | smoothed-test-loss = 3.007020 | rel_loss_delta = -0.005054568340832129 | loss_stagnation_count = 0

Epoch:   2%|▏         | 5/300 [00:14<14:25,  2.93s/it]2026-10-19 22:21:20.600 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 6, smoothed-train-loss = 3.023649 | smoothed-test-loss = 2.997056 | rel_loss_delta = -0.0018353810833207306 | loss_stagnation_count = 0

Epoch:   2%|▏         | 6/300 [00:17<14:18,  2.92s/it]2026-10-19 22:21:23.521 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 7, smoothed-train-loss = 3.017751 | smoothed-test-loss = 2.993735 | rel_loss_delta = -0.0019504936155943149 | loss_stagnation_count = 0

Epoch:   2%|▏         | 7/300 [00:20<14:15,  2.92s/it]2026-10-19 22:21:26.454 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 8, smoothed-train-loss = 3.015610 | smoothed-test-loss = 2.995431 | rel_loss_delta = -0.0007096219311700717 | loss_stagnation_count = 0

Epoch:   3%|▎         | 8/300 [00:23<14:13,  2.92s/it]2026-10-19 22:21:29.406 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 9, smoothed-train-loss = 3.013851 | smoothed-test-loss = 2.991559 | rel_loss_delta = -0.0005834253962017829 | loss_stagnation_count = 0

Epoch:   3%|▎         | 9/300 [00:26<14:13,  2.93s/it]2026-10-19 22:21:32.277 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 10, smoothed-train-loss = 3.010674 | smoothed-test-loss = 2.994827 | rel_loss_delta = -0.0010538983491302188 | loss_stagnation_count = 0

Epoch:   3%|▎         | 10/300 [00:29<14:05,  2.91s/it]2026-10-19 22:21:35.079 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 11, smoothed-train-loss = 3.008469 | smoothed-test-loss = 2.986484 | rel_loss_delta = -0.0007324275870244316 | loss_stagnation_count = 0

Epoch:   4%|▎         | 11/300 [00:32<13:52,  2.88s/it]2026-10-19 22:21:38.086 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 12, smoothed-train-loss = 3.007434 | smoothed-test-loss = 2.992033 | rel_loss_delta = -0.0003439678187534017 | loss_stagnation_count = 0

Epoch:   4%|▍         | 12/300 [00:35<14:00,  2.92s/it]2026-10-19 22:21:41.153 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 13, smoothed-train-loss = 3.004191 | smoothed-test-loss = 2.986331 | rel_loss_delta = -0.0010785643292247412 | loss_stagnation_count = 0

Epoch:   4%|▍         | 13/300 [00:38<14:10,  2.96s/it]2026-10-19 22:21:44.153 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 14, smoothed-train-loss = 3.002972 | smoothed-test-loss = 2.984105 | rel_loss_delta = -0.0004057752965604648 | loss_stagnation_count = 0

Epoch:   5%|▍         | 14/300 [00:41<14:10,  2.97s/it]2026-10-19 22:21:47.058 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 15, smoothed-train-loss = 3.001786 | smoothed-test-loss = 2.982777 | rel_loss_delta = -0.00039484743163746994 | loss_stagnation_count = 0

Epoch:   5%|▌         | 15/300 [00:44<14:01,  2.95s/it]2026-10-19 22:21:50.047 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 16, smoothed-train-loss = 2.999915 | smoothed-test-loss = 2.983353 | rel_loss_delta = -0.0006233893293400826 | loss_stagnation_count = 0

Epoch:   5%|▌         | 16/300 [00:47<14:01,  2.96s/it]2026-10-19 22:21:52.911 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 17, smoothed-train-loss = 2.998028 | smoothed-test-loss = 2.974457 | rel_loss_delta = -0.0006289027754865997 | loss_stagnation_count = 0

Epoch:   6%|▌         | 17/300 [00:50<13:50,  2.93s/it]2026-10-19 22:21:55.885 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 18, smoothed-train-loss = 2.995693 | smoothed-test-loss = 2.978176 | rel_loss_delta = -0.0007788863424931835 | loss_stagnation_count = 0

Epoch:   6%|▌         | 18/300 [00:53<13:50,  2.95s/it]2026-10-19 22:21:58.790 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 19, smoothed-train-loss = 2.994601 | smoothed-test-loss = 2.968999 | rel_loss_delta = -0.00036457409930794365 | loss_stagnation_count = 0

Epoch:   6%|▋         | 19/300 [00:56<13:44,  2.93s/it]2026-10-19 22:22:01.676 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 20, smoothed-train-loss = 2.991222 | smoothed-test-loss = 2.968121 | rel_loss_delta = -0.0011280771494224376 | loss_stagnation_count = 0

Epoch:   7%|▋         | 20/300 [00:58<13:37,  2.92s/it]2026-10-19 22:22:04.559 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 21, smoothed-train-loss = 2.988700 | smoothed-test-loss = 2.973112 | rel_loss_delta = -0.0008434634411591822 | loss_stagnation_count = 0

Epoch:   7%|▋         | 21/300 [01:01<13:31,  2.91s/it]2026-10-19 22:22:07.582 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 22, smoothed-train-loss = 2.985974 | smoothed-test-loss = 2.969044 | rel_loss_delta = -0.0009118889291836393 | loss_stagnation_count = 0

Epoch:   7%|▋         | 22/300 [01:04<13:38,  2.94s/it]2026-10-19 22:22:10.431 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 23, smoothed-train-loss = 2.984834 | smoothed-test-loss = 2.957987 | rel_loss_delta = -0.0003817816632511191 | loss_stagnation_count = 0

Epoch:   8%|▊         | 23/300 [01:07<13:27,  2.91s/it]2026-10-19 22:22:13.295 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 24, smoothed-train-loss = 2.981347 | smoothed-test-loss = 2.954426 | rel_loss_delta = -0.0011683429511390958 | loss_stagnation_count = 0

Epoch:   8%|▊         | 24/300 [01:10<13:20,  2.90s/it]2026-10-19 22:22:16.385 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 25, smoothed-train-loss = 2.979214 | smoothed-test-loss = 2.952812 | rel_loss_delta = -0.0007153644560100585 | loss_stagnation_count = 0

Epoch:   8%|▊         | 25/300 [01:13<13:33,  2.96s/it]2026-10-19 22:22:19.383 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 26, smoothed-train-loss = 2.975569 | smoothed-test-loss = 2.952219 | rel_loss_delta = -0.0012233548205733355 | loss_stagnation_count = 0

Epoch:   9%|▊         | 26/300 [01:16<13:33,  2.97s/it]2026-10-19 22:22:22.279 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 27, smoothed-train-loss = 2.974798 | smoothed-test-loss = 2.943592 | rel_loss_delta = -0.00025932838086073343 | loss_stagnation_count = 0

Epoch:   9%|▉         | 27/300 [01:19<13:24,  2.95s/it]2026-10-19 22:22:25.189 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 28, smoothed-train-loss = 2.970326 | smoothed-test-loss = 2.944360 | rel_loss_delta = -0.0015030823294843677 | loss_stagnation_count = 0

Epoch:   9%|▉         | 28/300 [01:22<13:18,  2.94s/it]2026-10-19 22:22:28.211 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 29, smoothed-train-loss = 2.967680 | smoothed-test-loss = 2.943569 | rel_loss_delta = -0.0008911267575932555 | loss_stagnation_count = 0

Epoch:  10%|▉         | 29/300 [01:25<13:22,  2.96s/it]2026-10-19 22:22:31.241 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 30, smoothed-train-loss = 2.963453 | smoothed-test-loss = 2.953601 | rel_loss_delta = -0.0014241728496132595 | loss_stagnation_count = 0

Epoch:  10%|█         | 30/300 [01:28<13:25,  2.98s/it]2026-10-19 22:22:34.178 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 31, smoothed-train-loss = 2.959207 | smoothed-test-loss = 2.937871 | rel_loss_delta = -0.0014326634947123642 | loss_stagnation_count = 0

Epoch:  10%|█         | 31/300 [01:31<13:18,  2.97s/it]2026-10-19 22:22:37.216 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 32, smoothed-train-loss = 2.957435 | smoothed-test-loss = 2.936146 | rel_loss_delta = -0.0005990408028821664 | loss_stagnation_count = 0

Epoch:  11%|█         | 32/300 [01:34<13:21,  2.99s/it]2026-10-19 22:22:40.199 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 33, smoothed-train-loss = 2.954127 | smoothed-test-loss = 2.932810 | rel_loss_delta = -0.0011183138481253478 | loss_stagnation_count = 0

Epoch:  11%|█         | 33/300 [01:37<13:17,  2.99s/it]2026-10-19 22:22:43.230 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 34, smoothed-train-loss = 2.951591 | smoothed-test-loss = 2.933943 | rel_loss_delta = -0.0008587448864931317 | loss_stagnation_count = 0

Epoch:  11%|█▏        | 34/300 [01:40<13:18,  3.00s/it]2026-10-19 22:22:46.169 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 35, smoothed-train-loss = 2.947923 | smoothed-test-loss = 2.931859 | rel_loss_delta = -0.0012424941949374583 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 35/300 [01:43<13:10,  2.98s/it]2026-10-19 22:22:49.217 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 36, smoothed-train-loss = 2.946115 | smoothed-test-loss = 2.915694 | rel_loss_delta = -0.0006132910707726557 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 36/300 [01:46<13:12,  3.00s/it]2026-10-19 22:22:52.244 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 37, smoothed-train-loss = 2.941433 | smoothed-test-loss = 2.920850 | rel_loss_delta = -0.0015894041974214097 | loss_stagnation_count = 0

Epoch:  12%|█▏        | 37/300 [01:49<13:11,  3.01s/it]2026-10-19 22:22:55.131 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 38, smoothed-train-loss = 2.936544 | smoothed-test-loss = 2.916035 | rel_loss_delta = -0.0016619243968395987 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 38/300 [01:52<12:58,  2.97s/it]2026-10-19 22:22:58.058 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 39, smoothed-train-loss = 2.934365 | smoothed-test-loss = 2.910688 | rel_loss_delta = -0.0007420117929141204 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 39/300 [01:55<12:52,  2.96s/it]2026-10-19 22:23:01.130 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 40, smoothed-train-loss = 2.931531 | smoothed-test-loss = 2.911668 | rel_loss_delta = -0.0009657700630843472 | loss_stagnation_count = 0

Epoch:  13%|█▎        | 40/300 [01:58<12:58,  2.99s/it]2026-10-19 22:23:04.088 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 41, smoothed-train-loss = 2.927860 | smoothed-test-loss = 2.907923 | rel_loss_delta = -0.0012522954068683904 | loss_stagnation_count = 0

Epoch:  14%|█▎        | 41/300 [02:01<12:52,  2.98s/it]2026-10-19 22:23:06.989 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 42, smoothed-train-loss = 2.924363 | smoothed-test-loss = 2.902554 | rel_loss_delta = -0.0011945574626756063 | loss_stagnation_count = 0

Epoch:  14%|█▍        | 42/300 [02:04<12:43,  2.96s/it]2026-10-19 22:23:09.918 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 43, smoothed-train-loss = 2.918747 | smoothed-test-loss = 2.917418 | rel_loss_delta = -0.0019201611304407236 | loss_stagnation_count = 0

Epoch:  14%|█▍        | 43/300 [02:07<12:37,  2.95s/it]2026-10-19 22:23:12.826 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 44, smoothed-train-loss = 2.916476 | smoothed-test-loss = 2.895005 | rel_loss_delta = -0.0007782460908581673 | loss_stagnation_count = 0

Epoch:  15%|█▍        | 44/300 [02:10<12:31,  2.94s/it]2026-10-19 22:23:15.719 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 45, smoothed-train-loss = 2.914064 | smoothed-test-loss = 2.899925 | rel_loss_delta = -0.0008271593287244737 | loss_stagnation_count = 0

Epoch:  15%|█▌        | 45/300 [02:12<12:25,  2.92s/it]2026-10-19 22:23:18.584 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 46, smoothed-train-loss = 2.909401 | smoothed-test-loss = 2.881474 | rel_loss_delta = -0.0016001492388488462 | loss_stagnation_count = 0

Epoch:  15%|█▌        | 46/300 [02:15<12:18,  2.91s/it]2026-10-19 22:23:21.351 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 47, smoothed-train-loss = 2.904387 | smoothed-test-loss = 2.881573 | rel_loss_delta = -0.0017233857732096639 | loss_stagnation_count = 0

Epoch:  16%|█▌        | 47/300 [02:18<12:04,  2.86s/it]2026-10-19 22:23:24.223 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 48, smoothed-train-loss = 2.900684 | smoothed-test-loss = 2.882638 | rel_loss_delta = -0.0012747701686932262 | loss_stagnation_count = 0

Epoch:  16%|█▌        | 48/300 [02:21<12:02,  2.87s/it]2026-10-19 22:23:27.115 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 49, smoothed-train-loss = 2.897925 | smoothed-test-loss = 2.882802 | rel_loss_delta = -0.0009513446975010451 | loss_stagnation_count = 0

Epoch:  16%|█▋        | 49/300 [02:24<12:01,  2.87s/it]2026-10-19 22:23:30.144 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 50, smoothed-train-loss = 2.893950 | smoothed-test-loss = 2.881920 | rel_loss_delta = -0.0013714792604880485 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 50/300 [02:27<12:10,  2.92s/it]2026-10-19 22:23:32.967 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 51, smoothed-train-loss = 2.891664 | smoothed-test-loss = 2.881321 | rel_loss_delta = -0.0007898734643124975 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 51/300 [02:30<11:59,  2.89s/it]2026-10-19 22:23:35.980 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 52, smoothed-train-loss = 2.887495 | smoothed-test-loss = 2.877416 | rel_loss_delta = -0.0014418831382067222 | loss_stagnation_count = 0

Epoch:  17%|█▋        | 52/300 [02:33<12:06,  2.93s/it]2026-10-19 22:23:38.887 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 53, smoothed-train-loss = 2.885063 | smoothed-test-loss = 2.861467 | rel_loss_delta = -0.0008420813771296923 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 53/300 [02:36<12:01,  2.92s/it]2026-10-19 22:23:41.799 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 54, smoothed-train-loss = 2.881786 | smoothed-test-loss = 2.856807 | rel_loss_delta = -0.0011360816776447404 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 54/300 [02:39<11:58,  2.92s/it]2026-10-19 22:23:44.602 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 55, smoothed-train-loss = 2.879169 | smoothed-test-loss = 2.869923 | rel_loss_delta = -0.0009080103649713843 | loss_stagnation_count = 0

Epoch:  18%|█▊        | 55/300 [02:41<11:46,  2.88s/it]2026-10-19 22:23:47.474 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 56, smoothed-train-loss = 2.875426 | smoothed-test-loss = 2.861127 | rel_loss_delta = -0.0012999328344131143 | loss_stagnation_count = 0

Epoch:  19%|█▊        | 56/300 [02:44<11:42,  2.88s/it]2026-10-19 22:23:50.375 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 57, smoothed-train-loss = 2.871110 | smoothed-test-loss = 2.859277 | rel_loss_delta = -0.0015010077149926239 | loss_stagnation_count = 0

Epoch:  19%|█▉        | 57/300 [02:47<11:41,  2.89s/it]2026-10-19 22:23:53.239 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 58, smoothed-train-loss = 2.866819 | smoothed-test-loss = 2.853380 | rel_loss_delta = -0.0014946434193899778 | loss_stagnation_count = 0

Epoch:  19%|█▉        | 58/300 [02:50<11:36,  2.88s/it]2026-10-19 22:23:56.068 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 59, smoothed-train-loss = 2.865682 | smoothed-test-loss = 2.854350 | rel_loss_delta = -0.0003967619664454096 | loss_stagnation_count = 0

Epoch:  20%|█▉        | 59/300 [02:53<11:30,  2.86s/it]2026-10-19 22:23:58.940 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 60, smoothed-train-loss = 2.863006 | smoothed-test-loss = 2.849524 | rel_loss_delta = -0.0009337073134183991 | loss_stagnation_count = 0

Epoch:  20%|██        | 60/300 [02:56<11:28,  2.87s/it]2026-10-19 22:24:01.812 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 61, smoothed-train-loss = 2.859174 | smoothed-test-loss = 2.849671 | rel_loss_delta = -0.0013382311252876486 | loss_stagnation_count = 0

Epoch:  20%|██        | 61/300 [02:59<11:25,  2.87s/it]2026-10-19 22:24:04.688 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 62, smoothed-train-loss = 2.857198 | smoothed-test-loss = 2.853652 | rel_loss_delta = -0.000691337185598679 | loss_stagnation_count = 0

Epoch:  21%|██        | 62/300 [03:01<11:23,  2.87s/it]2026-10-19 22:24:07.589 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 63, smoothed-train-loss = 2.855237 | smoothed-test-loss = 2.831499 | rel_loss_delta = -0.0006861824649512183 | loss_stagnation_count = 0

Epoch:  21%|██        | 63/300 [03:04<11:22,  2.88s/it]2026-10-19 22:24:10.519 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 64, smoothed-train-loss = 2.852469 | smoothed-test-loss = 2.835746 | rel_loss_delta = -0.0009694078443477742 | loss_stagnation_count = 0

Epoch:  21%|██▏       | 64/300 [03:07<11:23,  2.89s/it]2026-10-19 22:24:13.540 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 65, smoothed-train-loss = 2.847785 | smoothed-test-loss = 2.830240 | rel_loss_delta = -0.0016423587928230217 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 65/300 [03:10<11:29,  2.93s/it]2026-10-19 22:24:16.556 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 66, smoothed-train-loss = 2.846420 | smoothed-test-loss = 2.826261 | rel_loss_delta = -0.0004793153863858893 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 66/300 [03:13<11:32,  2.96s/it]2026-10-19 22:24:19.476 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 67, smoothed-train-loss = 2.844768 | smoothed-test-loss = 2.836972 | rel_loss_delta = -0.0005801472375183144 | loss_stagnation_count = 0

Epoch:  22%|██▏       | 67/300 [03:16<11:26,  2.95s/it]2026-10-19 22:24:22.548 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 68, smoothed-train-loss = 2.841869 | smoothed-test-loss = 2.811621 | rel_loss_delta = -0.0010190242532173586 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 68/300 [03:19<11:32,  2.98s/it]2026-10-19 22:24:25.507 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 69, smoothed-train-loss = 2.837661 | smoothed-test-loss = 2.823254 | rel_loss_delta = -0.0014807020952008916 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 69/300 [03:22<11:27,  2.98s/it]2026-10-19 22:24:28.549 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 70, smoothed-train-loss = 2.835534 | smoothed-test-loss = 2.826466 | rel_loss_delta = -0.0007497077671048462 | loss_stagnation_count = 0

Epoch:  23%|██▎       | 70/300 [03:25<11:29,  3.00s/it]2026-10-19 22:24:31.395 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 71, smoothed-train-loss = 2.834152 | smoothed-test-loss = 2.811065 | rel_loss_delta = -0.00048743731841822925 | loss_stagnation_count = 0

Epoch:  24%|██▎       | 71/300 [03:28<11:15,  2.95s/it]2026-10-19 22:24:34.420 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 72, smoothed-train-loss = 2.829962 | smoothed-test-loss = 2.813293 | rel_loss_delta = -0.0014783259525373465 | loss_stagnation_count = 0

Epoch:  24%|██▍       | 72/300 [03:31<11:17,  2.97s/it]2026-10-19 22:24:37.308 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 73, smoothed-train-loss = 2.830021 | smoothed-test-loss = 2.824077 | rel_loss_delta = 2.0856608172741732e-05 | loss_stagnation_count = 1

Epoch:  24%|██▍       | 73/300 [03:34<11:09,  2.95s/it]2026-10-19 22:24:40.414 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 74, smoothed-train-loss = 2.828928 | smoothed-test-loss = 2.823229 | rel_loss_delta = -0.00038628867372755675 | loss_stagnation_count = 0

Epoch:  25%|██▍       | 74/300 [03:37<11:16,  3.00s/it]2026-10-19 22:24:43.361 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 75, smoothed-train-loss = 2.825163 | smoothed-test-loss = 2.804439 | rel_loss_delta = -0.0013306950773649526 | loss_stagnation_count = 0

Epoch:  25%|██▌       | 75/300 [03:40<11:10,  2.98s/it]2026-10-19 22:24:46.420 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 76, smoothed-train-loss = 2.820435 | smoothed-test-loss = 2.794801 | rel_loss_delta = -0.001673635327573986 | loss_stagnation_count = 0

Epoch:  25%|██▌       | 76/300 [03:43<11:12,  3.00s/it]2026-10-19 22:24:49.302 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 77, smoothed-train-loss = 2.819308 | smoothed-test-loss = 2.808437 | rel_loss_delta = -0.00039974085975647743 | loss_stagnation_count = 0

Epoch:  26%|██▌       | 77/300 [03:46<11:01,  2.97s/it]2026-10-19 22:24:52.259 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 78, smoothed-train-loss = 2.817012 | smoothed-test-loss = 2.813907 | rel_loss_delta = -0.0008144267094728655 | loss_stagnation_count = 0

Epoch:  26%|██▌       | 78/300 [03:49<10:58,  2.96s/it]2026-10-19 22:24:55.065 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 79, smoothed-train-loss = 2.814308 | smoothed-test-loss = 2.849684 | rel_loss_delta = -0.0009596566195155258 | loss_stagnation_count = 0

Epoch:  26%|██▋       | 79/300 [03:52<10:44,  2.92s/it]2026-10-19 22:24:58.100 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 80, smoothed-train-loss = 2.813482 | smoothed-test-loss = 2.794884 | rel_loss_delta = -0.00029363382570233144 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 80/300 [03:55<10:49,  2.95s/it]2026-10-19 22:25:01.088 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 81, smoothed-train-loss = 2.808672 | smoothed-test-loss = 2.797384 | rel_loss_delta = -0.0017094137617147567 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 81/300 [03:58<10:48,  2.96s/it]2026-10-19 22:25:04.135 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 82, smoothed-train-loss = 2.805167 | smoothed-test-loss = 2.790352 | rel_loss_delta = -0.00124815179089521 | loss_stagnation_count = 0

Epoch:  27%|██▋       | 82/300 [04:01<10:51,  2.99s/it]2026-10-19 22:25:06.918 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 83, smoothed-train-loss = 2.802865 | smoothed-test-loss = 2.798693 | rel_loss_delta = -0.0008207127309977689 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 83/300 [04:04<10:35,  2.93s/it]2026-10-19 22:25:09.863 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 84, smoothed-train-loss = 2.802156 | smoothed-test-loss = 2.783653 | rel_loss_delta = -0.00025270235136219075 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 84/300 [04:07<10:33,  2.93s/it]2026-10-19 22:25:12.740 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 85, smoothed-train-loss = 2.799150 | smoothed-test-loss = 2.778511 | rel_loss_delta = -0.0010727262896694583 | loss_stagnation_count = 0

Epoch:  28%|██▊       | 85/300 [04:09<10:26,  2.92s/it]2026-10-19 22:25:15.795 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 86, smoothed-train-loss = 2.798446 | smoothed-test-loss = 2.773340 | rel_loss_delta = -0.0002517088009059701 | loss_stagnation_count = 0

Epoch:  29%|██▊       | 86/300 [04:13<10:32,  2.96s/it]2026-10-19 22:25:18.742 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 87, smoothed-train-loss = 2.795521 | smoothed-test-loss = 2.784413 | rel_loss_delta = -0.0010451457760937144 | loss_stagnation_count = 0

Epoch:  29%|██▉       | 87/300 [04:16<10:29,  2.95s/it]2026-10-19 22:25:21.745 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 88, smoothed-train-loss = 2.795703 | smoothed-test-loss = 2.770467 | rel_loss_delta = 6.501594594125179e-05 | loss_stagnation_count = 0

Epoch:  29%|██▉       | 88/300 [04:19<10:29,  2.97s/it]2026-10-19 22:25:24.817 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 89, smoothed-train-loss = 2.792469 | smoothed-test-loss = 2.770425 | rel_loss_delta = -0.0011566686325942765 | loss_stagnation_count = 0

Epoch:  30%|██▉       | 89/300 [04:22<10:32,  3.00s/it]2026-10-19 22:25:27.826 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 90, smoothed-train-loss = 2.791663 | smoothed-test-loss = 2.764502 | rel_loss_delta = -0.0002887346800108865 | loss_stagnation_count = 0

Epoch:  30%|███       | 90/300 [04:25<10:30,  3.00s/it]2026-10-19 22:25:30.746 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 91, smoothed-train-loss = 2.791928 | smoothed-test-loss = 2.766991 | rel_loss_delta = 9.485515053376248e-05 | loss_stagnation_count = 0

Epoch:  30%|███       | 91/300 [04:28<10:22,  2.98s/it]2026-10-19 22:25:33.791 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 92, smoothed-train-loss = 2.789244 | smoothed-test-loss = 2.771934 | rel_loss_delta = -0.0009611261678824616 | loss_stagnation_count = 0

Epoch:  31%|███       | 92/300 [04:31<10:23,  3.00s/it]2026-10-19 22:25:36.809 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 93, smoothed-train-loss = 2.788754 | smoothed-test-loss = 2.776305 | rel_loss_delta = -0.0001757469095085709 | loss_stagnation_count = 0

Epoch:  31%|███       | 93/300 [04:34<10:21,  3.00s/it]2026-10-19 22:25:39.880 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 94, smoothed-train-loss = 2.787074 | smoothed-test-loss = 2.786692 | rel_loss_delta = -0.0006023707582769468 | loss_stagnation_count = 0

Epoch:  31%|███▏      | 94/300 [04:37<10:22,  3.02s/it]2026-10-19 22:25:42.810 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 95, smoothed-train-loss = 2.786752 | smoothed-test-loss = 2.778175 | rel_loss_delta = -0.00011545072442196702 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 95/300 [04:40<10:14,  3.00s/it]2026-10-19 22:25:45.816 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 96, smoothed-train-loss = 2.785022 | smoothed-test-loss = 2.759793 | rel_loss_delta = -0.0006208915308916317 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 96/300 [04:43<10:11,  3.00s/it]2026-10-19 22:25:48.756 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 97, smoothed-train-loss = 2.782847 | smoothed-test-loss = 2.778900 | rel_loss_delta = -0.000781076699206004 | loss_stagnation_count = 0

Epoch:  32%|███▏      | 97/300 [04:46<10:05,  2.98s/it]2026-10-19 22:25:51.745 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 98, smoothed-train-loss = 2.783164 | smoothed-test-loss = 2.788880 | rel_loss_delta = 0.00011409434984543177 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 98/300 [04:49<10:02,  2.98s/it]2026-10-19 22:25:54.616 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 99, smoothed-train-loss = 2.781076 | smoothed-test-loss = 2.767969 | rel_loss_delta = -0.0007502514037193152 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 99/300 [04:51<09:52,  2.95s/it]2026-10-19 22:25:57.580 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 100, smoothed-train-loss = 2.778610 | smoothed-test-loss = 2.774141 | rel_loss_delta = -0.0008867060324330882 | loss_stagnation_count = 0

Epoch:  33%|███▎      | 100/300 [04:54<09:50,  2.95s/it]2026-10-19 22:26:00.518 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 101, smoothed-train-loss = 2.780197 | smoothed-test-loss = 2.763513 | rel_loss_delta = 0.0005710152651875866 | loss_stagnation_count = 0

Epoch:  34%|███▎      | 101/300 [04:57<09:46,  2.95s/it]2026-10-19 22:26:03.458 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 102, smoothed-train-loss = 2.777631 | smoothed-test-loss = 2.766949 | rel_loss_delta = -0.000923038262895915 | loss_stagnation_count = 0

Epoch:  34%|███▍      | 102/300 [05:00<09:43,  2.95s/it]2026-10-19 22:26:06.343 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 103, smoothed-train-loss = 2.778259 | smoothed-test-loss = 2.756528 | rel_loss_delta = 0.0002262863865478744 | loss_stagnation_count = 0

Epoch:  34%|███▍      | 103/300 [05:03<09:36,  2.93s/it]2026-10-19 22:26:09.430 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 104, smoothed-train-loss = 2.776166 | smoothed-test-loss = 2.760627 | rel_loss_delta = -0.0007533407302785604 | loss_stagnation_count = 0

Epoch:  35%|███▍      | 104/300 [05:06<09:43,  2.98s/it]2026-10-19 22:26:12.482 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 105, smoothed-train-loss = 2.775329 | smoothed-test-loss = 2.782225 | rel_loss_delta = -0.00030139207454981694 | loss_stagnation_count = 0

Epoch:  35%|███▌      | 105/300 [05:09<09:44,  3.00s/it]2026-10-19 22:26:15.585 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 106, smoothed-train-loss = 2.773781 | smoothed-test-loss = 2.764173 | rel_loss_delta = -0.0005579123917894645 | loss_stagnation_count = 0

Epoch:  35%|███▌      | 106/300 [05:12<09:47,  3.03s/it]2026-10-19 22:26:18.521 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 107, smoothed-train-loss = 2.772584 | smoothed-test-loss = 2.759294 | rel_loss_delta = -0.0004315897846498085 | loss_stagnation_count = 0

Epoch:  36%|███▌      | 107/300 [05:15<09:39,  3.00s/it]2026-10-19 22:26:21.558 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 108, smoothed-train-loss = 2.770228 | smoothed-test-loss = 2.766078 | rel_loss_delta = -0.0008495350728247871 | loss_stagnation_count = 0

Epoch:  36%|███▌      | 108/300 [05:18<09:38,  3.01s/it]2026-10-19 22:26:24.529 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 109, smoothed-train-loss = 2.772567 | smoothed-test-loss = 2.754000 | rel_loss_delta = 0.0008440346887113767 | loss_stagnation_count = 0

Epoch:  36%|███▋      | 109/300 [05:21<09:33,  3.00s/it]2026-10-19 22:26:27.542 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 110, smoothed-train-loss = 2.771695 | smoothed-test-loss = 2.751309 | rel_loss_delta = -0.00031425576419925736 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 110/300 [05:24<09:30,  3.00s/it]2026-10-19 22:26:30.381 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 111, smoothed-train-loss = 2.769738 | smoothed-test-loss = 2.753530 | rel_loss_delta = -0.0007063651279055704 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 111/300 [05:27<09:18,  2.95s/it]2026-10-19 22:26:33.335 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 112, smoothed-train-loss = 2.769889 | smoothed-test-loss = 2.764679 | rel_loss_delta = 5.46761178677601e-05 | loss_stagnation_count = 0

Epoch:  37%|███▋      | 112/300 [05:30<09:15,  2.95s/it]2026-10-19 22:26:36.359 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 113, smoothed-train-loss = 2.771824 | smoothed-test-loss = 2.747695 | rel_loss_delta = 0.0006984224623258716 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 113/300 [05:33<09:16,  2.98s/it]2026-10-19 22:26:39.379 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 114, smoothed-train-loss = 2.766877 | smoothed-test-loss = 2.763522 | rel_loss_delta = -0.0017844627315815494 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 114/300 [05:36<09:15,  2.99s/it]2026-10-19 22:26:42.220 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 115, smoothed-train-loss = 2.768634 | smoothed-test-loss = 2.753734 | rel_loss_delta = 0.0006350651641105004 | loss_stagnation_count = 0

Epoch:  38%|███▊      | 115/300 [05:39<09:04,  2.94s/it]2026-10-19 22:26:45.156 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 116, smoothed-train-loss = 2.768846 | smoothed-test-loss = 2.762914 | rel_loss_delta = 7.635197591056849e-05 | loss_stagnation_count = 0

Epoch:  39%|███▊      | 116/300 [05:42<09:01,  2.94s/it]2026-10-19 22:26:48.165 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 117, smoothed-train-loss = 2.766826 | smoothed-test-loss = 2.744391 | rel_loss_delta = -0.0007294292320636954 | loss_stagnation_count = 0

Epoch:  39%|███▉      | 117/300 [05:45<09:02,  2.96s/it]2026-10-19 22:26:51.163 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 118, smoothed-train-loss = 2.764265 | smoothed-test-loss = 2.748001 | rel_loss_delta = -0.0009256920844586174 | loss_stagnation_count = 0

Epoch:  39%|███▉      | 118/300 [05:48<09:01,  2.97s/it]2026-10-19 22:26:54.030 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 119, smoothed-train-loss = 2.765722 | smoothed-test-loss = 2.753605 | rel_loss_delta = 0.0005269346120844155 | loss_stagnation_count = 0

Epoch:  40%|███▉      | 119/300 [05:51<08:52,  2.94s/it]2026-10-19 22:26:57.032 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 120, smoothed-train-loss = 2.764567 | smoothed-test-loss = 2.742089 | rel_loss_delta = -0.0004172999610984523 | loss_stagnation_count = 0

Epoch:  40%|████      | 120/300 [05:54<08:52,  2.96s/it]2026-10-19 22:27:00.024 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 121, smoothed-train-loss = 2.763977 | smoothed-test-loss = 2.755965 | rel_loss_delta = -0.00021372239420482063 | loss_stagnation_count = 0

Epoch:  40%|████      | 121/300 [05:57<08:51,  2.97s/it]2026-10-19 22:27:02.964 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 122, smoothed-train-loss = 2.762143 | smoothed-test-loss = 2.761110 | rel_loss_delta = -0.0006635189287729319 | loss_stagnation_count = 0

Epoch:  41%|████      | 122/300 [06:00<08:46,  2.96s/it]2026-10-19 22:27:05.888 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 123, smoothed-train-loss = 2.763789 | smoothed-test-loss = 2.743370 | rel_loss_delta = 0.0005959875362310841 | loss_stagnation_count = 0

Epoch:  41%|████      | 123/300 [06:03<08:42,  2.95s/it]2026-10-19 22:27:08.872 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 124, smoothed-train-loss = 2.759577 | smoothed-test-loss = 2.751618 | rel_loss_delta = -0.0015239057050805025 | loss_stagnation_count = 0

Epoch:  41%|████▏     | 124/300 [06:06<08:40,  2.96s/it]2026-10-19 22:27:11.745 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 125, smoothed-train-loss = 2.760606 | smoothed-test-loss = 2.753682 | rel_loss_delta = 0.0003729727157422374 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 125/300 [06:09<08:33,  2.93s/it]2026-10-19 22:27:14.725 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 126, smoothed-train-loss = 2.760199 | smoothed-test-loss = 2.742807 | rel_loss_delta = -0.00014740709115029207 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 126/300 [06:11<08:32,  2.95s/it]2026-10-19 22:27:17.476 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 127, smoothed-train-loss = 2.761425 | smoothed-test-loss = 2.739906 | rel_loss_delta = 0.0004440681424327597 | loss_stagnation_count = 0

Epoch:  42%|████▏     | 127/300 [06:14<08:19,  2.89s/it]2026-10-19 22:27:20.335 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 128, smoothed-train-loss = 2.757566 | smoothed-test-loss = 2.755507 | rel_loss_delta = -0.0013974479113548338 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 128/300 [06:17<08:15,  2.88s/it]2026-10-19 22:27:23.347 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 129, smoothed-train-loss = 2.756661 | smoothed-test-loss = 2.777051 | rel_loss_delta = -0.00032835597192811 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 129/300 [06:20<08:19,  2.92s/it]2026-10-19 22:27:26.374 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 130, smoothed-train-loss = 2.759565 | smoothed-test-loss = 2.740279 | rel_loss_delta = 0.0010536040724398647 | loss_stagnation_count = 0

Epoch:  43%|████▎     | 130/300 [06:23<08:21,  2.95s/it]2026-10-19 22:27:29.303 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 131, smoothed-train-loss = 2.757575 | smoothed-test-loss = 2.740644 | rel_loss_delta = -0.0007211078309512819 | loss_stagnation_count = 0

Epoch:  44%|████▎     | 131/300 [06:26<08:17,  2.94s/it]2026-10-19 22:27:32.309 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 132, smoothed-train-loss = 2.757491 | smoothed-test-loss = 2.737317 | rel_loss_delta = -3.0507569067606583e-05 | loss_stagnation_count = 1

Epoch:  44%|████▍     | 132/300 [06:29<08:17,  2.96s/it]2026-10-19 22:27:35.190 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 133, smoothed-train-loss = 2.755892 | smoothed-test-loss = 2.740564 | rel_loss_delta = -0.0005798615762017465 | loss_stagnation_count = 0

Epoch:  44%|████▍     | 133/300 [06:32<08:10,  2.94s/it]2026-10-19 22:27:38.026 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 134, smoothed-train-loss = 2.758530 | smoothed-test-loss = 2.754622 | rel_loss_delta = 0.00095712151646808 | loss_stagnation_count = 0

Epoch:  45%|████▍     | 134/300 [06:35<08:02,  2.91s/it]2026-10-19 22:27:40.896 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 135, smoothed-train-loss = 2.757817 | smoothed-test-loss = 2.745173 | rel_loss_delta = -0.00025847011529532316 | loss_stagnation_count = 0

Epoch:  45%|████▌     | 135/300 [06:38<07:57,  2.90s/it]2026-10-19 22:27:43.945 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 136, smoothed-train-loss = 2.755947 | smoothed-test-loss = 2.735657 | rel_loss_delta = -0.0006778985903375535 | loss_stagnation_count = 0

Epoch:  45%|████▌     | 136/300 [06:41<08:02,  2.94s/it]2026-10-19 22:27:46.960 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 137, smoothed-train-loss = 2.754123 | smoothed-test-loss = 2.757537 | rel_loss_delta = -0.000661771952425443 | loss_stagnation_count = 0

Epoch:  46%|████▌     | 137/300 [06:44<08:03,  2.96s/it]2026-10-19 22:27:49.928 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 138, smoothed-train-loss = 2.755725 | smoothed-test-loss = 2.739208 | rel_loss_delta = 0.0005814099771247491 | loss_stagnation_count = 0

Epoch:  46%|████▌     | 138/300 [06:47<08:00,  2.97s/it]2026-10-19 22:27:52.826 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 139, smoothed-train-loss = 2.753354 | smoothed-test-loss = 2.743239 | rel_loss_delta = -0.0008604636675223175 | loss_stagnation_count = 0

Epoch:  46%|████▋     | 139/300 [06:50<07:54,  2.95s/it]2026-10-19 22:27:55.757 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 140, smoothed-train-loss = 2.753351 | smoothed-test-loss = 2.770997 | rel_loss_delta = -1.0648234542668009e-06 | loss_stagnation_count = 1

Epoch:  47%|████▋     | 140/300 [06:53<07:50,  2.94s/it]2026-10-19 22:27:58.726 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 141, smoothed-train-loss = 2.752834 | smoothed-test-loss = 2.736945 | rel_loss_delta = -0.00018776852891246696 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 141/300 [06:55<07:48,  2.95s/it]2026-10-19 22:28:01.745 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 142, smoothed-train-loss = 2.751682 | smoothed-test-loss = 2.744826 | rel_loss_delta = -0.000418209197033635 | loss_stagnation_count = 0

Epoch:  47%|████▋     | 142/300 [06:59<07:49,  2.97s/it]2026-10-19 22:28:04.557 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 143, smoothed-train-loss = 2.753625 | smoothed-test-loss = 2.729075 | rel_loss_delta = 0.0007059161202662467 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 143/300 [07:01<07:38,  2.92s/it]2026-10-19 22:28:07.562 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 144, smoothed-train-loss = 2.751384 | smoothed-test-loss = 2.746591 | rel_loss_delta = -0.0008135797186309033 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 144/300 [07:04<07:39,  2.95s/it]2026-10-19 22:28:10.549 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 145, smoothed-train-loss = 2.751893 | smoothed-test-loss = 2.740712 | rel_loss_delta = 0.00018476772491205957 | loss_stagnation_count = 0

Epoch:  48%|████▊     | 145/300 [07:07<07:38,  2.96s/it]2026-10-19 22:28:13.459 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 146, smoothed-train-loss = 2.750167 | smoothed-test-loss = 2.739754 | rel_loss_delta = -0.000627288974859995 | loss_stagnation_count = 0

Epoch:  49%|████▊     | 146/300 [07:10<07:33,  2.94s/it]2026-10-19 22:28:16.333 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 147, smoothed-train-loss = 2.752087 | smoothed-test-loss = 2.725836 | rel_loss_delta = 0.0006981994070630955 | loss_stagnation_count = 0

Epoch:  49%|████▉     | 147/300 [07:13<07:27,  2.92s/it]2026-10-19 22:28:19.175 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 148, smoothed-train-loss = 2.749876 | smoothed-test-loss = 2.732581 | rel_loss_delta = -0.0008031899861859147 | loss_stagnation_count = 0

Epoch:  49%|████▉     | 148/300 [07:16<07:20,  2.90s/it]2026-10-19 22:28:22.059 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 149, smoothed-train-loss = 2.751745 | smoothed-test-loss = 2.733070 | rel_loss_delta = 0.0006795790804737623 | loss_stagnation_count = 0

Epoch:  50%|████▉     | 149/300 [07:19<07:17,  2.89s/it]2026-10-19 22:28:24.954 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 150, smoothed-train-loss = 2.750594 | smoothed-test-loss = 2.737373 | rel_loss_delta = -0.00041844014508293727 | loss_stagnation_count = 0

Epoch:  50%|█████     | 150/300 [07:22<07:14,  2.89s/it]2026-10-19 22:28:27.731 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 151, smoothed-train-loss = 2.753081 | smoothed-test-loss = 2.747745 | rel_loss_delta = 0.0009043676893865396 | loss_stagnation_count = 0

Epoch:  50%|█████     | 151/300 [07:24<07:06,  2.86s/it]2026-10-19 22:28:30.693 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 152, smoothed-train-loss = 2.750322 | smoothed-test-loss = 2.734040 | rel_loss_delta = -0.0010023181840759937 | loss_stagnation_count = 0

Epoch:  51%|█████     | 152/300 [07:27<07:07,  2.89s/it]2026-10-19 22:28:33.691 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 153, smoothed-train-loss = 2.748832 | smoothed-test-loss = 2.726460 | rel_loss_delta = -0.0005418343385707032 | loss_stagnation_count = 0

Epoch:  51%|█████     | 153/300 [07:30<07:09,  2.92s/it]2026-10-19 22:28:36.594 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 154, smoothed-train-loss = 2.749441 | smoothed-test-loss = 2.738453 | rel_loss_delta = 0.0002215430676806951 | loss_stagnation_count = 0

Epoch:  51%|█████▏    | 154/300 [07:33<07:05,  2.92s/it]2026-10-19 22:28:39.413 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 155, smoothed-train-loss = 2.748562 | smoothed-test-loss = 2.749705 | rel_loss_delta = -0.0003194366920167847 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 155/300 [07:36<06:58,  2.89s/it]2026-10-19 22:28:42.363 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 156, smoothed-train-loss = 2.749864 | smoothed-test-loss = 2.736669 | rel_loss_delta = 0.0004734343819438808 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 156/300 [07:39<06:58,  2.91s/it]2026-10-19 22:28:45.363 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 157, smoothed-train-loss = 2.747910 | smoothed-test-loss = 2.730484 | rel_loss_delta = -0.0007104354261643088 | loss_stagnation_count = 0

Epoch:  52%|█████▏    | 157/300 [07:42<06:59,  2.93s/it]2026-10-19 22:28:48.331 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 158, smoothed-train-loss = 2.748282 | smoothed-test-loss = 2.748894 | rel_loss_delta = 0.0001355440223467023 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 158/300 [07:45<06:58,  2.94s/it]2026-10-19 22:28:51.254 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 159, smoothed-train-loss = 2.749695 | smoothed-test-loss = 2.743757 | rel_loss_delta = 0.0005141034518782524 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 159/300 [07:48<06:54,  2.94s/it]2026-10-19 22:28:54.255 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 160, smoothed-train-loss = 2.747745 | smoothed-test-loss = 2.730952 | rel_loss_delta = -0.000709257661951646 | loss_stagnation_count = 0

Epoch:  53%|█████▎    | 160/300 [07:51<06:53,  2.96s/it]2026-10-19 22:28:57.259 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 161, smoothed-train-loss = 2.746760 | smoothed-test-loss = 2.761228 | rel_loss_delta = -0.0003585722558211792 | loss_stagnation_count = 0

Epoch:  54%|█████▎    | 161/300 [07:54<06:52,  2.97s/it]2026-10-19 22:29:00.180 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 162, smoothed-train-loss = 2.746797 | smoothed-test-loss = 2.735311 | rel_loss_delta = 1.3630211699460911e-05 | loss_stagnation_count = 1

Epoch:  54%|█████▍    | 162/300 [07:57<06:47,  2.96s/it]2026-10-19 22:29:03.018 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 163, smoothed-train-loss = 2.746416 | smoothed-test-loss = 2.744027 | rel_loss_delta = -0.00013885675233780016 | loss_stagnation_count = 0

Epoch:  54%|█████▍    | 163/300 [08:00<06:40,  2.92s/it]2026-10-19 22:29:05.873 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 164, smoothed-train-loss = 2.746720 | smoothed-test-loss = 2.735085 | rel_loss_delta = 0.00011061600206067825 | loss_stagnation_count = 0

Epoch:  55%|█████▍    | 164/300 [08:03<06:34,  2.90s/it]2026-10-19 22:29:08.733 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 165, smoothed-train-loss = 2.748056 | smoothed-test-loss = 2.741771 | rel_loss_delta = 0.00048647301142591297 | loss_stagnation_count = 0

Epoch:  55%|█████▌    | 165/300 [08:05<06:29,  2.89s/it]2026-10-19 22:29:11.599 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 166, smoothed-train-loss = 2.745976 | smoothed-test-loss = 2.728087 | rel_loss_delta = -0.0007566791728622867 | loss_stagnation_count = 0

Epoch:  55%|█████▌    | 166/300 [08:08<06:26,  2.88s/it]2026-10-19 22:29:14.390 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 167, smoothed-train-loss = 2.745871 | smoothed-test-loss = 2.745111 | rel_loss_delta = -3.831885871718178e-05 | loss_stagnation_count = 1

Epoch:  56%|█████▌    | 167/300 [08:11<06:19,  2.85s/it]2026-10-19 22:29:17.309 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 168, smoothed-train-loss = 2.746774 | smoothed-test-loss = 2.736262 | rel_loss_delta = 0.00032866663784365634 | loss_stagnation_count = 0

Epoch:  56%|█████▌    | 168/300 [08:14<06:19,  2.87s/it]2026-10-19 22:29:20.217 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 169, smoothed-train-loss = 2.748035 | smoothed-test-loss = 2.725043 | rel_loss_delta = 0.00045914166454380143 | loss_stagnation_count = 0

Epoch:  56%|█████▋    | 169/300 [08:17<06:17,  2.88s/it]2026-10-19 22:29:23.195 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 170, smoothed-train-loss = 2.747291 | smoothed-test-loss = 2.734038 | rel_loss_delta = -0.00027085005168683276 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 170/300 [08:20<06:18,  2.91s/it]2026-10-19 22:29:26.014 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 171, smoothed-train-loss = 2.746231 | smoothed-test-loss = 2.729146 | rel_loss_delta = -0.0003856873895344999 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 171/300 [08:23<06:12,  2.88s/it]2026-10-19 22:29:28.877 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 172, smoothed-train-loss = 2.744915 | smoothed-test-loss = 2.725129 | rel_loss_delta = -0.0004792589359485057 | loss_stagnation_count = 0

Epoch:  57%|█████▋    | 172/300 [08:26<06:08,  2.88s/it]2026-10-19 22:29:31.778 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 173, smoothed-train-loss = 2.745035 | smoothed-test-loss = 2.727577 | rel_loss_delta = 4.375943184297899e-05 | loss_stagnation_count = 1

Epoch:  58%|█████▊    | 173/300 [08:29<06:06,  2.88s/it]2026-10-19 22:29:34.754 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 174, smoothed-train-loss = 2.745012 | smoothed-test-loss = 2.735602 | rel_loss_delta = -8.466857595164122e-06 | loss_stagnation_count = 2

Epoch:  58%|█████▊    | 174/300 [08:32<06:06,  2.91s/it]2026-10-19 22:29:37.615 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 175, smoothed-train-loss = 2.746236 | smoothed-test-loss = 2.740463 | rel_loss_delta = 0.00044606782784089324 | loss_stagnation_count = 0

Epoch:  58%|█████▊    | 175/300 [08:34<06:02,  2.90s/it]2026-10-19 22:29:40.644 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 176, smoothed-train-loss = 2.743285 | smoothed-test-loss = 2.731343 | rel_loss_delta = -0.0010744698754013213 | loss_stagnation_count = 0

Epoch:  59%|█████▊    | 176/300 [08:37<06:04,  2.94s/it]2026-10-19 22:29:43.492 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 177, smoothed-train-loss = 2.746474 | smoothed-test-loss = 2.725784 | rel_loss_delta = 0.0011622324072729814 | loss_stagnation_count = 0

Epoch:  59%|█████▉    | 177/300 [08:40<05:57,  2.91s/it]2026-10-19 22:29:46.445 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 178, smoothed-train-loss = 2.744793 | smoothed-test-loss = 2.737198 | rel_loss_delta = -0.0006120626545958999 | loss_stagnation_count = 0

Epoch:  59%|█████▉    | 178/300 [08:43<05:56,  2.92s/it]2026-10-19 22:29:49.365 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 179, smoothed-train-loss = 2.744491 | smoothed-test-loss = 2.734995 | rel_loss_delta = -0.00010989478860848433 | loss_stagnation_count = 0

Epoch:  60%|█████▉    | 179/300 [08:46<05:53,  2.92s/it]2026-10-19 22:29:52.374 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 180, smoothed-train-loss = 2.746266 | smoothed-test-loss = 2.758719 | rel_loss_delta = 0.0006468455548452659 | loss_stagnation_count = 0

Epoch:  60%|██████    | 180/300 [08:49<05:53,  2.95s/it]2026-10-19 22:29:55.293 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 181, smoothed-train-loss = 2.744717 | smoothed-test-loss = 2.738313 | rel_loss_delta = -0.0005641990447264934 | loss_stagnation_count = 0

Epoch:  60%|██████    | 181/300 [08:52<05:49,  2.94s/it]2026-10-19 22:29:58.193 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 182, smoothed-train-loss = 2.743260 | smoothed-test-loss = 2.734522 | rel_loss_delta = -0.0005306869596342067 | loss_stagnation_count = 0

Epoch:  61%|██████    | 182/300 [08:55<05:45,  2.93s/it]2026-10-19 22:30:01.120 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 183, smoothed-train-loss = 2.743802 | smoothed-test-loss = 2.746786 | rel_loss_delta = 0.0001975389152201585 | loss_stagnation_count = 0

Epoch:  61%|██████    | 183/300 [08:58<05:42,  2.93s/it]2026-10-19 22:30:04.133 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 184, smoothed-train-loss = 2.744459 | smoothed-test-loss = 2.770003 | rel_loss_delta = 0.00023925193249240223 | loss_stagnation_count = 0

Epoch:  61%|██████▏   | 184/300 [09:01<05:42,  2.95s/it]2026-10-19 22:30:07.146 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 185, smoothed-train-loss = 2.745199 | smoothed-test-loss = 2.747659 | rel_loss_delta = 0.0002698055625834179 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 185/300 [09:04<05:41,  2.97s/it]2026-10-19 22:30:10.133 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 186, smoothed-train-loss = 2.743518 | smoothed-test-loss = 2.724099 | rel_loss_delta = -0.0006121964318619455 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 186/300 [09:07<05:39,  2.98s/it]2026-10-19 22:30:12.888 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 187, smoothed-train-loss = 2.744350 | smoothed-test-loss = 2.718751 | rel_loss_delta = 0.00030298409069618647 | loss_stagnation_count = 0

Epoch:  62%|██████▏   | 187/300 [09:10<05:28,  2.91s/it]2026-10-19 22:30:15.799 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 188, smoothed-train-loss = 2.742623 | smoothed-test-loss = 2.736139 | rel_loss_delta = -0.000629242981548381 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 188/300 [09:13<05:25,  2.91s/it]2026-10-19 22:30:18.800 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 189, smoothed-train-loss = 2.742712 | smoothed-test-loss = 2.739626 | rel_loss_delta = 3.2660233222206036e-05 | loss_stagnation_count = 1

Epoch:  63%|██████▎   | 189/300 [09:16<05:26,  2.94s/it]2026-10-19 22:30:21.824 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 190, smoothed-train-loss = 2.743831 | smoothed-test-loss = 2.740421 | rel_loss_delta = 0.0004076814922262426 | loss_stagnation_count = 0

Epoch:  63%|██████▎   | 190/300 [09:19<05:25,  2.96s/it]2026-10-19 22:30:24.703 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 191, smoothed-train-loss = 2.743134 | smoothed-test-loss = 2.739554 | rel_loss_delta = -0.0002539964531827523 | loss_stagnation_count = 0

Epoch:  64%|██████▎   | 191/300 [09:21<05:20,  2.94s/it]2026-10-19 22:30:27.595 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 192, smoothed-train-loss = 2.742772 | smoothed-test-loss = 2.721712 | rel_loss_delta = -0.00013178611442746664 | loss_stagnation_count = 0

Epoch:  64%|██████▍   | 192/300 [09:24<05:15,  2.92s/it]2026-10-19 22:30:30.470 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 193, smoothed-train-loss = 2.742259 | smoothed-test-loss = 2.725647 | rel_loss_delta = -0.00018705642706631128 | loss_stagnation_count = 0

Epoch:  64%|██████▍   | 193/300 [09:27<05:11,  2.91s/it]2026-10-19 22:30:33.337 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 194, smoothed-train-loss = 2.743064 | smoothed-test-loss = 2.739308 | rel_loss_delta = 0.000293641494556896 | loss_stagnation_count = 0

Epoch:  65%|██████▍   | 194/300 [09:30<05:07,  2.90s/it]2026-10-19 22:30:36.119 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 195, smoothed-train-loss = 2.742102 | smoothed-test-loss = 2.739458 | rel_loss_delta = -0.00035065652489208967 | loss_stagnation_count = 0

Epoch:  65%|██████▌   | 195/300 [09:33<05:00,  2.86s/it]2026-10-19 22:30:39.050 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 196, smoothed-train-loss = 2.744989 | smoothed-test-loss = 2.726516 | rel_loss_delta = 0.0010526099166748336 | loss_stagnation_count = 0

Epoch:  65%|██████▌   | 196/300 [09:36<04:59,  2.88s/it]2026-10-19 22:30:41.953 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 197, smoothed-train-loss = 2.742535 | smoothed-test-loss = 2.731199 | rel_loss_delta = -0.0008938899052462456 | loss_stagnation_count = 0

Epoch:  66%|██████▌   | 197/300 [09:39<04:57,  2.89s/it]2026-10-19 22:30:44.873 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 198, smoothed-train-loss = 2.742404 | smoothed-test-loss = 2.733389 | rel_loss_delta = -4.76782601340469e-05 | loss_stagnation_count = 1

Epoch:  66%|██████▌   | 198/300 [09:42<04:55,  2.90s/it]2026-10-19 22:30:47.648 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 199, smoothed-train-loss = 2.741039 | smoothed-test-loss = 2.728730 | rel_loss_delta = -0.0004979023907898616 | loss_stagnation_count = 0

Epoch:  66%|██████▋   | 199/300 [09:44<04:48,  2.86s/it]2026-10-19 22:30:50.517 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 200, smoothed-train-loss = 2.741450 | smoothed-test-loss = 2.727745 | rel_loss_delta = 0.00015003895540314087 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 200/300 [09:47<04:46,  2.86s/it]2026-10-19 22:30:53.546 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 201, smoothed-train-loss = 2.740220 | smoothed-test-loss = 2.732717 | rel_loss_delta = -0.00044887819644325554 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 201/300 [09:50<04:48,  2.91s/it]2026-10-19 22:30:56.569 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 202, smoothed-train-loss = 2.741176 | smoothed-test-loss = 2.737268 | rel_loss_delta = 0.0003489794230518856 | loss_stagnation_count = 0

Epoch:  67%|██████▋   | 202/300 [09:53<04:48,  2.95s/it]2026-10-19 22:30:59.473 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 203, smoothed-train-loss = 2.740605 | smoothed-test-loss = 2.726656 | rel_loss_delta = -0.0002081291477864439 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 203/300 [09:56<04:44,  2.93s/it]2026-10-19 22:31:02.406 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 204, smoothed-train-loss = 2.742227 | smoothed-test-loss = 2.728319 | rel_loss_delta = 0.000591609486183787 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 204/300 [09:59<04:41,  2.93s/it]2026-10-19 22:31:05.321 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 205, smoothed-train-loss = 2.741248 | smoothed-test-loss = 2.736168 | rel_loss_delta = -0.0003570523763362914 | loss_stagnation_count = 0

Epoch:  68%|██████▊   | 205/300 [10:02<04:38,  2.93s/it]2026-10-19 22:31:08.214 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 206, smoothed-train-loss = 2.741966 | smoothed-test-loss = 2.742669 | rel_loss_delta = 0.00026211270646973547 | loss_stagnation_count = 0

Epoch:  69%|██████▊   | 206/300 [10:05<04:34,  2.92s/it]2026-10-19 22:31:11.016 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 207, smoothed-train-loss = 2.739844 | smoothed-test-loss = 2.736084 | rel_loss_delta = -0.000773821124242334 | loss_stagnation_count = 0

Epoch:  69%|██████▉   | 207/300 [10:08<04:28,  2.88s/it]2026-10-19 22:31:13.859 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 208, smoothed-train-loss = 2.741333 | smoothed-test-loss = 2.732185 | rel_loss_delta = 0.0005432951783937869 | loss_stagnation_count = 0

Epoch:  69%|██████▉   | 208/300 [10:11<04:24,  2.87s/it]2026-10-19 22:31:16.711 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 209, smoothed-train-loss = 2.741825 | smoothed-test-loss = 2.730504 | rel_loss_delta = 0.00017936267619543116 | loss_stagnation_count = 0

Epoch:  70%|██████▉   | 209/300 [10:13<04:20,  2.87s/it]2026-10-19 22:31:19.667 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 210, smoothed-train-loss = 2.741196 | smoothed-test-loss = 2.717214 | rel_loss_delta = -0.00022915964216082258 | loss_stagnation_count = 0

Epoch:  70%|███████   | 210/300 [10:16<04:20,  2.89s/it]2026-10-19 22:31:22.490 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 211, smoothed-train-loss = 2.741302 | smoothed-test-loss = 2.727918 | rel_loss_delta = 3.847785555539058e-05 | loss_stagnation_count = 1

Epoch:  70%|███████   | 211/300 [10:19<04:15,  2.87s/it]2026-10-19 22:31:25.433 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 212, smoothed-train-loss = 2.741601 | smoothed-test-loss = 2.724589 | rel_loss_delta = 0.00010902881798431746 | loss_stagnation_count = 0

Epoch:  71%|███████   | 212/300 [10:22<04:14,  2.89s/it]2026-10-19 22:31:28.366 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 213, smoothed-train-loss = 2.741505 | smoothed-test-loss = 2.732859 | rel_loss_delta = -3.499788383227659e-05 | loss_stagnation_count = 1

Epoch:  71%|███████   | 213/300 [10:25<04:12,  2.90s/it]2026-10-19 22:31:31.333 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 214, smoothed-train-loss = 2.740036 | smoothed-test-loss = 2.744003 | rel_loss_delta = -0.0005357130163789098 | loss_stagnation_count = 0

Epoch:  71%|███████▏  | 214/300 [10:28<04:11,  2.92s/it]2026-10-19 22:31:34.220 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 215, smoothed-train-loss = 2.740175 | smoothed-test-loss = 2.745665 | rel_loss_delta = 5.0629056895386866e-05 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 215/300 [10:31<04:07,  2.91s/it]2026-10-19 22:31:37.151 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 216, smoothed-train-loss = 2.740339 | smoothed-test-loss = 2.725021 | rel_loss_delta = 6.0031988289193356e-05 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 216/300 [10:34<04:05,  2.92s/it]2026-10-19 22:31:40.107 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 217, smoothed-train-loss = 2.739474 | smoothed-test-loss = 2.741313 | rel_loss_delta = -0.000315597447856509 | loss_stagnation_count = 0

Epoch:  72%|███████▏  | 217/300 [10:37<04:03,  2.93s/it]2026-10-19 22:31:42.943 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 218, smoothed-train-loss = 2.737481 | smoothed-test-loss = 2.743807 | rel_loss_delta = -0.0007274778692797764 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 218/300 [10:40<03:57,  2.90s/it]2026-10-19 22:31:45.705 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 219, smoothed-train-loss = 2.739671 | smoothed-test-loss = 2.723266 | rel_loss_delta = 0.000799703563899426 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 219/300 [10:42<03:51,  2.86s/it]2026-10-19 22:31:48.556 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 220, smoothed-train-loss = 2.738401 | smoothed-test-loss = 2.725394 | rel_loss_delta = -0.00046345390312632737 | loss_stagnation_count = 0

Epoch:  73%|███████▎  | 220/300 [10:45<03:48,  2.86s/it]2026-10-19 22:31:51.542 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 221, smoothed-train-loss = 2.736875 | smoothed-test-loss = 2.723412 | rel_loss_delta = -0.0005571764012280223 | loss_stagnation_count = 0

Epoch:  74%|███████▎  | 221/300 [10:48<03:48,  2.90s/it]2026-10-19 22:31:54.399 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 222, smoothed-train-loss = 2.741254 | smoothed-test-loss = 2.734391 | rel_loss_delta = 0.0015997555505098222 | loss_stagnation_count = 0

Epoch:  74%|███████▍  | 222/300 [10:51<03:44,  2.88s/it]2026-10-19 22:31:57.312 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 223, smoothed-train-loss = 2.738440 | smoothed-test-loss = 2.734968 | rel_loss_delta = -0.001026323832586146 | loss_stagnation_count = 0

Epoch:  74%|███████▍  | 223/300 [10:54<03:42,  2.89s/it]2026-10-19 22:32:00.363 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 224, smoothed-train-loss = 2.737992 | smoothed-test-loss = 2.724640 | rel_loss_delta = -0.00016374427251816654 | loss_stagnation_count = 0

Epoch:  75%|███████▍  | 224/300 [10:57<03:43,  2.94s/it]2026-10-19 22:32:03.336 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 225, smoothed-train-loss = 2.739978 | smoothed-test-loss = 2.733292 | rel_loss_delta = 0.0007254575704514181 | loss_stagnation_count = 0

Epoch:  75%|███████▌  | 225/300 [11:00<03:41,  2.95s/it]2026-10-19 22:32:06.346 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 226, smoothed-train-loss = 2.739969 | smoothed-test-loss = 2.727240 | rel_loss_delta = -3.3399565012069944e-06 | loss_stagnation_count = 1

Epoch:  75%|███████▌  | 226/300 [11:03<03:39,  2.97s/it]2026-10-19 22:32:09.195 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 227, smoothed-train-loss = 2.736929 | smoothed-test-loss = 2.735031 | rel_loss_delta = -0.0011095744888155483 | loss_stagnation_count = 0

Epoch:  76%|███████▌  | 227/300 [11:06<03:34,  2.93s/it]2026-10-19 22:32:12.157 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 228, smoothed-train-loss = 2.735636 | smoothed-test-loss = 2.722327 | rel_loss_delta = -0.00047233375097477875 | loss_stagnation_count = 0

Epoch:  76%|███████▌  | 228/300 [11:09<03:31,  2.94s/it]2026-10-19 22:32:15.168 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 229, smoothed-train-loss = 2.737075 | smoothed-test-loss = 2.744677 | rel_loss_delta = 0.0005259284955260328 | loss_stagnation_count = 0

Epoch:  76%|███████▋  | 229/300 [11:12<03:30,  2.96s/it]2026-10-19 22:32:18.138 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 230, smoothed-train-loss = 2.739100 | smoothed-test-loss = 2.723143 | rel_loss_delta = 0.0007401121184623817 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 230/300 [11:15<03:27,  2.96s/it]2026-10-19 22:32:21.013 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 231, smoothed-train-loss = 2.739449 | smoothed-test-loss = 2.714624 | rel_loss_delta = 0.00012719225936269558 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 231/300 [11:18<03:22,  2.94s/it]2026-10-19 22:32:24.053 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 232, smoothed-train-loss = 2.735439 | smoothed-test-loss = 2.729345 | rel_loss_delta = -0.0014635420870367862 | loss_stagnation_count = 0

Epoch:  77%|███████▋  | 232/300 [11:21<03:21,  2.97s/it]2026-10-19 22:32:26.991 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 233, smoothed-train-loss = 2.735381 | smoothed-test-loss = 2.718407 | rel_loss_delta = -2.146783358822749e-05 | loss_stagnation_count = 1

Epoch:  78%|███████▊  | 233/300 [11:24<03:18,  2.96s/it]2026-10-19 22:32:29.991 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 234, smoothed-train-loss = 2.736530 | smoothed-test-loss = 2.720344 | rel_loss_delta = 0.0004202630091171771 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 234/300 [11:27<03:16,  2.97s/it]2026-10-19 22:32:32.882 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 235, smoothed-train-loss = 2.739412 | smoothed-test-loss = 2.721871 | rel_loss_delta = 0.0010531407658940534 | loss_stagnation_count = 0

Epoch:  78%|███████▊  | 235/300 [11:30<03:11,  2.95s/it]2026-10-19 22:32:35.918 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 236, smoothed-train-loss = 2.736164 | smoothed-test-loss = 2.725553 | rel_loss_delta = -0.001185888423847707 | loss_stagnation_count = 0

Epoch:  79%|███████▊  | 236/300 [11:33<03:10,  2.97s/it]2026-10-19 22:32:38.977 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 237, smoothed-train-loss = 2.736615 | smoothed-test-loss = 2.734635 | rel_loss_delta = 0.00016498455299802518 | loss_stagnation_count = 0

Epoch:  79%|███████▉  | 237/300 [11:36<03:08,  3.00s/it]2026-10-19 22:32:42.007 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 238, smoothed-train-loss = 2.736343 | smoothed-test-loss = 2.723148 | rel_loss_delta = -9.959515402139997e-05 | loss_stagnation_count = 0

Epoch:  79%|███████▉  | 238/300 [11:39<03:06,  3.01s/it]2026-10-19 22:32:44.868 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 239, smoothed-train-loss = 2.737274 | smoothed-test-loss = 2.712964 | rel_loss_delta = 0.00034039537274853534 | loss_stagnation_count = 0

Epoch:  80%|███████▉  | 239/300 [11:42<03:00,  2.96s/it]2026-10-19 22:32:47.813 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 240, smoothed-train-loss = 2.736062 | smoothed-test-loss = 2.723302 | rel_loss_delta = -0.00044293337289338805 | loss_stagnation_count = 0

Epoch:  80%|████████  | 240/300 [11:45<02:57,  2.96s/it]2026-10-19 22:32:50.703 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 241, smoothed-train-loss = 2.737250 | smoothed-test-loss = 2.734356 | rel_loss_delta = 0.00043433219310655955 | loss_stagnation_count = 0

Epoch:  80%|████████  | 241/300 [11:47<02:53,  2.94s/it]2026-10-19 22:32:53.548 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 242, smoothed-train-loss = 2.737099 | smoothed-test-loss = 2.732977 | rel_loss_delta = -5.528257778587853e-05 | loss_stagnation_count = 0

Epoch:  81%|████████  | 242/300 [11:50<02:48,  2.91s/it]2026-10-19 22:32:56.316 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 243, smoothed-train-loss = 2.734769 | smoothed-test-loss = 2.718262 | rel_loss_delta = -0.0008510886767281086 | loss_stagnation_count = 0

Epoch:  81%|████████  | 243/300 [11:53<02:43,  2.87s/it]2026-10-19 22:32:59.336 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 244, smoothed-train-loss = 2.733441 | smoothed-test-loss = 2.732179 | rel_loss_delta = -0.0004855589496073548 | loss_stagnation_count = 0

Epoch:  81%|████████▏ | 244/300 [11:56<02:43,  2.91s/it]2026-10-19 22:33:02.382 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 245, smoothed-train-loss = 2.735558 | smoothed-test-loss = 2.723740 | rel_loss_delta = 0.0007743273971862468 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 245/300 [11:59<02:42,  2.95s/it]2026-10-19 22:33:05.399 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 246, smoothed-train-loss = 2.735285 | smoothed-test-loss = 2.727484 | rel_loss_delta = -9.956008000648404e-05 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 246/300 [12:02<02:40,  2.97s/it]2026-10-19 22:33:08.301 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 247, smoothed-train-loss = 2.733968 | smoothed-test-loss = 2.719060 | rel_loss_delta = -0.0004817606109714691 | loss_stagnation_count = 0

Epoch:  82%|████████▏ | 247/300 [12:05<02:36,  2.95s/it]2026-10-19 22:33:11.220 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 248, smoothed-train-loss = 2.734990 | smoothed-test-loss = 2.731964 | rel_loss_delta = 0.000374044354238949 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 248/300 [12:08<02:32,  2.94s/it]2026-10-19 22:33:14.238 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 249, smoothed-train-loss = 2.736208 | smoothed-test-loss = 2.714266 | rel_loss_delta = 0.0004452350378139212 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 249/300 [12:11<02:31,  2.96s/it]2026-10-19 22:33:17.240 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 250, smoothed-train-loss = 2.735925 | smoothed-test-loss = 2.722963 | rel_loss_delta = -0.00010341455777706642 | loss_stagnation_count = 0

Epoch:  83%|████████▎ | 250/300 [12:14<02:28,  2.98s/it]2026-10-19 22:33:20.100 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 251, smoothed-train-loss = 2.736919 | smoothed-test-loss = 2.721839 | rel_loss_delta = 0.0003631709994795354 | loss_stagnation_count = 0

Epoch:  84%|████████▎ | 251/300 [12:17<02:24,  2.94s/it]2026-10-19 22:33:22.949 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 252, smoothed-train-loss = 2.734760 | smoothed-test-loss = 2.720961 | rel_loss_delta = -0.0007885795038937233 | loss_stagnation_count = 0

Epoch:  84%|████████▍ | 252/300 [12:20<02:19,  2.91s/it]2026-10-19 22:33:25.846 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 253, smoothed-train-loss = 2.735587 | smoothed-test-loss = 2.715774 | rel_loss_delta = 0.00030210035426342705 | loss_stagnation_count = 0

Epoch:  84%|████████▍ | 253/300 [12:23<02:16,  2.91s/it]2026-10-19 22:33:28.727 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 254, smoothed-train-loss = 2.732553 | smoothed-test-loss = 2.718198 | rel_loss_delta = -0.0011089885772571219 | loss_stagnation_count = 0

Epoch:  85%|████████▍ | 254/300 [12:25<02:13,  2.90s/it]2026-10-19 22:33:31.563 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 255, smoothed-train-loss = 2.733636 | smoothed-test-loss = 2.734750 | rel_loss_delta = 0.0003965012976442955 | loss_stagnation_count = 0

Epoch:  85%|████████▌ | 255/300 [12:28<02:09,  2.88s/it]2026-10-19 22:33:34.427 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 256, smoothed-train-loss = 2.735267 | smoothed-test-loss = 2.736104 | rel_loss_delta = 0.0005964100033481678 | loss_stagnation_count = 0

Epoch:  85%|████████▌ | 256/300 [12:31<02:06,  2.88s/it]2026-10-19 22:33:37.276 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 257, smoothed-train-loss = 2.733399 | smoothed-test-loss = 2.737761 | rel_loss_delta = -0.0006828036938144085 | loss_stagnation_count = 0

Epoch:  86%|████████▌ | 257/300 [12:34<02:03,  2.87s/it]2026-10-19 22:33:40.300 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 258, smoothed-train-loss = 2.735493 | smoothed-test-loss = 2.726156 | rel_loss_delta = 0.0007659715902237148 | loss_stagnation_count = 0

Epoch:  86%|████████▌ | 258/300 [12:37<02:02,  2.91s/it]2026-10-19 22:33:43.061 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 259, smoothed-train-loss = 2.734774 | smoothed-test-loss = 2.740709 | rel_loss_delta = -0.0002625725649516988 | loss_stagnation_count = 0

Epoch:  86%|████████▋ | 259/300 [12:40<01:57,  2.87s/it]2026-10-19 22:33:45.952 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 260, smoothed-train-loss = 2.734443 | smoothed-test-loss = 2.714661 | rel_loss_delta = -0.0001211065381734811 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 260/300 [12:43<01:55,  2.88s/it]2026-10-19 22:33:48.810 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 261, smoothed-train-loss = 2.733469 | smoothed-test-loss = 2.722328 | rel_loss_delta = -0.00035637865213410976 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 261/300 [12:46<01:51,  2.87s/it]2026-10-19 22:33:51.797 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 262, smoothed-train-loss = 2.732980 | smoothed-test-loss = 2.725927 | rel_loss_delta = -0.00017888605687442648 | loss_stagnation_count = 0

Epoch:  87%|████████▋ | 262/300 [12:49<01:50,  2.91s/it]2026-10-19 22:33:54.629 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 263, smoothed-train-loss = 2.734272 | smoothed-test-loss = 2.749476 | rel_loss_delta = 0.00047292813655640915 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 263/300 [12:51<01:46,  2.88s/it]2026-10-19 22:33:57.549 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 264, smoothed-train-loss = 2.733341 | smoothed-test-loss = 2.717432 | rel_loss_delta = -0.0003406417529294681 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 264/300 [12:54<01:44,  2.89s/it]2026-10-19 22:34:00.423 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 265, smoothed-train-loss = 2.735051 | smoothed-test-loss = 2.713792 | rel_loss_delta = 0.000625762983114476 | loss_stagnation_count = 0

Epoch:  88%|████████▊ | 265/300 [12:57<01:41,  2.89s/it]2026-10-19 22:34:03.359 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 266, smoothed-train-loss = 2.734221 | smoothed-test-loss = 2.724667 | rel_loss_delta = -0.00030345821516827224 | loss_stagnation_count = 0

Epoch:  89%|████████▊ | 266/300 [13:00<01:38,  2.90s/it]2026-10-19 22:34:06.203 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 267, smoothed-train-loss = 2.734648 | smoothed-test-loss = 2.723416 | rel_loss_delta = 0.00015594373979310418 | loss_stagnation_count = 0

Epoch:  89%|████████▉ | 267/300 [13:03<01:35,  2.89s/it]2026-10-19 22:34:09.124 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 268, smoothed-train-loss = 2.733856 | smoothed-test-loss = 2.721938 | rel_loss_delta = -0.00028955581896480194 | loss_stagnation_count = 0

Epoch:  89%|████████▉ | 268/300 [13:06<01:32,  2.90s/it]2026-10-19 22:34:12.123 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 269, smoothed-train-loss = 2.734105 | smoothed-test-loss = 2.725649 | rel_loss_delta = 9.115162163450298e-05 | loss_stagnation_count = 0

Epoch:  90%|████████▉ | 269/300 [13:09<01:30,  2.93s/it]2026-10-19 22:34:14.983 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 270, smoothed-train-loss = 2.733454 | smoothed-test-loss = 2.723818 | rel_loss_delta = -0.00023818562722516633 | loss_stagnation_count = 0

Epoch:  90%|█████████ | 270/300 [13:12<01:27,  2.91s/it]2026-10-19 22:34:17.751 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 271, smoothed-train-loss = 2.732838 | smoothed-test-loss = 2.718412 | rel_loss_delta = -0.00022536890568473482 | loss_stagnation_count = 0

Epoch:  90%|█████████ | 271/300 [13:15<01:23,  2.86s/it]2026-10-19 22:34:20.731 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 272, smoothed-train-loss = 2.735085 | smoothed-test-loss = 2.724056 | rel_loss_delta = 0.000822227565987772 | loss_stagnation_count = 0

Epoch:  91%|█████████ | 272/300 [13:17<01:21,  2.90s/it]2026-10-19 22:34:23.700 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 273, smoothed-train-loss = 2.733381 | smoothed-test-loss = 2.717871 | rel_loss_delta = -0.000622919511958675 | loss_stagnation_count = 0

Epoch:  91%|█████████ | 273/300 [13:20<01:18,  2.92s/it]2026-10-19 22:34:26.712 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 274, smoothed-train-loss = 2.732948 | smoothed-test-loss = 2.715447 | rel_loss_delta = -0.0001585052857366165 | loss_stagnation_count = 0

Epoch:  91%|█████████▏| 274/300 [13:23<01:16,  2.95s/it]2026-10-19 22:34:29.522 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 275, smoothed-train-loss = 2.733397 | smoothed-test-loss = 2.721488 | rel_loss_delta = 0.00016420049962750282 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 275/300 [13:26<01:12,  2.91s/it]2026-10-19 22:34:32.498 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 276, smoothed-train-loss = 2.733317 | smoothed-test-loss = 2.723827 | rel_loss_delta = -2.8930693105364686e-05 | loss_stagnation_count = 1

Epoch:  92%|█████████▏| 276/300 [13:29<01:10,  2.93s/it]2026-10-19 22:34:35.449 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 277, smoothed-train-loss = 2.732848 | smoothed-test-loss = 2.723477 | rel_loss_delta = -0.0001717239799557343 | loss_stagnation_count = 0

Epoch:  92%|█████████▏| 277/300 [13:32<01:07,  2.93s/it]2026-10-19 22:34:38.378 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 278, smoothed-train-loss = 2.732650 | smoothed-test-loss = 2.723122 | rel_loss_delta = -7.26136251777482e-05 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 278/300 [13:35<01:04,  2.93s/it]2026-10-19 22:34:41.364 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 279, smoothed-train-loss = 2.733351 | smoothed-test-loss = 2.746067 | rel_loss_delta = 0.000256510064301713 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 279/300 [13:38<01:01,  2.95s/it]2026-10-19 22:34:44.363 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 280, smoothed-train-loss = 2.734098 | smoothed-test-loss = 2.717510 | rel_loss_delta = 0.00027335901034063024 | loss_stagnation_count = 0

Epoch:  93%|█████████▎| 280/300 [13:41<00:59,  2.96s/it]2026-10-19 22:34:47.315 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 281, smoothed-train-loss = 2.732412 | smoothed-test-loss = 2.739439 | rel_loss_delta = -0.0006165996103451725 | loss_stagnation_count = 0

Epoch:  94%|█████████▎| 281/300 [13:44<00:56,  2.96s/it]2026-10-19 22:34:50.275 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 282, smoothed-train-loss = 2.732455 | smoothed-test-loss = 2.725305 | rel_loss_delta = 1.594746309689323e-05 | loss_stagnation_count = 1

Epoch:  94%|█████████▍| 282/300 [13:47<00:53,  2.96s/it]2026-10-19 22:34:53.157 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 283, smoothed-train-loss = 2.733705 | smoothed-test-loss = 2.733399 | rel_loss_delta = 0.0004573258705633043 | loss_stagnation_count = 0

Epoch:  94%|█████████▍| 283/300 [13:50<00:49,  2.94s/it]2026-10-19 22:34:56.224 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 284, smoothed-train-loss = 2.732647 | smoothed-test-loss = 2.723247 | rel_loss_delta = -0.00038690874106063206 | loss_stagnation_count = 0

Epoch:  95%|█████████▍| 284/300 [13:53<00:47,  2.98s/it]2026-10-19 22:34:59.161 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 285, smoothed-train-loss = 2.731101 | smoothed-test-loss = 2.718392 | rel_loss_delta = -0.000565933607021479 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 285/300 [13:56<00:44,  2.96s/it]2026-10-19 22:35:02.181 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 286, smoothed-train-loss = 2.732187 | smoothed-test-loss = 2.730880 | rel_loss_delta = 0.00039769205859618926 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 286/300 [13:59<00:41,  2.98s/it]2026-10-19 22:35:05.100 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 287, smoothed-train-loss = 2.731899 | smoothed-test-loss = 2.717636 | rel_loss_delta = -0.0001052720292287172 | loss_stagnation_count = 0

Epoch:  96%|█████████▌| 287/300 [14:02<00:38,  2.96s/it]2026-10-19 22:35:08.100 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 288, smoothed-train-loss = 2.732474 | smoothed-test-loss = 2.725595 | rel_loss_delta = 0.00021048886073033357 | loss_stagnation_count = 0

Epoch:  96%|█████████▌| 288/300 [14:05<00:35,  2.97s/it]2026-10-19 22:35:11.131 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 289, smoothed-train-loss = 2.731100 | smoothed-test-loss = 2.719038 | rel_loss_delta = -0.000503071521246859 | loss_stagnation_count = 0

Epoch:  96%|█████████▋| 289/300 [14:08<00:32,  2.99s/it]2026-10-19 22:35:14.059 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 290, smoothed-train-loss = 2.731471 | smoothed-test-loss = 2.719855 | rel_loss_delta = 0.00013582560036936843 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 290/300 [14:11<00:29,  2.97s/it]2026-10-19 22:35:16.841 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 291, smoothed-train-loss = 2.731547 | smoothed-test-loss = 2.717526 | rel_loss_delta = 2.7827926217146067e-05 | loss_stagnation_count = 1

Epoch:  97%|█████████▋| 291/300 [14:14<00:26,  2.92s/it]2026-10-19 22:35:19.853 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 292, smoothed-train-loss = 2.731860 | smoothed-test-loss = 2.712069 | rel_loss_delta = 0.00011471115428144751 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 292/300 [14:17<00:23,  2.94s/it]2026-10-19 22:35:22.865 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 293, smoothed-train-loss = 2.732328 | smoothed-test-loss = 2.723748 | rel_loss_delta = 0.00017142943049394292 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 293/300 [14:20<00:20,  2.96s/it]2026-10-19 22:35:25.945 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 294, smoothed-train-loss = 2.734340 | smoothed-test-loss = 2.720875 | rel_loss_delta = 0.0007363395378009494 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 294/300 [14:23<00:17,  3.00s/it]2026-10-19 22:35:28.880 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 295, smoothed-train-loss = 2.731509 | smoothed-test-loss = 2.712636 | rel_loss_delta = -0.0010355846943055451 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 295/300 [14:26<00:14,  2.98s/it]2026-10-19 22:35:31.927 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 296, smoothed-train-loss = 2.730366 | smoothed-test-loss = 2.718519 | rel_loss_delta = -0.0004183670496089206 | loss_stagnation_count = 0

Epoch:  99%|█████████▊| 296/300 [14:29<00:11,  3.00s/it]2026-10-19 22:35:34.924 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 297, smoothed-train-loss = 2.732831 | smoothed-test-loss = 2.726657 | rel_loss_delta = 0.0009028842228195373 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 297/300 [14:32<00:08,  3.00s/it]2026-10-19 22:35:37.890 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 298, smoothed-train-loss = 2.731367 | smoothed-test-loss = 2.711358 | rel_loss_delta = -0.0005358039540366482 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 298/300 [14:35<00:05,  2.99s/it]2026-10-19 22:35:40.887 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 299, smoothed-train-loss = 2.733595 | smoothed-test-loss = 2.713623 | rel_loss_delta = 0.0008156609420404588 | loss_stagnation_count = 0

Epoch: 100%|█████████▉| 299/300 [14:38<00:02,  2.99s/it]2026-10-19 22:35:43.743 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 300, smoothed-train-loss = 2.731427 | smoothed-test-loss = 2.724937 | rel_loss_delta = -0.0007928787712099878 | loss_stagnation_count = 0

Epoch: 100%|██████████| 300/300 [14:41<00:00,  2.94s/it]
2026-10-19 22:35:43.747 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:2.726193904876709
2026-10-19 22:35:43.749 | INFO     | __main__:<module>:297 - Given num_reflow_iterations = 0 and current reflow_iteration = 0, no further reflows are needed, exiting the reflow loop
reflow iterations:   0%|          | 0/1 [14:41<?, ?it/s]
2026-10-19 22:35:43.753 | INFO     | __main__:<module>:355 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-19T22:21:00.926949.pkl
2026-10-19 22:35:43.754 | INFO     | __main__:<module>:356 - vt training process finished successfully.

Process finished with exit code 0

```

# Inference
