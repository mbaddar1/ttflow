
# Desc
Try to fit the Gaussian Mixture 2D with N-Comp=16 with RegNet 
as the approximating model for $v_t$. Trying different architecture `hidden_dim` and `depth`
# Sub-Experiments
The section documents sub-experiments

## Sub-Experiment 8.1
Directory `experiments_and_results/8/8_1`
### Trainer Run
**Trainer args**
```log
vt_trainer.py --data-snapshot-filepath data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_data_dim_2_n_train_test_eval_500000_200000_200000.pkl --vt-model nn --vt-opt adam --trajectory-model global --nn-hidden-dim 256 --nn-depth 2 --reflow-iterations 0 --nn-epochs 300 --reflow-iterations 0 --ode-solver-method dopri8 --prior gaussian
```
**Arch and opts**
```log
Network is on: cuda:0
2026-10-20 13:49:14.284 | INFO     | __main__:<module>:256 - ***********************************************************
Training Model Specs:
  [Model Architecture]
    - model =                   VelocityFieldRegressionNet(
  (model): Sequential(
    (0): Linear(in_features=3, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=2, bias=True)
  )
)
    - Input Dim:                3
    - Hidden Dim:               256
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
```
**Snippet from training log**
```log
...
Epoch:  96%|█████████▋| 289/300 [15:06<00:33,  3.02s/it]2026-10-20 14:04:24.578 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 290, smoothed-train-loss = 9.495483 | smoothed-test-loss = 9.631007 | rel_loss_delta = -0.00011459437787736638 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 290/300 [15:09<00:30,  3.03s/it]2026-10-20 14:04:27.461 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 291, smoothed-train-loss = 9.493696 | smoothed-test-loss = 9.602713 | rel_loss_delta = -0.0001881550853107278 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 291/300 [15:12<00:26,  2.98s/it]2026-10-20 14:04:30.509 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 292, smoothed-train-loss = 9.494677 | smoothed-test-loss = 9.604125 | rel_loss_delta = 0.00010327490610427965 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 292/300 [15:15<00:24,  3.00s/it]2026-10-20 14:04:33.575 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 293, smoothed-train-loss = 9.493868 | smoothed-test-loss = 9.633326 | rel_loss_delta = -8.519096366570399e-05 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 293/300 [15:18<00:21,  3.02s/it]2026-10-20 14:04:36.525 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 294, smoothed-train-loss = 9.496062 | smoothed-test-loss = 9.603591 | rel_loss_delta = 0.00023106643352113177 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 294/300 [15:21<00:17,  3.00s/it]2026-10-20 14:04:39.365 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 295, smoothed-train-loss = 9.492917 | smoothed-test-loss = 9.590979 | rel_loss_delta = -0.0003311064796932387 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 295/300 [15:24<00:14,  2.95s/it]2026-10-20 14:04:42.297 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 296, smoothed-train-loss = 9.492042 | smoothed-test-loss = 9.605785 | rel_loss_delta = -9.216178562774068e-05 | loss_stagnation_count = 0

Epoch:  99%|█████████▊| 296/300 [15:27<00:11,  2.95s/it]2026-10-20 14:04:45.252 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 297, smoothed-train-loss = 9.495410 | smoothed-test-loss = 9.590878 | rel_loss_delta = 0.00035477988654758074 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 297/300 [15:30<00:08,  2.95s/it]2026-10-20 14:04:48.187 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 298, smoothed-train-loss = 9.494583 | smoothed-test-loss = 9.592063 | rel_loss_delta = -8.71435237647159e-05 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 298/300 [15:33<00:05,  2.94s/it]2026-10-20 14:04:51.018 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 299, smoothed-train-loss = 9.492370 | smoothed-test-loss = 9.604372 | rel_loss_delta = -0.00023302309368039223 | loss_stagnation_count = 0

Epoch: 100%|█████████▉| 299/300 [15:35<00:02,  2.91s/it]2026-10-20 14:04:54.010 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 300, smoothed-train-loss = 9.494943 | smoothed-test-loss = 9.619873 | rel_loss_delta = 0.0002710340460313968 | loss_stagnation_count = 0

Epoch: 100%|██████████| 300/300 [15:38<00:00,  3.13s/it]
2026-10-20 14:04:54.021 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:9.621594429016113
2026-10-20 14:04:54.022 | INFO     | __main__:<module>:297 - Given num_reflow_iterations = 0 and current reflow_iteration = 0, no further reflows are needed, exiting the reflow loop
reflow iterations:   0%|          | 0/1 [15:39<?, ?it/s]
2026-10-20 14:04:54.034 | INFO     | __main__:<module>:355 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_256_depth_2_opt_adam_nn_hidden_dim_256_timestamp_2026-10-20T13:49:13.332714.pkl
2026-10-20 14:04:54.034 | INFO     | __main__:<module>:356 - vt training process finished successfully.
```
**Inference**
* model 
```log
loaded vt artifact(dict) : 
 {'models': [VelocityFieldRegressionNet(
  (model): Sequential(
    (0): Linear(in_features=3, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=2, bias=True)
  )
)], 'reflows': 0, 'prior': 'gaussian', 'hyperparams': {'hidden_dim': 256, 'depth': 2, 'lr': 0.001, 'batch_size': 2048, 'epochs': 300}, 'data_snapshot_filepath': 'data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_data_dim_2_n_train_test_eval_500000_200000_200000.pkl', 'trajectory_model': 'global', 'learning_curves': [], 'arch': 'nn', 'opt': 'adam'} 
```
- results 
```log
2026-10-20 14:24:38.894 | INFO     | __main__:<module>:320 - 
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : dopri8
  VT Trajectory Model  : global
  Num Reflows          : 0
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 138.56s
  Evaluations (t)      : 1000
  Eval Results         : {0.005461769923567773: {'real_generated': {'mean': 0.2445937365293503, 'stddev': 0.07203739081752313}, 'real_real': {'mean': 0.07854330912232399, 'stddev': 0.03811017679982903}, 'generated_generated': {'mean': 0.050732604786753656, 'stddev': 0.01733853152749989}}, 0.01: {'real_generated': {'mean': 0.24796492755413055, 'stddev': 0.07068886616180606}, 'real_real': {'mean': 0.07765441611409188, 'stddev': 0.02799670859245522}, 'generated_generated': {'mean': 0.06046536937355995, 'stddev': 0.011316321911524747}}, 0.01365442480891943: {'real_generated': {'mean': 0.24397651851177216, 'stddev': 0.08372910031408697}, 'real_real': {'mean': 0.08811530396342278, 'stddev': 0.02836928169256609}, 'generated_generated': {'mean': 0.042622994631528854, 'stddev': 0.008346132452624534}}, 0.02: {'real_generated': {'mean': 0.26321679055690766, 'stddev': 0.06406760736185543}, 'real_real': {'mean': 0.04955028556287289, 'stddev': 0.016318299270156027}, 'generated_generated': {'mean': 0.060901975631713866, 'stddev': 0.0274333395071299}}, 0.02730884961783886: {'real_generated': {'mean': 0.31855522394180297, 'stddev': 0.0895801176096937}, 'real_real': {'mean': 0.09381783679127693, 'stddev': 0.06341205665928307}, 'generated_generated': {'mean': 0.07326858080923557, 'stddev': 0.03297159188521162}}, 0.05: {'real_generated': {'mean': 0.303778076171875, 'stddev': 0.06041010172324353}, 'real_real': {'mean': 0.081813133507967, 'stddev': 0.04039382949472252}, 'generated_generated': {'mean': 0.07775079011917115, 'stddev': 0.03332285065119099}}, 0.05461769923567772: {'real_generated': {'mean': 0.2650935649871826, 'stddev': 0.07307135280968295}, 'real_real': {'mean': 0.09450588524341583, 'stddev': 0.020143383646283073}, 'generated_generated': {'mean': 0.0484047994017601, 'stddev': 0.020689273853978347}}, 0.1: {'real_generated': {'mean': 0.335673651099205, 'stddev': 0.09289317840926392}, 'real_real': {'mean': 0.08674676716327667, 'stddev': 0.05266821962709289}, 'generated_generated': {'mean': 0.05285775363445282, 'stddev': 0.038146265368925014}}, 0.10923539847135544: {'real_generated': {'mean': 0.246291384100914, 'stddev': 0.09180427307366061}, 'real_real': {'mean': 0.07282101586461068, 'stddev': 0.01792426517530834}, 'generated_generated': {'mean': 0.06785427778959274, 'stddev': 0.03903840245481056}}, 0.2: {'real_generated': {'mean': 0.21565671861171723, 'stddev': 0.08172960302938208}, 'real_real': {'mean': 0.06768339425325394, 'stddev': 0.027486900734515724}, 'generated_generated': {'mean': 0.058635410666465757, 'stddev': 0.04377966535936047}}, 0.5: {'real_generated': {'mean': 0.23236789405345917, 'stddev': 0.08147426211403859}, 'real_real': {'mean': 0.083970782533288, 'stddev': 0.05486221917095659}, 'generated_generated': {'mean': 0.047258677706122396, 'stddev': 0.013884460839681511}}, 1.0: {'real_generated': {'mean': 0.1795678436756134, 'stddev': 0.04473598388291193}, 'real_real': {'mean': 0.06248415969312191, 'stddev': 0.03597147994584856}, 'generated_generated': {'mean': 0.04215064086019993, 'stddev': 0.01671635122672313}}, 2.0: {'real_generated': {'mean': 0.18613759875297547, 'stddev': 0.041669184202018}, 'real_real': {'mean': 0.03056037463247776, 'stddev': 0.012419934910286652}, 'generated_generated': {'mean': 0.025860287249088287, 'stddev': 0.022648088077928084}}, 5.0: {'real_generated': {'mean': 0.14689309597015382, 'stddev': 0.04759342479119805}, 'real_real': {'mean': 0.024758874624967574, 'stddev': 0.0019868701734331437}, 'generated_generated': {'mean': 0.024724316783249378, 'stddev': 0.019676146112604292}}, 10.0: {'real_generated': {'mean': 0.14719095826148987, 'stddev': 0.042614139163371055}, 'real_real': {'mean': 0.011752016004174947, 'stddev': 0.01859885531647872}, 'generated_generated': {'mean': 0.014542609825730324, 'stddev': 0.01032036798544929}}}
================================================== 
```
* Figure 
```log
experiments_and_results/8/8_1/inference_gaussian_mixture_2d_data_params_num_components_16_timestamp_2026-10-20T14:15:30.748471_trajectory_model_global_ode_solver_method_dopri8_num_reflows_0.png
```
___
## Sub-Experiment 2
* Model arch and opt
```log
2026-10-20 15:13:06.981 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-20 15:13:07.657 | INFO     | __main__:<module>:256 - ***********************************************************
Training Model Specs:
  [Model Architecture]
    - model =                   VelocityFieldRegressionNet(
  (model): Sequential(
    (0): Linear(in_features=3, out_features=1024, bias=True)
    (1): ReLU()
    (2): Linear(in_features=1024, out_features=1024, bias=True)
    (3): ReLU()
    (4): Linear(in_features=1024, out_features=2, bias=True)
  )
)
    - Input Dim:                3
    - Hidden Dim:               1024
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
reflow iterations:   0%|          | 0/1 [00:00<?, ?it/s]2026-10-20 15:13:07.657 | INFO     | __main__:<module>:281 - Reflow iteration: 0 
```
* snippet from training log 
```log
 ...
 Epoch:  96%|█████████▋| 289/300 [17:40<00:43,  3.91s/it]2026-10-20 15:30:53.137 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 290, smoothed-train-loss = 9.507231 | smoothed-test-loss = 9.585744 | rel_loss_delta = -9.226297179882204e-05 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 290/300 [17:44<00:38,  3.85s/it]2026-10-20 15:30:56.684 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 291, smoothed-train-loss = 9.506644 | smoothed-test-loss = 9.567075 | rel_loss_delta = -6.172112337918149e-05 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 291/300 [17:47<00:33,  3.76s/it]2026-10-20 15:31:00.568 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 292, smoothed-train-loss = 9.503171 | smoothed-test-loss = 9.559169 | rel_loss_delta = -0.00036541233197442805 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 292/300 [17:51<00:30,  3.80s/it]2026-10-20 15:31:04.879 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 293, smoothed-train-loss = 9.505963 | smoothed-test-loss = 9.564302 | rel_loss_delta = 0.0002938611000920845 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 293/300 [17:56<00:27,  3.95s/it]2026-10-20 15:31:08.552 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 294, smoothed-train-loss = 9.507717 | smoothed-test-loss = 9.548748 | rel_loss_delta = 0.00018444324548208568 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 294/300 [17:59<00:23,  3.87s/it]2026-10-20 15:31:12.385 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 295, smoothed-train-loss = 9.504060 | smoothed-test-loss = 9.550779 | rel_loss_delta = -0.0003845335925397892 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 295/300 [18:03<00:19,  3.86s/it]2026-10-20 15:31:16.044 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 296, smoothed-train-loss = 9.506069 | smoothed-test-loss = 9.559342 | rel_loss_delta = 0.00021130902407507862 | loss_stagnation_count = 0

Epoch:  99%|█████████▊| 296/300 [18:07<00:15,  3.80s/it]2026-10-20 15:31:19.743 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 297, smoothed-train-loss = 9.505665 | smoothed-test-loss = 9.611108 | rel_loss_delta = -4.249337792404718e-05 | loss_stagnation_count = 1

Epoch:  99%|█████████▉| 297/300 [18:11<00:11,  3.77s/it]2026-10-20 15:31:23.462 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 298, smoothed-train-loss = 9.508546 | smoothed-test-loss = 9.564242 | rel_loss_delta = 0.000303069787706454 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 298/300 [18:14<00:07,  3.75s/it]2026-10-20 15:31:27.091 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 299, smoothed-train-loss = 9.505062 | smoothed-test-loss = 9.572867 | rel_loss_delta = -0.00036642468181887367 | loss_stagnation_count = 0

Epoch: 100%|█████████▉| 299/300 [18:18<00:03,  3.72s/it]2026-10-20 15:31:30.580 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 300, smoothed-train-loss = 9.505961 | smoothed-test-loss = 9.554682 | rel_loss_delta = 9.460910091359073e-05 | loss_stagnation_count = 0

Epoch: 100%|██████████| 300/300 [18:21<00:00,  3.67s/it]
2026-10-20 15:31:30.689 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:9.552661895751953
2026-10-20 15:31:30.691 | INFO     | __main__:<module>:297 - Given num_reflow_iterations = 0 and current reflow_iteration = 0, no further reflows are needed, exiting the reflow loop
reflow iterations:   0%|          | 0/1 [18:23<?, ?it/s]
2026-10-20 15:31:30.823 | INFO     | __main__:<module>:355 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_1024_depth_2_opt_adam_nn_hidden_dim_1024_timestamp_2026-10-20T15:13:06.677069.pkl
2026-10-20 15:31:30.823 | INFO     | __main__:<module>:356 - vt training process finished successfully.
```
**Inference**

log
```log
2026-10-20 16:02:53.748 | INFO     | __main__:<module>:320 - 
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : dopri8
  VT Trajectory Model  : global
  Num Reflows          : 0
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 1208.49s
  Evaluations (t)      : 1000
  Eval Results         : {0.005461769923567773: {'real_generated': {'mean': 0.16319307386875154, 'stddev': 0.026879910552888343}, 'real_real': {'mean': 0.07854330912232399, 'stddev': 0.03811017679982903}, 'generated_generated': {'mean': 0.04962067566812038, 'stddev': 0.019702230864183545}}, 0.01: {'real_generated': {'mean': 0.18811777979135513, 'stddev': 0.04474053378781849}, 'real_real': {'mean': 0.07765441611409188, 'stddev': 0.02799670859245522}, 'generated_generated': {'mean': 0.07123555541038513, 'stddev': 0.05159011086892578}}, 0.01365442480891943: {'real_generated': {'mean': 0.16601527482271194, 'stddev': 0.04753920655403241}, 'real_real': {'mean': 0.08811530396342278, 'stddev': 0.02836928169256609}, 'generated_generated': {'mean': 0.07020258009433747, 'stddev': 0.008565348256082527}}, 0.02: {'real_generated': {'mean': 0.1612454444169998, 'stddev': 0.032271848626502446}, 'real_real': {'mean': 0.04955028556287289, 'stddev': 0.016318299270156027}, 'generated_generated': {'mean': 0.06968587003648281, 'stddev': 0.03032217654563785}}, 0.02730884961783886: {'real_generated': {'mean': 0.19841629713773729, 'stddev': 0.11777533205172906}, 'real_real': {'mean': 0.09381783679127693, 'stddev': 0.06341205665928307}, 'generated_generated': {'mean': 0.04361335709691048, 'stddev': 0.014232743492366234}}, 0.05: {'real_generated': {'mean': 0.21628547310829163, 'stddev': 0.0623938757463708}, 'real_real': {'mean': 0.081813133507967, 'stddev': 0.04039382949472252}, 'generated_generated': {'mean': 0.0682249404489994, 'stddev': 0.03814168464839377}}, 0.05461769923567772: {'real_generated': {'mean': 0.15319233238697053, 'stddev': 0.0310185736483956}, 'real_real': {'mean': 0.09450588524341583, 'stddev': 0.020143383646283073}, 'generated_generated': {'mean': 0.05570565462112427, 'stddev': 0.020853677834492934}}, 0.1: {'real_generated': {'mean': 0.13876013159751893, 'stddev': 0.03354626190926605}, 'real_real': {'mean': 0.08674676716327667, 'stddev': 0.05266821962709289}, 'generated_generated': {'mean': 0.07351244017481803, 'stddev': 0.02068449730964201}}, 0.10923539847135544: {'real_generated': {'mean': 0.15488951206207274, 'stddev': 0.02970402442795164}, 'real_real': {'mean': 0.07282101586461068, 'stddev': 0.01792426517530834}, 'generated_generated': {'mean': 0.04620735794305801, 'stddev': 0.019510556958988155}}, 0.2: {'real_generated': {'mean': 0.15982001572847365, 'stddev': 0.06332241181424787}, 'real_real': {'mean': 0.06768339425325394, 'stddev': 0.027486900734515724}, 'generated_generated': {'mean': 0.04031851477921009, 'stddev': 0.014341904825186302}}, 0.5: {'real_generated': {'mean': 0.22527142167091369, 'stddev': 0.10185721589361166}, 'real_real': {'mean': 0.083970782533288, 'stddev': 0.05486221917095659}, 'generated_generated': {'mean': 0.07082113549113274, 'stddev': 0.04200306364709024}}, 1.0: {'real_generated': {'mean': 0.1766783267259598, 'stddev': 0.05160046424623615}, 'real_real': {'mean': 0.06248415969312191, 'stddev': 0.03597147994584856}, 'generated_generated': {'mean': 0.05155206397175789, 'stddev': 0.016710088203409686}}, 2.0: {'real_generated': {'mean': 0.05566047206521034, 'stddev': 0.03372735004941087}, 'real_real': {'mean': 0.03056037463247776, 'stddev': 0.012419934910286652}, 'generated_generated': {'mean': 0.04650417268276215, 'stddev': 0.014974748483924225}}, 5.0: {'real_generated': {'mean': 0.08979548215866089, 'stddev': 0.039518468550505684}, 'real_real': {'mean': 0.024758874624967574, 'stddev': 0.0019868701734331437}, 'generated_generated': {'mean': 0.027722546830773354, 'stddev': 0.016381553258461308}}, 10.0: {'real_generated': {'mean': 0.031113225594162942, 'stddev': 0.02301352211335225}, 'real_real': {'mean': 0.011752016004174947, 'stddev': 0.01859885531647872}, 'generated_generated': {'mean': 0.0127702122554183, 'stddev': 0.00845124400186906}}}
==================================================
/home/mbaddar/Documents/mbaddar/ttflow/ttflow/viz/viz.py:232: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
2026-10-20 16:02:55.168 | INFO     | __main__:<module>:362 - Inference and evaluation process finished successfully!
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_16_timestamp_2026-10-20T15:35:58.336409_trajectory_model_global_ode_solver_method_dopri8_num_reflows_0.png
Solving ODE | Method = dopri8: 100%|██████████| 1.002745270729065/1.0 [26:56<-00:04, 1612.29s/it]
```
- figure 
```path
experiments_and_results/8/8_2/inference_gaussian_mixture_2d_data_params_num_components_16_timestamp_2026-10-20T15:35:58.336409_trajectory_model_global_ode_solver_method_dopri8_num_reflows_0.png
```
# Observations and Takeaways 
- Uniform prior is bad visually - Gaussian is the way to go
support
- wider shallower networks are better : hidden_dim = 1024 and depth = 2 are the best for visuals
