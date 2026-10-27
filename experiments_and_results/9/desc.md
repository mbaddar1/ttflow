# Desc
* Dataset : GM2D Nc = 16 with radius = 0.5 and var = 0.0001
* Reason : explore if this x1 would lead to curved paths from $\pi_0$ = N(0,I)

## Subexperiment 9.1
* dataset GM 2D Nc=16, radius = 0.5, var = 0.0001
### Model-Training
**log**


(arch, opt, hyper-params and loss curve)
```log
2026-10-20 16:11:47.650 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-20 16:11:48.231 | INFO     | __main__:<module>:256 - ***********************************************************
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
reflow iterations:   0%|          | 0/1 [00:00<?, ?it/s]2026-10-20 16:11:48.232 | INFO     | __main__:<module>:281 - Reflow iteration: 0
2026-10-20 16:11:49.114 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:57 - @epoch = 0 (before training), loss_fn = MSELoss(), loss_value = 1.123082
2026-10-20 16:11:49.229 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:64 - MSELoss(test) before training:1.1230816841125488
```

* Snippet of the training log 
```log
 
Epoch:  94%|█████████▎| 281/300 [17:06<01:06,  3.48s/it]2026-10-20 16:28:58.860 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 282, smoothed-train-loss = 0.487589 | smoothed-test-loss = 0.489043 | rel_loss_delta = -0.0003989081639785558 | loss_stagnation_count = 0

Epoch:  94%|█████████▍| 282/300 [17:09<01:02,  3.49s/it]2026-10-20 16:29:02.267 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 283, smoothed-train-loss = 0.487642 | smoothed-test-loss = 0.491293 | rel_loss_delta = 0.00010921529909471864 | loss_stagnation_count = 0

Epoch:  94%|█████████▍| 283/300 [17:13<00:58,  3.46s/it]2026-10-20 16:29:05.818 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 284, smoothed-train-loss = 0.487715 | smoothed-test-loss = 0.489339 | rel_loss_delta = 0.00015087708283399492 | loss_stagnation_count = 0

Epoch:  95%|█████████▍| 284/300 [17:16<00:55,  3.49s/it]2026-10-20 16:29:09.388 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 285, smoothed-train-loss = 0.487536 | smoothed-test-loss = 0.488776 | rel_loss_delta = -0.00036851509120559037 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 285/300 [17:20<00:52,  3.51s/it]2026-10-20 16:29:12.965 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 286, smoothed-train-loss = 0.487568 | smoothed-test-loss = 0.490134 | rel_loss_delta = 6.568724661882058e-05 | loss_stagnation_count = 0

Epoch:  95%|█████████▌| 286/300 [17:23<00:49,  3.53s/it]2026-10-20 16:29:16.699 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 287, smoothed-train-loss = 0.487712 | smoothed-test-loss = 0.488998 | rel_loss_delta = 0.00029550616801745706 | loss_stagnation_count = 0

Epoch:  96%|█████████▌| 287/300 [17:27<00:46,  3.59s/it]2026-10-20 16:29:20.262 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 288, smoothed-train-loss = 0.487588 | smoothed-test-loss = 0.490548 | rel_loss_delta = -0.00025350629363508345 | loss_stagnation_count = 0

Epoch:  96%|█████████▌| 288/300 [17:31<00:43,  3.58s/it]2026-10-20 16:29:23.815 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 289, smoothed-train-loss = 0.487726 | smoothed-test-loss = 0.489218 | rel_loss_delta = 0.0002834027823915917 | loss_stagnation_count = 0

Epoch:  96%|█████████▋| 289/300 [17:34<00:39,  3.57s/it]2026-10-20 16:29:27.506 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 290, smoothed-train-loss = 0.487722 | smoothed-test-loss = 0.489216 | rel_loss_delta = -1.0032002418029852e-05 | loss_stagnation_count = 1

Epoch:  97%|█████████▋| 290/300 [17:38<00:36,  3.61s/it]2026-10-20 16:29:31.033 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 291, smoothed-train-loss = 0.487619 | smoothed-test-loss = 0.489775 | rel_loss_delta = -0.00021119926451005022 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 291/300 [17:41<00:32,  3.58s/it]2026-10-20 16:29:35.430 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 292, smoothed-train-loss = 0.487591 | smoothed-test-loss = 0.489214 | rel_loss_delta = -5.690515395456409e-05 | loss_stagnation_count = 0

Epoch:  97%|█████████▋| 292/300 [17:46<00:30,  3.83s/it]2026-10-20 16:29:39.180 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 293, smoothed-train-loss = 0.487626 | smoothed-test-loss = 0.488921 | rel_loss_delta = 7.214842809810121e-05 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 293/300 [17:49<00:26,  3.80s/it]2026-10-20 16:29:43.025 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 294, smoothed-train-loss = 0.487462 | smoothed-test-loss = 0.488922 | rel_loss_delta = -0.00033593879207343216 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 294/300 [17:53<00:22,  3.82s/it]2026-10-20 16:29:46.795 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 295, smoothed-train-loss = 0.487525 | smoothed-test-loss = 0.491004 | rel_loss_delta = 0.00012881883264423573 | loss_stagnation_count = 0

Epoch:  98%|█████████▊| 295/300 [17:57<00:19,  3.80s/it]2026-10-20 16:29:50.515 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 296, smoothed-train-loss = 0.487573 | smoothed-test-loss = 0.488816 | rel_loss_delta = 9.94065689176323e-05 | loss_stagnation_count = 0

Epoch:  99%|█████████▊| 296/300 [18:01<00:15,  3.78s/it]2026-10-20 16:29:54.336 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 297, smoothed-train-loss = 0.487529 | smoothed-test-loss = 0.488587 | rel_loss_delta = -9.029661045465183e-05 | loss_stagnation_count = 0

Epoch:  99%|█████████▉| 297/300 [18:05<00:11,  3.79s/it]2026-10-20 16:29:58.236 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 298, smoothed-train-loss = 0.487516 | smoothed-test-loss = 0.489759 | rel_loss_delta = -2.717243518591073e-05 | loss_stagnation_count = 1

Epoch:  99%|█████████▉| 298/300 [18:09<00:07,  3.82s/it]2026-10-20 16:30:01.807 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 299, smoothed-train-loss = 0.487528 | smoothed-test-loss = 0.492176 | rel_loss_delta = 2.453879171527877e-05 | loss_stagnation_count = 2

Epoch: 100%|█████████▉| 299/300 [18:12<00:03,  3.75s/it]2026-10-20 16:30:05.806 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 300, smoothed-train-loss = 0.487675 | smoothed-test-loss = 0.490164 | rel_loss_delta = 0.00030219437793306856 | loss_stagnation_count = 0

Epoch: 100%|██████████| 300/300 [18:16<00:00,  3.66s/it]
2026-10-20 16:30:05.916 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:0.48994070291519165
2026-10-20 16:30:05.917 | INFO     | __main__:<module>:297 - Given num_reflow_iterations = 0 and current reflow_iteration = 0, no further reflows are needed, exiting the reflow loop
reflow iterations:   0%|          | 0/1 [18:17<?, ?it/s]
2026-10-20 16:30:06.041 | INFO     | __main__:<module>:355 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_radius_0.5_variance_0.0001_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_1024_depth_2_opt_adam_nn_hidden_dim_1024_timestamp_2026-10-20T16:11:47.350580.pkl
2026-10-20 16:30:06.042 | INFO     | __main__:<module>:356 - vt training process finished successfully.

Process finished with exit code 0
```
### Inference 
* model
```log
 2026-10-20 18:06:27.941 | INFO     | __main__:<module>:218 - Parsed args = Namespace(vt_artifact_file_path='models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_radius_0.5_variance_0.0001_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_1024_depth_2_opt_adam_nn_hidden_dim_1024_timestamp_2026-10-20T16:11:47.350580.pkl', n_inference=None, n_time_points=1000, ode_solver_method='dopri8')
2026-10-20 18:06:27.941 | INFO     | __main__:<module>:224 - Loading vt_artifact from models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_radius_0.5_variance_0.0001_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_1024_depth_2_opt_adam_nn_hidden_dim_1024_timestamp_2026-10-20T16:11:47.350580.pkl
2026-10-20 18:06:28.053 | INFO     | __main__:<module>:228 - loaded vt artifact(dict) : 
 {'models': [VelocityFieldRegressionNet(
  (model): Sequential(
    (0): Linear(in_features=3, out_features=1024, bias=True)
    (1): ReLU()
    (2): Linear(in_features=1024, out_features=1024, bias=True)
    (3): ReLU()
    (4): Linear(in_features=1024, out_features=2, bias=True)
  )
)], 'reflows': 0, 'prior': 'gaussian', 'hyperparams': {'hidden_dim': 1024, 'depth': 2, 'lr': 0.001, 'batch_size': 2048, 'epochs': 300}, 'data_snapshot_filepath': 'data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_radius_0.5_variance_0.0001_data_dim_2_n_train_test_eval_500000_200000_200000.pkl', 'trajectory_model': 'global', 'learning_curves': [], 'arch': 'nn', 'opt': 'adam'}
2026-10-20 18:06:28.053 | INFO     | __main__:<module>:229 - vt_model = 
VelocityFieldRegressionNet(
  (model): Sequential(
    (0): Linear(in_features=3, out_features=1024, bias=True)
    (1): ReLU()
    (2): Linear(in_features=1024, out_features=1024, bias=True)
    (3): ReLU()
    (4): Linear(in_features=1024, out_features=2, bias=True)
  )
)
2026-10-20 18:06:28.053 | INFO     | __main__:<module>:230 - num_reflows = 0
2026-10-20 18:06:28.053 | INFO     | __main__:<module>:232 - Loading data_snapshot from data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_radius_0.5_variance_0.0001_data_dim_2_n_train_test_eval_500000_200000_200000.pkl
2026-10-20 18:06:28.063 | INFO     | __main__:<module>:235 - Successfully loaded data_snapshot from data-snapshots/experimental/data_snapshot_gaussian_mixture_2d_ncomp_16_radius_0.5_variance_0.0001_data_dim_2_n_train_test_eval_500000_200000_200000.pkl
```
* results
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : dopri8
  VT Trajectory Model  : global
  Num Reflows          : 0
  Sinkhorn Iterations  : 10
  Inference Samples    : 200,000
  Inference Time       : 772.74s
  Evaluations (t)      : 1000
  Eval Results         : {0.00010213134810328485: {'real_generated': {'mean': 0.002906250162050128, 'stddev': 0.00014379106611357926}, 'real_real': {'mean': 0.00023044858826324344, 'stddev': 9.886160176290283e-05}, 'generated_generated': {'mean': 0.00013128062128089368, 'stddev': 4.033033491282356e-05}}, 0.0002553283702582121: {'real_generated': {'mean': 0.0029245412442833185, 'stddev': 0.00013132570205914355}, 'real_real': {'mean': 0.0002181632284191437, 'stddev': 0.00011035967681657179}, 'generated_generated': {'mean': 0.0001543106191093102, 'stddev': 3.367473757408261e-05}}, 0.0005106567405164242: {'real_generated': {'mean': 0.003029116289690137, 'stddev': 0.00021002261815618607}, 'real_real': {'mean': 0.0002616250596474856, 'stddev': 6.992647511602529e-05}, 'generated_generated': {'mean': 0.0001693944534054026, 'stddev': 5.254939588958817e-05}}, 0.0010213134810328484: {'real_generated': {'mean': 0.0029546406120061876, 'stddev': 0.00025009487829112746}, 'real_real': {'mean': 0.00027657995524350555, 'stddev': 7.468971776659444e-05}, 'generated_generated': {'mean': 0.00018496818374842406, 'stddev': 4.780884900493773e-05}}, 0.0020426269620656967: {'real_generated': {'mean': 0.002877194155007601, 'stddev': 0.0001575247880058026}, 'real_real': {'mean': 0.0002503231691662222, 'stddev': 3.662751981354892e-05}, 'generated_generated': {'mean': 0.0001448851340683177, 'stddev': 2.232920611006446e-05}}, 0.01: {'real_generated': {'mean': 0.003070858307182789, 'stddev': 0.00017027076787483973}, 'real_real': {'mean': 0.00030165495700202883, 'stddev': 7.068789969433686e-05}, 'generated_generated': {'mean': 0.0001762472529662773, 'stddev': 4.350601305468614e-05}}, 0.02: {'real_generated': {'mean': 0.0029051640536636113, 'stddev': 0.00020410863933152155}, 'real_real': {'mean': 0.00019773559470195323, 'stddev': 4.757533456229235e-05}, 'generated_generated': {'mean': 0.00013069827109575273, 'stddev': 4.956238418664886e-05}}, 0.05: {'real_generated': {'mean': 0.0020820000674575566, 'stddev': 9.349219744869961e-05}, 'real_real': {'mean': 0.00018990239186678082, 'stddev': 3.618289198591287e-05}, 'generated_generated': {'mean': 9.810856718104333e-05, 'stddev': 2.420210048902312e-05}}, 0.1: {'real_generated': {'mean': 0.0009970571962185204, 'stddev': 9.438777449824871e-05}, 'real_real': {'mean': 8.279985777335242e-05, 'stddev': 9.414942026528818e-05}, 'generated_generated': {'mean': 6.205401878105477e-05, 'stddev': 1.889992234253631e-05}}, 0.2: {'real_generated': {'mean': 0.0005230666894931347, 'stddev': 0.00026502709924381107}, 'real_real': {'mean': 6.129675293777836e-05, 'stddev': 5.302416135451577e-05}, 'generated_generated': {'mean': 5.469866446219385e-05, 'stddev': 5.460009975838481e-05}}, 0.5: {'real_generated': {'mean': 0.00038806399679742755, 'stddev': 0.0001165454290618424}, 'real_real': {'mean': 1.5092401054062065e-05, 'stddev': 1.0084977581133796e-05}, 'generated_generated': {'mean': 2.558658779889811e-05, 'stddev': 2.0104145480338377e-05}}, 1.0: {'real_generated': {'mean': 0.00034593832097016273, 'stddev': 8.95378370538663e-05}, 'real_real': {'mean': 2.741303997026989e-05, 'stddev': 1.4055503145209753e-05}, 'generated_generated': {'mean': 2.288282266817987e-05, 'stddev': 1.3357737300680312e-05}}, 2.0: {'real_generated': {'mean': 0.00039962463779374957, 'stddev': 0.0001536991113365075}, 'real_real': {'mean': 1.949760644492926e-05, 'stddev': 2.132245909236475e-05}, 'generated_generated': {'mean': 3.359062102390453e-05, 'stddev': 2.0722103425721962e-05}}, 5.0: {'real_generated': {'mean': 0.0003760604769922793, 'stddev': 0.00012426265593070378}, 'real_real': {'mean': 2.8042316625942475e-05, 'stddev': 2.6650917172115546e-05}, 'generated_generated': {'mean': 3.493022559268866e-05, 'stddev': 3.9857071600900144e-05}}, 10.0: {'real_generated': {'mean': 0.0004406242340337485, 'stddev': 0.0002537032315760795}, 'real_real': {'mean': 2.8379438481351828e-05, 'stddev': 2.5771703853320138e-05}, 'generated_generated': {'mean': 6.237018533283845e-06, 'stddev': 5.960714312634631e-06}}}
==================================================
Plot saved successfully to figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_16_radius_0.5_variance_0.0001_timestamp_2026-10-20T18:06:27.941133_trajectory_model_global_ode_solver_method_dopri8_num_reflows_0.png
/home/mbaddar/Documents/mbaddar/ttflow/ttflow/viz/viz.py:232: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  
2026-10-20 18:27:03.762 | INFO     | __main__:<module>:364 - Inference and evaluation process finished successfully!
Solving ODE | Method = dopri8: 100%|██████████| 1.0006959438323975/1.0 [20:35<-00:00, 1234.89s/it] 
```