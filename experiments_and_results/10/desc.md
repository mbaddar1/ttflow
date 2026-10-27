# Desc
* Dataset : 
  * $\pi_0$ = $N(0,I)$
  * $\pi_1 \rightarrow$ Low variance Gaussian mixture 2D, $\text{radius} = 5.0, \text{var} = 0.0001, n_{comp} = 16$
* Objective : conduct an experiment where the flow is now completely linear to discover how curved flows are 

# Training 
## Model
* Paths
  * 1-RecFlow
```
models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_radius_5.0_variance_0.0001_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_1024_depth_2_opt_adam_nn_hidden_dim_1024_timestamp_2026-10-20T19:17:04.403537.pkl
```

  * 2-RecFlow

```log
models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_radius_5.0_variance_0.0001_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_1024_depth_2_opt_adam_nn_hidden_dim_1024_timestamp_2026-10-25T17:03:45.139418.pkl   
```
* Model Arch and specs
```log
2026-10-20 19:17:04.708 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-20 19:17:05.569 | INFO     | __main__:<module>:256 - ***********************************************************
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
```
* log snippet 
```log
2026-10-20 19:34:10.810 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 288, smoothed-train-loss = 4.452551 | smoothed-test-loss = 4.499774 | rel_loss_delta = 5.957793841160341e-06 | loss_stagnation_count = 1
2026-10-20 19:34:14.260 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 289, smoothed-train-loss = 4.454805 | smoothed-test-loss = 4.478199 | rel_loss_delta = 0.0005062594894438422 | loss_stagnation_count = 0
2026-10-20 19:34:17.754 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 290, smoothed-train-loss = 4.451166 | smoothed-test-loss = 4.511261 | rel_loss_delta = -0.0008169556107700059 | loss_stagnation_count = 0
2026-10-20 19:34:21.186 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 291, smoothed-train-loss = 4.451169 | smoothed-test-loss = 4.477276 | rel_loss_delta = 7.127323986902985e-07 | loss_stagnation_count = 1
2026-10-20 19:34:24.672 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 292, smoothed-train-loss = 4.450262 | smoothed-test-loss = 4.488959 | rel_loss_delta = -0.00020381989156086224 | loss_stagnation_count = 0
2026-10-20 19:34:28.133 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 293, smoothed-train-loss = 4.451848 | smoothed-test-loss = 4.497894 | rel_loss_delta = 0.0003562743345470435 | loss_stagnation_count = 0
2026-10-20 19:34:31.641 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 294, smoothed-train-loss = 4.450977 | smoothed-test-loss = 4.483770 | rel_loss_delta = -0.00019561090495403888 | loss_stagnation_count = 0
2026-10-20 19:34:35.135 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 295, smoothed-train-loss = 4.449813 | smoothed-test-loss = 4.495957 | rel_loss_delta = -0.00026144142655384005 | loss_stagnation_count = 0
2026-10-20 19:34:38.676 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 296, smoothed-train-loss = 4.454414 | smoothed-test-loss = 4.488470 | rel_loss_delta = 0.0010340671809288163 | loss_stagnation_count = 0
2026-10-20 19:34:42.285 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 297, smoothed-train-loss = 4.453014 | smoothed-test-loss = 4.484941 | rel_loss_delta = -0.000314452761420204 | loss_stagnation_count = 0
2026-10-20 19:34:45.807 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 298, smoothed-train-loss = 4.450345 | smoothed-test-loss = 4.488129 | rel_loss_delta = -0.0005992495603610302 | loss_stagnation_count = 0
2026-10-20 19:34:49.300 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 299, smoothed-train-loss = 4.451238 | smoothed-test-loss = 4.499469 | rel_loss_delta = 0.00020064519519458708 | loss_stagnation_count = 0
2026-10-20 19:34:52.812 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 300, smoothed-train-loss = 4.452120 | smoothed-test-loss = 4.524581 | rel_loss_delta = 0.00019812637173100947 | loss_stagnation_count = 0
2026-10-20 19:34:52.916 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:4.527371406555176
2026-10-20 19:34:52.917 | INFO     | __main__:<module>:297 - Given num_reflow_iterations = 0 and current reflow_iteration = 0, no further reflows are needed, exiting the reflow loop
2026-10-20 19:34:53.042 | INFO     | __main__:<module>:355 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_16_radius_5.0_variance_0.0001_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_1024_depth_2_opt_adam_nn_hidden_dim_1024_timestamp_2026-10-20T19:17:04.403537.pkl
2026-10-20 19:34:53.042 | INFO     | __main__:<module>:356 - vt training process finished successfully.
 
```
# Inference 
## 1-RecFlow
### `euler`
```log
2026-10-25 20:46:27.399 | INFO     | __main__:<module>:306 - 
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : euler
  VT Trajectory Model  : global
  Num Reflows          : 0
  Inference Samples    : 200,000
  Inference Time       : 113.37s
  Evaluations (t)      : 1000
  Eval Results         : {0.01: {'real_generated': {'mean': 0.14798807799816133, 'stddev': 0.027600873833632543}, 'real_real': {'mean': 0.031197956949472427, 'stddev': 0.019636812991331333}, 'generated_generated': {'mean': 0.02682473585009575, 'stddev': 0.017295537914265975}}, 0.02: {'real_generated': {'mean': 0.14464247822761536, 'stddev': 0.0345513572126019}, 'real_real': {'mean': 0.03423663899302483, 'stddev': 0.006530169817913573}, 'generated_generated': {'mean': 0.019617877528071405, 'stddev': 0.005051437823553485}}, 0.05: {'real_generated': {'mean': 0.18276061713695527, 'stddev': 0.04770077168222224}, 'real_real': {'mean': 0.027442961931228638, 'stddev': 0.004827480172108056}, 'generated_generated': {'mean': 0.027476074919104577, 'stddev': 0.015250204869016196}}, 0.1: {'real_generated': {'mean': 0.1588472306728363, 'stddev': 0.026363140444862886}, 'real_real': {'mean': 0.03518671914935112, 'stddev': 0.010400045691178725}, 'generated_generated': {'mean': 0.019667621329426767, 'stddev': 0.008043413030401928}}, 0.2: {'real_generated': {'mean': 0.18420564830303193, 'stddev': 0.03759000369739014}, 'real_real': {'mean': 0.026647913828492166, 'stddev': 0.004406427725675187}, 'generated_generated': {'mean': 0.018378240801393986, 'stddev': 0.005325744592151193}}, 0.5: {'real_generated': {'mean': 0.12543603777885437, 'stddev': 0.02158081457316151}, 'real_real': {'mean': 0.031124579161405562, 'stddev': 0.008517388055837323}, 'generated_generated': {'mean': 0.012765251658856868, 'stddev': 0.006509885335289499}}, 1.0: {'real_generated': {'mean': 0.07063236907124519, 'stddev': 0.01183820400299384}, 'real_real': {'mean': 0.007536591030657291, 'stddev': 0.003529180295863758}, 'generated_generated': {'mean': 0.007835856452584267, 'stddev': 0.004610973791808413}}, 2.0: {'real_generated': {'mean': 0.05417515784502029, 'stddev': 0.014098666003951517}, 'real_real': {'mean': 0.0040034180041402575, 'stddev': 0.0020894385241719752}, 'generated_generated': {'mean': 0.008302569435909391, 'stddev': 0.008736715053283866}}, 5.0: {'real_generated': {'mean': 0.035006043314933774, 'stddev': 0.01213614753971587}, 'real_real': {'mean': 0.0028985088458284734, 'stddev': 0.0027463960447198545}, 'generated_generated': {'mean': 0.0014770789653994142, 'stddev': 0.0017179921988548447}}, 10.0: {'real_generated': {'mean': 0.019225110299885273, 'stddev': 0.010933163570882925}, 'real_real': {'mean': 0.0017393854912370443, 'stddev': 0.0009635582116313749}, 'generated_generated': {'mean': 0.0035054056672379373, 'stddev': 0.0021254761769720753}}}
==================================================
2026-10-25 20:46:27.399 | INFO     | __main__:<module>:330 - Staring visualization...
2026-10-25 20:46:27.399 | INFO     | __main__:<module>:335 - Visualizing gaussian_mixture_2d
/home/mbaddar/Documents/mbaddar/ttflow/ttflow/viz/viz.py:1056: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
2026-10-25 20:51:17.440 | INFO     | __main__:<module>:348 - Inference figure saved in: figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_16_radius_5.0_variance_0.0001_timestamp_2026-10-25T20:41:32.601197_trajectory_model_global_ode_solver_method_euler_num_reflows_0.png
2026-10-25 20:51:17.440 | INFO     | __main__:<module>:349 - Inference and evaluation process finished successfully!
Solving ODE | Method = euler: 100%|█████████▉| 0.9989989995956421/1.0 [09:44<00:00, 585.37s/it]
```
### `rk4`
```log 

sinkhorn-blurs: 100%|██████████| 10/10 [02:58<00:00, 17.88s/it]
2026-10-25 14:54:33.860 | INFO     | __main__:<module>:306 - 
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : rk4
  VT Trajectory Model  : global
  Num Reflows          : 0
  Inference Samples    : 200,000
  Inference Time       : 455.99s
  Evaluations (t)      : 1000
  Eval Results         : {0.01: {'real_generated': {'mean': 0.1819908946752548, 'stddev': 0.04268294283748561}, 'real_real': {'mean': 0.031197956949472427, 'stddev': 0.019636812991331333}, 'generated_generated': {'mean': 0.026781808584928513, 'stddev': 0.01470766645840225}}, 0.02: {'real_generated': {'mean': 0.16637404561042785, 'stddev': 0.03889284271202869}, 'real_real': {'mean': 0.03423663899302483, 'stddev': 0.006530169817913573}, 'generated_generated': {'mean': 0.018249526806175708, 'stddev': 0.007472810073898533}}, 0.05: {'real_generated': {'mean': 0.19095224738121033, 'stddev': 0.0437234130682182}, 'real_real': {'mean': 0.027442961931228638, 'stddev': 0.004827480172108056}, 'generated_generated': {'mean': 0.017249255254864693, 'stddev': 0.008979036002905809}}, 0.1: {'real_generated': {'mean': 0.16476795673370362, 'stddev': 0.03445234998676567}, 'real_real': {'mean': 0.03518671914935112, 'stddev': 0.010400045691178725}, 'generated_generated': {'mean': 0.022711730003356932, 'stddev': 0.008257389232802092}}, 0.2: {'real_generated': {'mean': 0.1484043315052986, 'stddev': 0.0324491838295375}, 'real_real': {'mean': 0.026647913828492166, 'stddev': 0.004406427725675187}, 'generated_generated': {'mean': 0.018688863702118398, 'stddev': 0.0069366316897560435}}, 0.5: {'real_generated': {'mean': 0.1271915003657341, 'stddev': 0.03766666361075043}, 'real_real': {'mean': 0.031124579161405562, 'stddev': 0.008517388055837323}, 'generated_generated': {'mean': 0.015914807841181756, 'stddev': 0.007386988541036855}}, 1.0: {'real_generated': {'mean': 0.06448425278067589, 'stddev': 0.010447964209460515}, 'real_real': {'mean': 0.007536591030657291, 'stddev': 0.003529180295863758}, 'generated_generated': {'mean': 0.01100860582664609, 'stddev': 0.004418843417744633}}, 2.0: {'real_generated': {'mean': 0.050712347775697705, 'stddev': 0.011774797488935396}, 'real_real': {'mean': 0.0040034180041402575, 'stddev': 0.0020894385241719752}, 'generated_generated': {'mean': 0.012594960443675519, 'stddev': 0.012241740719348158}}, 5.0: {'real_generated': {'mean': 0.034301198273897174, 'stddev': 0.00517818667592556}, 'real_real': {'mean': 0.0028985088458284734, 'stddev': 0.0027463960447198545}, 'generated_generated': {'mean': 0.001470844610594213, 'stddev': 0.0006360504475082133}}, 10.0: {'real_generated': {'mean': 0.0225984800606966, 'stddev': 0.007599507527859347}, 'real_real': {'mean': 0.0017393854912370443, 'stddev': 0.0009635582116313749}, 'generated_generated': {'mean': 0.0025062445434741675, 'stddev': 0.002241726533804641}}}
==================================================
2026-10-25 14:54:33.860 | INFO     | __main__:<module>:330 - Staring visualization...
2026-10-25 14:54:33.860 | INFO     | __main__:<module>:335 - Visualizing gaussian_mixture_2d
/home/mbaddar/Documents/mbaddar/ttflow/ttflow/viz/viz.py:1056: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
2026-10-25 14:59:21.761 | INFO     | __main__:<module>:348 - Inference figure saved in: figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_16_radius_5.0_variance_0.0001_timestamp_2026-10-25T14:43:56.341703_trajectory_model_global_ode_solver_method_rk4_num_reflows_0.png
2026-10-25 14:59:21.761 | INFO     | __main__:<module>:349 - Inference and evaluation process finished successfully!
Solving ODE | Method = rk4: 100%|██████████| 1.0/1.0 [15:25<00:00, 925.37s/it]
```
### `dopri8`
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : dopri8
  VT Trajectory Model  : global
  Num Reflows          : 0
  Inference Samples    : 200,000
  Inference Time       : 1011.16s
  Evaluations (t)      : 1000
  Eval Results         : {0.01: {'real_generated': {'mean': 0.14568061977624894, 'stddev': 0.028572584242006645}, 'real_real': {'mean': 0.031197956949472427, 'stddev': 0.019636812991331333}, 'generated_generated': {'mean': 0.020366404578089713, 'stddev': 0.0073067434104595215}}, 0.02: {'real_generated': {'mean': 0.13735740184783934, 'stddev': 0.03040292265928205}, 'real_real': {'mean': 0.03423663899302483, 'stddev': 0.006530169817913573}, 'generated_generated': {'mean': 0.02191299796104431, 'stddev': 0.01273649968492268}}, 0.05: {'real_generated': {'mean': 0.14877895414829254, 'stddev': 0.042264417437503574}, 'real_real': {'mean': 0.027442961931228638, 'stddev': 0.004827480172108056}, 'generated_generated': {'mean': 0.01567539293318987, 'stddev': 0.004033386440167646}}, 0.1: {'real_generated': {'mean': 0.14765023291110993, 'stddev': 0.02782703451751761}, 'real_real': {'mean': 0.03518671914935112, 'stddev': 0.010400045691178725}, 'generated_generated': {'mean': 0.016472127102315427, 'stddev': 0.006461290124044105}}, 0.2: {'real_generated': {'mean': 0.14811024963855743, 'stddev': 0.011874172406229322}, 'real_real': {'mean': 0.026647913828492166, 'stddev': 0.004406427725675187}, 'generated_generated': {'mean': 0.013872570358216763, 'stddev': 0.003910212666416282}}, 0.5: {'real_generated': {'mean': 0.11058982461690903, 'stddev': 0.02987053267930355}, 'real_real': {'mean': 0.031124579161405562, 'stddev': 0.008517388055837323}, 'generated_generated': {'mean': 0.01920442543923855, 'stddev': 0.014237691260638431}}, 1.0: {'real_generated': {'mean': 0.06365195363759994, 'stddev': 0.018732064404891575}, 'real_real': {'mean': 0.007536591030657291, 'stddev': 0.003529180295863758}, 'generated_generated': {'mean': 0.009541903110221028, 'stddev': 0.0047544064310402695}}, 2.0: {'real_generated': {'mean': 0.044527113810181616, 'stddev': 0.018707603138708968}, 'real_real': {'mean': 0.0040034180041402575, 'stddev': 0.0020894385241719752}, 'generated_generated': {'mean': 0.004208287433721125, 'stddev': 0.002719479580753069}}, 5.0: {'real_generated': {'mean': 0.03217872902750969, 'stddev': 0.012654308397072217}, 'real_real': {'mean': 0.0028985088458284734, 'stddev': 0.0027463960447198545}, 'generated_generated': {'mean': 0.002992409560829401, 'stddev': 0.0013914736704487148}}, 10.0: {'real_generated': {'mean': 0.018064714036881924, 'stddev': 0.0066939370707473}, 'real_real': {'mean': 0.0017393854912370443, 'stddev': 0.0009635582116313749}, 'generated_generated': {'mean': 0.001279804133810103, 'stddev': 0.0011380473391559354}}}
==================================================
2026-10-25 11:54:58.304 | INFO     | __main__:<module>:330 - Staring visualization...
2026-10-25 11:54:58.304 | INFO     | __main__:<module>:335 - Visualizing gaussian_mixture_2d
/home/mbaddar/Documents/mbaddar/ttflow/ttflow/viz/viz.py:1056: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
2026-10-25 11:59:51.797 | INFO     | __main__:<module>:348 - Inference figure saved in: figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_16_radius_5.0_variance_0.0001_timestamp_2026-10-25T11:35:04.784253_trajectory_model_global_ode_solver_method_dopri8_num_reflows_0.png
2026-10-25 11:59:51.797 | INFO     | __main__:<module>:349 - Inference and evaluation process finished successfully!
Solving ODE | Method = dopri8: 100%|██████████| 1.0017255549319088/1.0 [24:47<-00:02, 1484.46s/it]
```

## 2-RecFlow
### `euler`
```log
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : rk4
  VT Trajectory Model  : global
  Num Reflows          : 1
  Inference Samples    : 200,000
  Inference Time       : 460.71s
  Evaluations (t)      : 1000
  Eval Results         : {0.01: {'real_generated': {'mean': 0.13221558928489685, 'stddev': 0.033291759764870615}, 'real_real': {'mean': 0.031197956949472427, 'stddev': 0.019636812991331333}, 'generated_generated': {'mean': 0.008593258447945118, 'stddev': 0.004534323871616461}}, 0.02: {'real_generated': {'mean': 0.1274097219109535, 'stddev': 0.017664345380738754}, 'real_real': {'mean': 0.03423663899302483, 'stddev': 0.006530169817913573}, 'generated_generated': {'mean': 0.006421312224119902, 'stddev': 0.0022296537241408665}}, 0.05: {'real_generated': {'mean': 0.11848661303520203, 'stddev': 0.006662963436458072}, 'real_real': {'mean': 0.027442961931228638, 'stddev': 0.004827480172108056}, 'generated_generated': {'mean': 0.009901288244873286, 'stddev': 0.006450384696140008}}, 0.1: {'real_generated': {'mean': 0.11972099691629409, 'stddev': 0.016393599677176986}, 'real_real': {'mean': 0.03518671914935112, 'stddev': 0.010400045691178725}, 'generated_generated': {'mean': 0.01351709645241499, 'stddev': 0.010329839942538499}}, 0.2: {'real_generated': {'mean': 0.09903215020895004, 'stddev': 0.01768020916158259}, 'real_real': {'mean': 0.026647913828492166, 'stddev': 0.004406427725675187}, 'generated_generated': {'mean': 0.007220345921814442, 'stddev': 0.003035735116968227}}, 0.5: {'real_generated': {'mean': 0.06465059965848922, 'stddev': 0.011161024725241695}, 'real_real': {'mean': 0.031124579161405562, 'stddev': 0.008517388055837323}, 'generated_generated': {'mean': 0.008232101518660783, 'stddev': 0.002689437961982343}}, 1.0: {'real_generated': {'mean': 0.029020720347762107, 'stddev': 0.00846988190511865}, 'real_real': {'mean': 0.007536591030657291, 'stddev': 0.003529180295863758}, 'generated_generated': {'mean': 0.00336719008628279, 'stddev': 0.0013217331569018913}}, 2.0: {'real_generated': {'mean': 0.024980196543037892, 'stddev': 0.011517801912534247}, 'real_real': {'mean': 0.0040034180041402575, 'stddev': 0.0020894385241719752}, 'generated_generated': {'mean': 0.0058179001556709405, 'stddev': 0.005508762252685208}}, 5.0: {'real_generated': {'mean': 0.011275862157344819, 'stddev': 0.004756935845115518}, 'real_real': {'mean': 0.0028985088458284734, 'stddev': 0.0027463960447198545}, 'generated_generated': {'mean': 0.0034019852988421916, 'stddev': 0.0028742681903620513}}, 10.0: {'real_generated': {'mean': 0.007392314821481704, 'stddev': 0.004309222157903455}, 'real_real': {'mean': 0.0017393854912370443, 'stddev': 0.0009635582116313749}, 'generated_generated': {'mean': 0.0020861624740064143, 'stddev': 0.0011973158150349091}}}
==================================================
2026-10-25 21:39:02.494 | INFO     | __main__:<module>:330 - Staring visualization...
2026-10-25 21:39:02.494 | INFO     | __main__:<module>:335 - Visualizing gaussian_mixture_2d
/home/mbaddar/Documents/mbaddar/ttflow/ttflow/viz/viz.py:1056: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
2026-10-25 21:43:57.836 | INFO     | __main__:<module>:348 - Inference figure saved in: figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_16_radius_5.0_variance_0.0001_timestamp_2026-10-25T21:28:17.678041_trajectory_model_global_ode_solver_method_rk4_num_reflows_1.png
2026-10-25 21:43:57.836 | INFO     | __main__:<module>:349 - Inference and evaluation process finished successfully!
Solving ODE | Method = rk4: 100%|██████████| 1.0/1.0 [15:40<00:00, 940.08s/it]
```
### `rk4`
```log
sinkhorn-blurs: 100%|██████████| 10/10 [03:01<00:00, 18.14s/it]
2026-10-25 21:39:02.494 | INFO     | __main__:<module>:306 - 
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : rk4
  VT Trajectory Model  : global
  Num Reflows          : 1
  Inference Samples    : 200,000
  Inference Time       : 460.71s
  Evaluations (t)      : 1000
  Eval Results         : {0.01: {'real_generated': {'mean': 0.13221558928489685, 'stddev': 0.033291759764870615}, 'real_real': {'mean': 0.031197956949472427, 'stddev': 0.019636812991331333}, 'generated_generated': {'mean': 0.008593258447945118, 'stddev': 0.004534323871616461}}, 0.02: {'real_generated': {'mean': 0.1274097219109535, 'stddev': 0.017664345380738754}, 'real_real': {'mean': 0.03423663899302483, 'stddev': 0.006530169817913573}, 'generated_generated': {'mean': 0.006421312224119902, 'stddev': 0.0022296537241408665}}, 0.05: {'real_generated': {'mean': 0.11848661303520203, 'stddev': 0.006662963436458072}, 'real_real': {'mean': 0.027442961931228638, 'stddev': 0.004827480172108056}, 'generated_generated': {'mean': 0.009901288244873286, 'stddev': 0.006450384696140008}}, 0.1: {'real_generated': {'mean': 0.11972099691629409, 'stddev': 0.016393599677176986}, 'real_real': {'mean': 0.03518671914935112, 'stddev': 0.010400045691178725}, 'generated_generated': {'mean': 0.01351709645241499, 'stddev': 0.010329839942538499}}, 0.2: {'real_generated': {'mean': 0.09903215020895004, 'stddev': 0.01768020916158259}, 'real_real': {'mean': 0.026647913828492166, 'stddev': 0.004406427725675187}, 'generated_generated': {'mean': 0.007220345921814442, 'stddev': 0.003035735116968227}}, 0.5: {'real_generated': {'mean': 0.06465059965848922, 'stddev': 0.011161024725241695}, 'real_real': {'mean': 0.031124579161405562, 'stddev': 0.008517388055837323}, 'generated_generated': {'mean': 0.008232101518660783, 'stddev': 0.002689437961982343}}, 1.0: {'real_generated': {'mean': 0.029020720347762107, 'stddev': 0.00846988190511865}, 'real_real': {'mean': 0.007536591030657291, 'stddev': 0.003529180295863758}, 'generated_generated': {'mean': 0.00336719008628279, 'stddev': 0.0013217331569018913}}, 2.0: {'real_generated': {'mean': 0.024980196543037892, 'stddev': 0.011517801912534247}, 'real_real': {'mean': 0.0040034180041402575, 'stddev': 0.0020894385241719752}, 'generated_generated': {'mean': 0.0058179001556709405, 'stddev': 0.005508762252685208}}, 5.0: {'real_generated': {'mean': 0.011275862157344819, 'stddev': 0.004756935845115518}, 'real_real': {'mean': 0.0028985088458284734, 'stddev': 0.0027463960447198545}, 'generated_generated': {'mean': 0.0034019852988421916, 'stddev': 0.0028742681903620513}}, 10.0: {'real_generated': {'mean': 0.007392314821481704, 'stddev': 0.004309222157903455}, 'real_real': {'mean': 0.0017393854912370443, 'stddev': 0.0009635582116313749}, 'generated_generated': {'mean': 0.0020861624740064143, 'stddev': 0.0011973158150349091}}}
==================================================
```
### `dopri8`
```log
2026-10-25 21:59:42.400 | INFO     | __main__:<module>:306 - 
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : dopri8
  VT Trajectory Model  : global
  Num Reflows          : 1
  Inference Samples    : 200,000
  Inference Time       : 570.23s
  Evaluations (t)      : 1000
  Eval Results         : {0.01: {'real_generated': {'mean': 0.12809525281190873, 'stddev': 0.017518088575901883}, 'real_real': {'mean': 0.031197956949472427, 'stddev': 0.019636812991331333}, 'generated_generated': {'mean': 0.017880380153656006, 'stddev': 0.008105459840843527}}, 0.02: {'real_generated': {'mean': 0.13121765553951265, 'stddev': 0.02366619246238063}, 'real_real': {'mean': 0.03423663899302483, 'stddev': 0.006530169817913573}, 'generated_generated': {'mean': 0.009018412325531244, 'stddev': 0.004438046394816861}}, 0.05: {'real_generated': {'mean': 0.11930734664201736, 'stddev': 0.014978580751812437}, 'real_real': {'mean': 0.027442961931228638, 'stddev': 0.004827480172108056}, 'generated_generated': {'mean': 0.01482510743662715, 'stddev': 0.009975758111542618}}, 0.1: {'real_generated': {'mean': 0.12241679728031159, 'stddev': 0.012724750334326514}, 'real_real': {'mean': 0.03518671914935112, 'stddev': 0.010400045691178725}, 'generated_generated': {'mean': 0.018632003851234914, 'stddev': 0.013218194082410602}}, 0.2: {'real_generated': {'mean': 0.10101294815540314, 'stddev': 0.02663604999730374}, 'real_real': {'mean': 0.026647913828492166, 'stddev': 0.004406427725675187}, 'generated_generated': {'mean': 0.01510402886196971, 'stddev': 0.010131965423201134}}, 0.5: {'real_generated': {'mean': 0.06957464814186096, 'stddev': 0.03327378891586979}, 'real_real': {'mean': 0.031124579161405562, 'stddev': 0.008517388055837323}, 'generated_generated': {'mean': 0.01044829236343503, 'stddev': 0.007204276217511686}}, 1.0: {'real_generated': {'mean': 0.03061770908534527, 'stddev': 0.018373876586404787}, 'real_real': {'mean': 0.007536591030657291, 'stddev': 0.003529180295863758}, 'generated_generated': {'mean': 0.006886934279464185, 'stddev': 0.00644891774976357}}, 2.0: {'real_generated': {'mean': 0.02285513561218977, 'stddev': 0.010089044948701574}, 'real_real': {'mean': 0.0040034180041402575, 'stddev': 0.0020894385241719752}, 'generated_generated': {'mean': 0.004166374169290066, 'stddev': 0.0020560779041065296}}, 5.0: {'real_generated': {'mean': 0.014651172608137131, 'stddev': 0.00542059236456032}, 'real_real': {'mean': 0.0028985088458284734, 'stddev': 0.0027463960447198545}, 'generated_generated': {'mean': 0.0022442227695137263, 'stddev': 0.0009297131800371098}}, 10.0: {'real_generated': {'mean': 0.0066716201603412625, 'stddev': 0.005497184256709716}, 'real_real': {'mean': 0.0017393854912370443, 'stddev': 0.0009635582116313749}, 'generated_generated': {'mean': 0.0022175826132297518, 'stddev': 0.0007484468701592652}}}
==================================================
2026-10-25 21:59:42.400 | INFO     | __main__:<module>:330 - Staring visualization...
2026-10-25 21:59:42.400 | INFO     | __main__:<module>:335 - Visualizing gaussian_mixture_2d
/home/mbaddar/Documents/mbaddar/ttflow/ttflow/viz/viz.py:1056: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
2026-10-25 22:04:35.592 | INFO     | __main__:<module>:348 - Inference figure saved in: figures/inference/experimental/inference_gaussian_mixture_2d_data_params_num_components_16_radius_5.0_variance_0.0001_timestamp_2026-10-25T21:47:04.168454_trajectory_model_global_ode_solver_method_dopri8_num_reflows_1.png
2026-10-25 22:04:35.592 | INFO     | __main__:<module>:349 - Inference and evaluation process finished successfully!
Solving ODE | Method = dopri8: 100%|██████████| 1.0037046670913696/1.0 [17:31<-00:03, 1047.57s/it]
```