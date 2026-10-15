# Training 
```log
2026-10-13 15:42:52.064 | INFO     | __main__:<module>:209 - data pkl size = 0.01 GB
2026-10-13 15:42:52.067 | INFO     | __main__:<module>:220 - Generating training data for the global trajectory training process..
2026-10-13 15:42:52.246 | INFO     | ttflow.training.nn.nn_models:__init__:56 - Network is on: cuda:0
2026-10-13 15:42:52.906 | INFO     | __main__:<module>:247 - ***********************************************************Training Model Specs:
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
    - Epochs:                   200
    - Batch Size:               2048
    - Patience:                 10
    - Min Rel Loss Delta:       5e-05
    - ZT Mode:                  linear
    - Reflow Iterations:  0**********************************************************
2026-10-13 15:42:52.907 | INFO     | __main__:<module>:272 - Reflow iteration: 0
2026-10-13 15:42:53.855 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:57 - @epoch = 0 (before training), loss_fn = MSELoss(), loss_value = 51.186342
2026-10-13 15:42:53.860 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:64 - MSELoss(test) before training:51.18634033203125
2026-10-13 15:42:57.009 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 1, smoothed-train-loss = 14.677600 | smoothed-test-loss = 10.018496 | rel_loss_delta = inf | loss_stagnation_count = 0
2026-10-13 15:43:00.062 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 2, smoothed-train-loss = 10.028801 | smoothed-test-loss = 9.423429 | rel_loss_delta = inf | loss_stagnation_count = 0
2026-10-13 15:43:03.025 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 3, smoothed-train-loss = 9.295859 | smoothed-test-loss = 9.168681 | rel_loss_delta = -0.0730836649398843 | loss_stagnation_count = 0
2026-10-13 15:43:06.036 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 4, smoothed-train-loss = 9.031790 | smoothed-test-loss = 8.952797 | rel_loss_delta = -0.02840718075728317 | loss_stagnation_count = 0
2026-10-13 15:43:09.221 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 5, smoothed-train-loss = 8.829838 | smoothed-test-loss = 8.757951 | rel_loss_delta = -0.02236020237913485 | loss_stagnation_count = 0
2026-10-13 15:43:12.171 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 6, smoothed-train-loss = 8.656136 | smoothed-test-loss = 8.630668 | rel_loss_delta = -0.019672118972531266 | loss_stagnation_count = 0
2026-10-13 15:43:15.145 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 7, smoothed-train-loss = 8.521914 | smoothed-test-loss = 8.492342 | rel_loss_delta = -0.01550594579352765 | loss_stagnation_count = 0
2026-10-13 15:43:18.194 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 8, smoothed-train-loss = 8.419753 | smoothed-test-loss = 8.381877 | rel_loss_delta = -0.01198807828881199 | loss_stagnation_count = 0
2026-10-13 15:43:21.137 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 9, smoothed-train-loss = 8.348506 | smoothed-test-loss = 8.319104 | rel_loss_delta = -0.008461909664012193 | loss_stagnation_count = 0
2026-10-13 15:43:24.264 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 10, smoothed-train-loss = 8.288246 | smoothed-test-loss = 8.266588 | rel_loss_delta = -0.007218043300811793 | loss_stagnation_count = 0
2026-10-13 15:43:27.289 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 11, smoothed-train-loss = 8.247131 | smoothed-test-loss = 8.262155 | rel_loss_delta = -0.004960682875001971 | loss_stagnation_count = 0
2026-10-13 15:43:30.352 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 12, smoothed-train-loss = 8.223192 | smoothed-test-loss = 8.221253 | rel_loss_delta = -0.002902688239535655 | loss_stagnation_count = 0
2026-10-13 15:43:33.404 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 13, smoothed-train-loss = 8.199009 | smoothed-test-loss = 8.195501 | rel_loss_delta = -0.0029407431614240595 | loss_stagnation_count = 0
2026-10-13 15:43:36.515 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 14, smoothed-train-loss = 8.180880 | smoothed-test-loss = 8.184025 | rel_loss_delta = -0.00221118730862746 | loss_stagnation_count = 0
2026-10-13 15:43:39.589 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 15, smoothed-train-loss = 8.172882 | smoothed-test-loss = 8.165162 | rel_loss_delta = -0.0009776332353353605 | loss_stagnation_count = 0
2026-10-13 15:43:42.607 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 16, smoothed-train-loss = 8.157087 | smoothed-test-loss = 8.169201 | rel_loss_delta = -0.0019325789296926032 | loss_stagnation_count = 0
2026-10-13 15:43:45.544 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 17, smoothed-train-loss = 8.149897 | smoothed-test-loss = 8.154267 | rel_loss_delta = -0.0008814339964606628 | loss_stagnation_count = 0
2026-10-13 15:43:48.499 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 18, smoothed-train-loss = 8.142875 | smoothed-test-loss = 8.171570 | rel_loss_delta = -0.0008616799986997176 | loss_stagnation_count = 0
2026-10-13 15:43:51.520 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 19, smoothed-train-loss = 8.141905 | smoothed-test-loss = 8.151446 | rel_loss_delta = -0.00011912194576371441 | loss_stagnation_count = 0
2026-10-13 15:43:54.599 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 20, smoothed-train-loss = 8.135027 | smoothed-test-loss = 8.124059 | rel_loss_delta = -0.0008447473676691873 | loss_stagnation_count = 0
2026-10-13 15:43:57.529 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 21, smoothed-train-loss = 8.133978 | smoothed-test-loss = 8.134032 | rel_loss_delta = -0.00012898093341213597 | loss_stagnation_count = 0
2026-10-13 15:44:00.705 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 22, smoothed-train-loss = 8.130427 | smoothed-test-loss = 8.202633 | rel_loss_delta = -0.00043653151097769954 | loss_stagnation_count = 0
2026-10-13 15:44:03.774 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 23, smoothed-train-loss = 8.129479 | smoothed-test-loss = 8.129674 | rel_loss_delta = -0.00011661374631138897 | loss_stagnation_count = 0
2026-10-13 15:44:06.813 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 24, smoothed-train-loss = 8.125716 | smoothed-test-loss = 8.127243 | rel_loss_delta = -0.0004628724713040573 | loss_stagnation_count = 0
2026-10-13 15:44:09.722 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 25, smoothed-train-loss = 8.122475 | smoothed-test-loss = 8.132512 | rel_loss_delta = -0.0003987849050043536 | loss_stagnation_count = 0
2026-10-13 15:44:12.833 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 26, smoothed-train-loss = 8.123019 | smoothed-test-loss = 8.151199 | rel_loss_delta = 6.693272093537255e-05 | loss_stagnation_count = 0
2026-10-13 15:44:15.828 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 27, smoothed-train-loss = 8.120558 | smoothed-test-loss = 8.139248 | rel_loss_delta = -0.00030293251591229916 | loss_stagnation_count = 0
2026-10-13 15:44:18.969 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 28, smoothed-train-loss = 8.119733 | smoothed-test-loss = 8.131355 | rel_loss_delta = -0.00010164588259562439 | loss_stagnation_count = 0
2026-10-13 15:44:21.973 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 29, smoothed-train-loss = 8.118055 | smoothed-test-loss = 8.129376 | rel_loss_delta = -0.0002066317913719565 | loss_stagnation_count = 0
2026-10-13 15:44:24.996 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 30, smoothed-train-loss = 8.116495 | smoothed-test-loss = 8.112181 | rel_loss_delta = -0.00019223032319170384 | loss_stagnation_count = 0
2026-10-13 15:44:28.020 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 31, smoothed-train-loss = 8.113249 | smoothed-test-loss = 8.108502 | rel_loss_delta = -0.0003998493379688019 | loss_stagnation_count = 0
2026-10-13 15:44:31.049 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 32, smoothed-train-loss = 8.112442 | smoothed-test-loss = 8.146670 | rel_loss_delta = -9.952150003197023e-05 | loss_stagnation_count = 0
2026-10-13 15:44:33.984 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 33, smoothed-train-loss = 8.110257 | smoothed-test-loss = 8.119982 | rel_loss_delta = -0.00026935142136848977 | loss_stagnation_count = 0
2026-10-13 15:44:37.079 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 34, smoothed-train-loss = 8.113246 | smoothed-test-loss = 8.119298 | rel_loss_delta = 0.00036862037966923025 | loss_stagnation_count = 0
2026-10-13 15:44:40.085 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 35, smoothed-train-loss = 8.107242 | smoothed-test-loss = 8.159435 | rel_loss_delta = -0.0007401097783287682 | loss_stagnation_count = 0
2026-10-13 15:44:43.161 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 36, smoothed-train-loss = 8.112368 | smoothed-test-loss = 8.147100 | rel_loss_delta = 0.000632267366571938 | loss_stagnation_count = 0
2026-10-13 15:44:46.117 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 37, smoothed-train-loss = 8.109340 | smoothed-test-loss = 8.148000 | rel_loss_delta = -0.0003732343927404783 | loss_stagnation_count = 0
2026-10-13 15:44:49.160 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 38, smoothed-train-loss = 8.106993 | smoothed-test-loss = 8.113962 | rel_loss_delta = -0.0002894306295225361 | loss_stagnation_count = 0
2026-10-13 15:44:52.358 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 39, smoothed-train-loss = 8.108509 | smoothed-test-loss = 8.140051 | rel_loss_delta = 0.0001870427631665205 | loss_stagnation_count = 0
2026-10-13 15:44:55.487 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 40, smoothed-train-loss = 8.107085 | smoothed-test-loss = 8.155093 | rel_loss_delta = -0.0001755795099083652 | loss_stagnation_count = 0
2026-10-13 15:44:58.408 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 41, smoothed-train-loss = 8.108839 | smoothed-test-loss = 8.138966 | rel_loss_delta = 0.00021625905064034023 | loss_stagnation_count = 0
2026-10-13 15:45:01.426 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 42, smoothed-train-loss = 8.107655 | smoothed-test-loss = 8.107692 | rel_loss_delta = -0.00014597544090570443 | loss_stagnation_count = 0
2026-10-13 15:45:04.549 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 43, smoothed-train-loss = 8.103694 | smoothed-test-loss = 8.139849 | rel_loss_delta = -0.0004885613548815273 | loss_stagnation_count = 0
2026-10-13 15:45:07.682 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 44, smoothed-train-loss = 8.101191 | smoothed-test-loss = 8.102714 | rel_loss_delta = -0.0003088997040134982 | loss_stagnation_count = 0
2026-10-13 15:45:10.636 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 45, smoothed-train-loss = 8.099266 | smoothed-test-loss = 8.134438 | rel_loss_delta = -0.0002376065571214405 | loss_stagnation_count = 0
2026-10-13 15:45:13.604 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 46, smoothed-train-loss = 8.102098 | smoothed-test-loss = 8.129132 | rel_loss_delta = 0.0003497546223128099 | loss_stagnation_count = 0
2026-10-13 15:45:16.608 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 47, smoothed-train-loss = 8.102703 | smoothed-test-loss = 8.097889 | rel_loss_delta = 7.458930712509455e-05 | loss_stagnation_count = 0
2026-10-13 15:45:19.690 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 48, smoothed-train-loss = 8.101875 | smoothed-test-loss = 8.116757 | rel_loss_delta = -0.00010211206340790752 | loss_stagnation_count = 0
2026-10-13 15:45:22.740 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 49, smoothed-train-loss = 8.102343 | smoothed-test-loss = 8.136017 | rel_loss_delta = 5.7671902594796766e-05 | loss_stagnation_count = 0
2026-10-13 15:45:25.869 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 50, smoothed-train-loss = 8.102332 | smoothed-test-loss = 8.127642 | rel_loss_delta = -1.3628213619732078e-06 | loss_stagnation_count = 1
2026-10-13 15:45:28.875 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 51, smoothed-train-loss = 8.100990 | smoothed-test-loss = 8.098293 | rel_loss_delta = -0.0001655517522914912 | loss_stagnation_count = 0
2026-10-13 15:45:32.145 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 52, smoothed-train-loss = 8.098753 | smoothed-test-loss = 8.106758 | rel_loss_delta = -0.00027615016528921067 | loss_stagnation_count = 0
2026-10-13 15:45:35.126 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 53, smoothed-train-loss = 8.099764 | smoothed-test-loss = 8.119478 | rel_loss_delta = 0.00012484745468827773 | loss_stagnation_count = 0
2026-10-13 15:45:38.280 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 54, smoothed-train-loss = 8.096120 | smoothed-test-loss = 8.132678 | rel_loss_delta = -0.0004499348322572118 | loss_stagnation_count = 0
2026-10-13 15:45:41.393 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 55, smoothed-train-loss = 8.098904 | smoothed-test-loss = 8.115640 | rel_loss_delta = 0.00034390836554800753 | loss_stagnation_count = 0
2026-10-13 15:45:44.347 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 56, smoothed-train-loss = 8.100741 | smoothed-test-loss = 8.134891 | rel_loss_delta = 0.00022675515086234503 | loss_stagnation_count = 0
2026-10-13 15:45:47.174 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 57, smoothed-train-loss = 8.098037 | smoothed-test-loss = 8.118212 | rel_loss_delta = -0.00033369071814473843 | loss_stagnation_count = 0
2026-10-13 15:45:50.147 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 58, smoothed-train-loss = 8.096821 | smoothed-test-loss = 8.107389 | rel_loss_delta = -0.00015024032805160064 | loss_stagnation_count = 0
2026-10-13 15:45:53.154 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 59, smoothed-train-loss = 8.094402 | smoothed-test-loss = 8.104827 | rel_loss_delta = -0.0002986909026192847 | loss_stagnation_count = 0
2026-10-13 15:45:56.225 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 60, smoothed-train-loss = 8.098625 | smoothed-test-loss = 8.097560 | rel_loss_delta = 0.0005216184912702564 | loss_stagnation_count = 0
2026-10-13 15:45:59.218 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 61, smoothed-train-loss = 8.096568 | smoothed-test-loss = 8.101259 | rel_loss_delta = -0.00025393948074583584 | loss_stagnation_count = 0
2026-10-13 15:46:02.307 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 62, smoothed-train-loss = 8.096023 | smoothed-test-loss = 8.102991 | rel_loss_delta = -6.728278144945995e-05 | loss_stagnation_count = 0
2026-10-13 15:46:05.245 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 63, smoothed-train-loss = 8.096407 | smoothed-test-loss = 8.097957 | rel_loss_delta = 4.734125517056316e-05 | loss_stagnation_count = 1
2026-10-13 15:46:08.315 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 64, smoothed-train-loss = 8.094486 | smoothed-test-loss = 8.111917 | rel_loss_delta = -0.00023722999331573543 | loss_stagnation_count = 0
2026-10-13 15:46:11.227 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 65, smoothed-train-loss = 8.095529 | smoothed-test-loss = 8.122753 | rel_loss_delta = 0.00012884945020179617 | loss_stagnation_count = 0
2026-10-13 15:46:14.185 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 66, smoothed-train-loss = 8.097545 | smoothed-test-loss = 8.108516 | rel_loss_delta = 0.00024908752923825397 | loss_stagnation_count = 0
2026-10-13 15:46:17.176 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 67, smoothed-train-loss = 8.092803 | smoothed-test-loss = 8.092393 | rel_loss_delta = -0.0005856626158897372 | loss_stagnation_count = 0
2026-10-13 15:46:20.164 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 68, smoothed-train-loss = 8.090189 | smoothed-test-loss = 8.112395 | rel_loss_delta = -0.0003230066585940113 | loss_stagnation_count = 0
2026-10-13 15:46:23.155 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 69, smoothed-train-loss = 8.097518 | smoothed-test-loss = 8.100853 | rel_loss_delta = 0.0009059941191202022 | loss_stagnation_count = 0
2026-10-13 15:46:26.207 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 70, smoothed-train-loss = 8.094315 | smoothed-test-loss = 8.094357 | rel_loss_delta = -0.00039557197762887585 | loss_stagnation_count = 0
2026-10-13 15:46:29.179 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 71, smoothed-train-loss = 8.092380 | smoothed-test-loss = 8.120696 | rel_loss_delta = -0.0002390480164068613 | loss_stagnation_count = 0
2026-10-13 15:46:32.237 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 72, smoothed-train-loss = 8.093400 | smoothed-test-loss = 8.100704 | rel_loss_delta = 0.00012596027544090058 | loss_stagnation_count = 0
2026-10-13 15:46:35.110 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 73, smoothed-train-loss = 8.095284 | smoothed-test-loss = 8.124312 | rel_loss_delta = 0.00023287526844846618 | loss_stagnation_count = 0
2026-10-13 15:46:38.334 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 74, smoothed-train-loss = 8.092788 | smoothed-test-loss = 8.114020 | rel_loss_delta = -0.0003083407120378748 | loss_stagnation_count = 0
2026-10-13 15:46:41.472 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 75, smoothed-train-loss = 8.091594 | smoothed-test-loss = 8.092600 | rel_loss_delta = -0.00014757086203563143 | loss_stagnation_count = 0
2026-10-13 15:46:44.460 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 76, smoothed-train-loss = 8.093050 | smoothed-test-loss = 8.102991 | rel_loss_delta = 0.0001799310866027213 | loss_stagnation_count = 0
2026-10-13 15:46:47.273 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 77, smoothed-train-loss = 8.091977 | smoothed-test-loss = 8.104501 | rel_loss_delta = -0.00013263369663714383 | loss_stagnation_count = 0
2026-10-13 15:46:50.348 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 78, smoothed-train-loss = 8.093879 | smoothed-test-loss = 8.151333 | rel_loss_delta = 0.00023512356565230397 | loss_stagnation_count = 0
2026-10-13 15:46:53.400 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 79, smoothed-train-loss = 8.095168 | smoothed-test-loss = 8.129972 | rel_loss_delta = 0.0001592547556462315 | loss_stagnation_count = 0
2026-10-13 15:46:56.541 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 80, smoothed-train-loss = 8.093955 | smoothed-test-loss = 8.087087 | rel_loss_delta = -0.0001498350184648062 | loss_stagnation_count = 0
2026-10-13 15:46:59.456 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 81, smoothed-train-loss = 8.091264 | smoothed-test-loss = 8.085934 | rel_loss_delta = -0.0003325364328929491 | loss_stagnation_count = 0
2026-10-13 15:47:02.531 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 82, smoothed-train-loss = 8.092616 | smoothed-test-loss = 8.091608 | rel_loss_delta = 0.00016710081277355044 | loss_stagnation_count = 0
2026-10-13 15:47:05.502 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 83, smoothed-train-loss = 8.090354 | smoothed-test-loss = 8.092097 | rel_loss_delta = -0.0002794765176659689 | loss_stagnation_count = 0
2026-10-13 15:47:08.569 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 84, smoothed-train-loss = 8.090466 | smoothed-test-loss = 8.118959 | rel_loss_delta = 1.3880060682445311e-05 | loss_stagnation_count = 1
2026-10-13 15:47:11.731 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 85, smoothed-train-loss = 8.093725 | smoothed-test-loss = 8.101584 | rel_loss_delta = 0.0004027374069913128 | loss_stagnation_count = 0
2026-10-13 15:47:14.727 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 86, smoothed-train-loss = 8.089387 | smoothed-test-loss = 8.110624 | rel_loss_delta = -0.0005359880934576131 | loss_stagnation_count = 0
2026-10-13 15:47:17.767 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 87, smoothed-train-loss = 8.091207 | smoothed-test-loss = 8.093612 | rel_loss_delta = 0.00022497916232020813 | loss_stagnation_count = 0
2026-10-13 15:47:20.816 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 88, smoothed-train-loss = 8.091156 | smoothed-test-loss = 8.114391 | rel_loss_delta = -6.236110292614308e-06 | loss_stagnation_count = 1
2026-10-13 15:47:23.832 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 89, smoothed-train-loss = 8.091118 | smoothed-test-loss = 8.130186 | rel_loss_delta = -4.7573431025773165e-06 | loss_stagnation_count = 2
2026-10-13 15:47:26.989 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 90, smoothed-train-loss = 8.095065 | smoothed-test-loss = 8.108541 | rel_loss_delta = 0.00048789819590987275 | loss_stagnation_count = 0
2026-10-13 15:47:30.004 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 91, smoothed-train-loss = 8.094227 | smoothed-test-loss = 8.097001 | rel_loss_delta = -0.00010354974675850412 | loss_stagnation_count = 0
2026-10-13 15:47:33.170 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 92, smoothed-train-loss = 8.090268 | smoothed-test-loss = 8.117933 | rel_loss_delta = -0.0004890865548950066 | loss_stagnation_count = 0
2026-10-13 15:47:36.280 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 93, smoothed-train-loss = 8.091588 | smoothed-test-loss = 8.092709 | rel_loss_delta = 0.00016318490184769155 | loss_stagnation_count = 0
2026-10-13 15:47:39.411 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 94, smoothed-train-loss = 8.089752 | smoothed-test-loss = 8.083715 | rel_loss_delta = -0.00022693301371828187 | loss_stagnation_count = 0
2026-10-13 15:47:42.661 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 95, smoothed-train-loss = 8.087360 | smoothed-test-loss = 8.105617 | rel_loss_delta = -0.0002957276078041474 | loss_stagnation_count = 0
2026-10-13 15:47:45.788 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 96, smoothed-train-loss = 8.088127 | smoothed-test-loss = 8.103365 | rel_loss_delta = 9.483956727002885e-05 | loss_stagnation_count = 0
2026-10-13 15:47:48.775 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 97, smoothed-train-loss = 8.090880 | smoothed-test-loss = 8.107336 | rel_loss_delta = 0.0003403544292942883 | loss_stagnation_count = 0
2026-10-13 15:47:51.788 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 98, smoothed-train-loss = 8.087270 | smoothed-test-loss = 8.103867 | rel_loss_delta = -0.0004461141626374925 | loss_stagnation_count = 0
2026-10-13 15:47:54.813 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 99, smoothed-train-loss = 8.087676 | smoothed-test-loss = 8.105018 | rel_loss_delta = 5.011568619183079e-05 | loss_stagnation_count = 0
2026-10-13 15:47:57.837 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 100, smoothed-train-loss = 8.089395 | smoothed-test-loss = 8.101182 | rel_loss_delta = 0.00021265825486106875 | loss_stagnation_count = 0
2026-10-13 15:48:00.819 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 101, smoothed-train-loss = 8.091674 | smoothed-test-loss = 8.090222 | rel_loss_delta = 0.00028162005122023574 | loss_stagnation_count = 0
2026-10-13 15:48:03.912 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 102, smoothed-train-loss = 8.086717 | smoothed-test-loss = 8.101174 | rel_loss_delta = -0.0006125029995337595 | loss_stagnation_count = 0
2026-10-13 15:48:06.976 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 103, smoothed-train-loss = 8.089722 | smoothed-test-loss = 8.094031 | rel_loss_delta = 0.0003715532199993344 | loss_stagnation_count = 0
2026-10-13 15:48:09.963 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 104, smoothed-train-loss = 8.090110 | smoothed-test-loss = 8.104164 | rel_loss_delta = 4.7919860435271725e-05 | loss_stagnation_count = 1
2026-10-13 15:48:12.915 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 105, smoothed-train-loss = 8.089600 | smoothed-test-loss = 8.111376 | rel_loss_delta = -6.297382818876516e-05 | loss_stagnation_count = 0
2026-10-13 15:48:15.905 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 106, smoothed-train-loss = 8.092449 | smoothed-test-loss = 8.101307 | rel_loss_delta = 0.00035218467117175203 | loss_stagnation_count = 0
2026-10-13 15:48:19.015 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 107, smoothed-train-loss = 8.090750 | smoothed-test-loss = 8.102542 | rel_loss_delta = -0.00020991753842466632 | loss_stagnation_count = 0
2026-10-13 15:48:22.048 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 108, smoothed-train-loss = 8.092363 | smoothed-test-loss = 8.103293 | rel_loss_delta = 0.00019924590617515615 | loss_stagnation_count = 0
2026-10-13 15:48:24.932 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 109, smoothed-train-loss = 8.087032 | smoothed-test-loss = 8.096152 | rel_loss_delta = -0.0006587566671310746 | loss_stagnation_count = 0
2026-10-13 15:48:28.002 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 110, smoothed-train-loss = 8.088764 | smoothed-test-loss = 8.089018 | rel_loss_delta = 0.0002142470366077647 | loss_stagnation_count = 0
2026-10-13 15:48:31.057 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 111, smoothed-train-loss = 8.087749 | smoothed-test-loss = 8.099179 | rel_loss_delta = -0.00012554603461939308 | loss_stagnation_count = 0
2026-10-13 15:48:34.041 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 112, smoothed-train-loss = 8.090233 | smoothed-test-loss = 8.125347 | rel_loss_delta = 0.00030721287758625596 | loss_stagnation_count = 0
2026-10-13 15:48:36.922 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 113, smoothed-train-loss = 8.087963 | smoothed-test-loss = 8.095912 | rel_loss_delta = -0.0002806703952231571 | loss_stagnation_count = 0
2026-10-13 15:48:39.921 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 114, smoothed-train-loss = 8.088399 | smoothed-test-loss = 8.109906 | rel_loss_delta = 5.3912422873830165e-05 | loss_stagnation_count = 0
2026-10-13 15:48:42.960 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 115, smoothed-train-loss = 8.088196 | smoothed-test-loss = 8.110846 | rel_loss_delta = -2.5036138567100613e-05 | loss_stagnation_count = 1
2026-10-13 15:48:45.988 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 116, smoothed-train-loss = 8.086621 | smoothed-test-loss = 8.094292 | rel_loss_delta = -0.0001947218935623788 | loss_stagnation_count = 0
2026-10-13 15:48:48.841 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 117, smoothed-train-loss = 8.089079 | smoothed-test-loss = 8.100837 | rel_loss_delta = 0.00030395753860892237 | loss_stagnation_count = 0
2026-10-13 15:48:51.925 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 118, smoothed-train-loss = 8.085846 | smoothed-test-loss = 8.096260 | rel_loss_delta = -0.00039965618429286067 | loss_stagnation_count = 0
2026-10-13 15:48:55.026 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 119, smoothed-train-loss = 8.086240 | smoothed-test-loss = 8.101164 | rel_loss_delta = 4.868404768330841e-05 | loss_stagnation_count = 1
2026-10-13 15:48:58.054 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 120, smoothed-train-loss = 8.086573 | smoothed-test-loss = 8.120669 | rel_loss_delta = 4.119702895654673e-05 | loss_stagnation_count = 2
2026-10-13 15:49:01.049 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 121, smoothed-train-loss = 8.086833 | smoothed-test-loss = 8.092191 | rel_loss_delta = 3.212873386063888e-05 | loss_stagnation_count = 3
2026-10-13 15:49:04.111 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 122, smoothed-train-loss = 8.088162 | smoothed-test-loss = 8.093060 | rel_loss_delta = 0.00016438708649321135 | loss_stagnation_count = 0
2026-10-13 15:49:07.099 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 123, smoothed-train-loss = 8.087968 | smoothed-test-loss = 8.093323 | rel_loss_delta = -2.4091279235799573e-05 | loss_stagnation_count = 1
2026-10-13 15:49:10.100 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 124, smoothed-train-loss = 8.087362 | smoothed-test-loss = 8.091927 | rel_loss_delta = -7.48970400117848e-05 | loss_stagnation_count = 0
2026-10-13 15:49:13.104 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 125, smoothed-train-loss = 8.084309 | smoothed-test-loss = 8.088946 | rel_loss_delta = -0.0003774603713445645 | loss_stagnation_count = 0
2026-10-13 15:49:16.106 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 126, smoothed-train-loss = 8.087216 | smoothed-test-loss = 8.133632 | rel_loss_delta = 0.00035962665509524215 | loss_stagnation_count = 0
2026-10-13 15:49:19.209 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 127, smoothed-train-loss = 8.090377 | smoothed-test-loss = 8.092942 | rel_loss_delta = 0.00039078952111078354 | loss_stagnation_count = 0
2026-10-13 15:49:22.206 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 128, smoothed-train-loss = 8.088973 | smoothed-test-loss = 8.100117 | rel_loss_delta = -0.00017348795304337528 | loss_stagnation_count = 0
2026-10-13 15:49:25.204 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 129, smoothed-train-loss = 8.085931 | smoothed-test-loss = 8.091764 | rel_loss_delta = -0.000376069454869932 | loss_stagnation_count = 0
2026-10-13 15:49:28.391 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 130, smoothed-train-loss = 8.085899 | smoothed-test-loss = 8.079737 | rel_loss_delta = -3.954923039425409e-06 | loss_stagnation_count = 1
2026-10-13 15:49:31.577 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 131, smoothed-train-loss = 8.088654 | smoothed-test-loss = 8.100148 | rel_loss_delta = 0.0003407172078888947 | loss_stagnation_count = 0
2026-10-13 15:49:34.626 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 132, smoothed-train-loss = 8.084877 | smoothed-test-loss = 8.099926 | rel_loss_delta = -0.00046692685216422346 | loss_stagnation_count = 0
2026-10-13 15:49:37.523 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 133, smoothed-train-loss = 8.086769 | smoothed-test-loss = 8.108140 | rel_loss_delta = 0.00023389813781464253 | loss_stagnation_count = 0
2026-10-13 15:49:40.608 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 134, smoothed-train-loss = 8.086495 | smoothed-test-loss = 8.093841 | rel_loss_delta = -3.3877240121609114e-05 | loss_stagnation_count = 1
2026-10-13 15:49:43.628 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 135, smoothed-train-loss = 8.084658 | smoothed-test-loss = 8.105124 | rel_loss_delta = -0.00022716024475055453 | loss_stagnation_count = 0
2026-10-13 15:49:46.687 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 136, smoothed-train-loss = 8.087023 | smoothed-test-loss = 8.089277 | rel_loss_delta = 0.0002925159317644757 | loss_stagnation_count = 0
2026-10-13 15:49:49.642 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 137, smoothed-train-loss = 8.084702 | smoothed-test-loss = 8.098961 | rel_loss_delta = -0.00028696299437337784 | loss_stagnation_count = 0
2026-10-13 15:49:52.679 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 138, smoothed-train-loss = 8.086551 | smoothed-test-loss = 8.128221 | rel_loss_delta = 0.00022877812196956345 | loss_stagnation_count = 0
2026-10-13 15:49:55.802 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 139, smoothed-train-loss = 8.087300 | smoothed-test-loss = 8.085539 | rel_loss_delta = 9.250831782125829e-05 | loss_stagnation_count = 0
2026-10-13 15:49:58.855 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 140, smoothed-train-loss = 8.086627 | smoothed-test-loss = 8.121276 | rel_loss_delta = -8.311200333183272e-05 | loss_stagnation_count = 0
2026-10-13 15:50:01.822 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 141, smoothed-train-loss = 8.089092 | smoothed-test-loss = 8.094528 | rel_loss_delta = 0.0003047411004454236 | loss_stagnation_count = 0
2026-10-13 15:50:04.850 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 142, smoothed-train-loss = 8.085099 | smoothed-test-loss = 8.092211 | rel_loss_delta = -0.0004936401569143691 | loss_stagnation_count = 0
2026-10-13 15:50:07.844 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 143, smoothed-train-loss = 8.085648 | smoothed-test-loss = 8.096153 | rel_loss_delta = 6.79739397594999e-05 | loss_stagnation_count = 0
2026-10-13 15:50:10.801 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 144, smoothed-train-loss = 8.084134 | smoothed-test-loss = 8.095371 | rel_loss_delta = -0.00018720745843632617 | loss_stagnation_count = 0
2026-10-13 15:50:13.766 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 145, smoothed-train-loss = 8.084446 | smoothed-test-loss = 8.112110 | rel_loss_delta = 3.850556157697618e-05 | loss_stagnation_count = 1
2026-10-13 15:50:16.949 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 146, smoothed-train-loss = 8.086226 | smoothed-test-loss = 8.094381 | rel_loss_delta = 0.00022023680851309532 | loss_stagnation_count = 0
2026-10-13 15:50:20.082 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 147, smoothed-train-loss = 8.084686 | smoothed-test-loss = 8.086415 | rel_loss_delta = -0.00019041975015377454 | loss_stagnation_count = 0
2026-10-13 15:50:23.175 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 148, smoothed-train-loss = 8.083628 | smoothed-test-loss = 8.124105 | rel_loss_delta = -0.00013098408515007653 | loss_stagnation_count = 0
2026-10-13 15:50:26.203 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 149, smoothed-train-loss = 8.084782 | smoothed-test-loss = 8.093528 | rel_loss_delta = 0.00014282846635519254 | loss_stagnation_count = 0
2026-10-13 15:50:29.160 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 150, smoothed-train-loss = 8.083493 | smoothed-test-loss = 8.087581 | rel_loss_delta = -0.00015941490233549918 | loss_stagnation_count = 0
2026-10-13 15:50:32.290 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 151, smoothed-train-loss = 8.084583 | smoothed-test-loss = 8.080276 | rel_loss_delta = 0.000134784125534243 | loss_stagnation_count = 0
2026-10-13 15:50:35.306 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 152, smoothed-train-loss = 8.086000 | smoothed-test-loss = 8.094272 | rel_loss_delta = 0.00017535796678780062 | loss_stagnation_count = 0
2026-10-13 15:50:38.274 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 153, smoothed-train-loss = 8.086014 | smoothed-test-loss = 8.089969 | rel_loss_delta = 1.6476081526569804e-06 | loss_stagnation_count = 1
2026-10-13 15:50:41.215 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 154, smoothed-train-loss = 8.085310 | smoothed-test-loss = 8.092272 | rel_loss_delta = -8.698975658218583e-05 | loss_stagnation_count = 0
2026-10-13 15:50:44.195 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 155, smoothed-train-loss = 8.083870 | smoothed-test-loss = 8.093890 | rel_loss_delta = -0.00017810724886366928 | loss_stagnation_count = 0
2026-10-13 15:50:47.289 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 156, smoothed-train-loss = 8.082122 | smoothed-test-loss = 8.115868 | rel_loss_delta = -0.00021630629231222514 | loss_stagnation_count = 0
2026-10-13 15:50:50.253 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 157, smoothed-train-loss = 8.086228 | smoothed-test-loss = 8.084610 | rel_loss_delta = 0.0005080545107889003 | loss_stagnation_count = 0
2026-10-13 15:50:53.282 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 158, smoothed-train-loss = 8.083440 | smoothed-test-loss = 8.081151 | rel_loss_delta = -0.0003447416816700703 | loss_stagnation_count = 0
2026-10-13 15:50:56.274 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 159, smoothed-train-loss = 8.084409 | smoothed-test-loss = 8.108424 | rel_loss_delta = 0.0001198483462786419 | loss_stagnation_count = 0
2026-10-13 15:50:59.256 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 160, smoothed-train-loss = 8.082455 | smoothed-test-loss = 8.117774 | rel_loss_delta = -0.0002417290772681934 | loss_stagnation_count = 0
2026-10-13 15:51:02.242 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 161, smoothed-train-loss = 8.084055 | smoothed-test-loss = 8.091544 | rel_loss_delta = 0.0001980410186242153 | loss_stagnation_count = 0
2026-10-13 15:51:05.334 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 162, smoothed-train-loss = 8.083579 | smoothed-test-loss = 8.099184 | rel_loss_delta = -5.898413831653569e-05 | loss_stagnation_count = 0
2026-10-13 15:51:08.459 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 163, smoothed-train-loss = 8.083844 | smoothed-test-loss = 8.089124 | rel_loss_delta = 3.282128377226118e-05 | loss_stagnation_count = 1
2026-10-13 15:51:11.503 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 164, smoothed-train-loss = 8.084355 | smoothed-test-loss = 8.116905 | rel_loss_delta = 6.321324406350658e-05 | loss_stagnation_count = 0
2026-10-13 15:51:14.420 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 165, smoothed-train-loss = 8.083333 | smoothed-test-loss = 8.085962 | rel_loss_delta = -0.000126441736747098 | loss_stagnation_count = 0
2026-10-13 15:51:17.442 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 166, smoothed-train-loss = 8.081154 | smoothed-test-loss = 8.085489 | rel_loss_delta = -0.00026953487663891693 | loss_stagnation_count = 0
2026-10-13 15:51:20.501 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 167, smoothed-train-loss = 8.081229 | smoothed-test-loss = 8.103741 | rel_loss_delta = 9.233802817758735e-06 | loss_stagnation_count = 1
2026-10-13 15:51:23.467 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 168, smoothed-train-loss = 8.085595 | smoothed-test-loss = 8.095216 | rel_loss_delta = 0.0005403038325987776 | loss_stagnation_count = 0
2026-10-13 15:51:26.341 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 169, smoothed-train-loss = 8.082170 | smoothed-test-loss = 8.088759 | rel_loss_delta = -0.00042359717807519 | loss_stagnation_count = 0
2026-10-13 15:51:29.393 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 170, smoothed-train-loss = 8.084367 | smoothed-test-loss = 8.080509 | rel_loss_delta = 0.00027187068859242387 | loss_stagnation_count = 0
2026-10-13 15:51:32.743 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 171, smoothed-train-loss = 8.081241 | smoothed-test-loss = 8.099289 | rel_loss_delta = -0.00038666234514734394 | loss_stagnation_count = 0
2026-10-13 15:51:35.755 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 172, smoothed-train-loss = 8.084171 | smoothed-test-loss = 8.095638 | rel_loss_delta = 0.0003625767650691003 | loss_stagnation_count = 0
2026-10-13 15:51:38.777 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 173, smoothed-train-loss = 8.080880 | smoothed-test-loss = 8.118226 | rel_loss_delta = -0.0004071259057507397 | loss_stagnation_count = 0
2026-10-13 15:51:41.724 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 174, smoothed-train-loss = 8.084675 | smoothed-test-loss = 8.088787 | rel_loss_delta = 0.00046967507962321597 | loss_stagnation_count = 0
2026-10-13 15:51:44.725 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 175, smoothed-train-loss = 8.084207 | smoothed-test-loss = 8.083026 | rel_loss_delta = -5.792484551107907e-05 | loss_stagnation_count = 0
2026-10-13 15:51:47.748 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 176, smoothed-train-loss = 8.082930 | smoothed-test-loss = 8.112997 | rel_loss_delta = -0.00015802256624758348 | loss_stagnation_count = 0
2026-10-13 15:51:50.598 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 177, smoothed-train-loss = 8.081655 | smoothed-test-loss = 8.098830 | rel_loss_delta = -0.00015764800932603572 | loss_stagnation_count = 0
2026-10-13 15:51:53.576 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 178, smoothed-train-loss = 8.082127 | smoothed-test-loss = 8.083468 | rel_loss_delta = 5.8305295836670155e-05 | loss_stagnation_count = 0
2026-10-13 15:51:56.565 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 179, smoothed-train-loss = 8.080319 | smoothed-test-loss = 8.096936 | rel_loss_delta = -0.000223696969408918 | loss_stagnation_count = 0
2026-10-13 15:51:59.602 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 180, smoothed-train-loss = 8.082780 | smoothed-test-loss = 8.089166 | rel_loss_delta = 0.00030456534034979703 | loss_stagnation_count = 0
2026-10-13 15:52:02.633 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 181, smoothed-train-loss = 8.080545 | smoothed-test-loss = 8.085328 | rel_loss_delta = -0.00027650184476616383 | loss_stagnation_count = 0
2026-10-13 15:52:05.667 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 182, smoothed-train-loss = 8.081278 | smoothed-test-loss = 8.111054 | rel_loss_delta = 9.071327989383029e-05 | loss_stagnation_count = 0
2026-10-13 15:52:08.681 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 183, smoothed-train-loss = 8.081998 | smoothed-test-loss = 8.086897 | rel_loss_delta = 8.917916280064667e-05 | loss_stagnation_count = 0
2026-10-13 15:52:11.732 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 184, smoothed-train-loss = 8.080115 | smoothed-test-loss = 8.107608 | rel_loss_delta = -0.00023300537016447033 | loss_stagnation_count = 0
2026-10-13 15:52:14.657 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 185, smoothed-train-loss = 8.080362 | smoothed-test-loss = 8.089576 | rel_loss_delta = 3.0484604783282615e-05 | loss_stagnation_count = 1
2026-10-13 15:52:17.706 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 186, smoothed-train-loss = 8.082679 | smoothed-test-loss = 8.096297 | rel_loss_delta = 0.0002868037185641836 | loss_stagnation_count = 0
2026-10-13 15:52:20.883 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 187, smoothed-train-loss = 8.082368 | smoothed-test-loss = 8.089274 | rel_loss_delta = -3.850614155530614e-05 | loss_stagnation_count = 1
2026-10-13 15:52:24.672 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 188, smoothed-train-loss = 8.086503 | smoothed-test-loss = 8.079748 | rel_loss_delta = 0.0005115651419469365 | loss_stagnation_count = 0
2026-10-13 15:52:28.284 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 189, smoothed-train-loss = 8.082874 | smoothed-test-loss = 8.093201 | rel_loss_delta = -0.0004487294740147118 | loss_stagnation_count = 0
2026-10-13 15:52:32.014 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 190, smoothed-train-loss = 8.079513 | smoothed-test-loss = 8.087577 | rel_loss_delta = -0.00041578139641423265 | loss_stagnation_count = 0
2026-10-13 15:52:35.514 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 191, smoothed-train-loss = 8.080969 | smoothed-test-loss = 8.090873 | rel_loss_delta = 0.0001802160642273937 | loss_stagnation_count = 0
2026-10-13 15:52:38.978 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 192, smoothed-train-loss = 8.080502 | smoothed-test-loss = 8.107825 | rel_loss_delta = -5.7772594289280833e-05 | loss_stagnation_count = 0
2026-10-13 15:52:42.636 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 193, smoothed-train-loss = 8.081278 | smoothed-test-loss = 8.077938 | rel_loss_delta = 9.593181881919597e-05 | loss_stagnation_count = 0
2026-10-13 15:52:46.893 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 194, smoothed-train-loss = 8.081005 | smoothed-test-loss = 8.078059 | rel_loss_delta = -3.3766560969297484e-05 | loss_stagnation_count = 1
2026-10-13 15:52:50.190 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 195, smoothed-train-loss = 8.081999 | smoothed-test-loss = 8.087713 | rel_loss_delta = 0.0001230785462522363 | loss_stagnation_count = 0
2026-10-13 15:52:53.240 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 196, smoothed-train-loss = 8.082086 | smoothed-test-loss = 8.109593 | rel_loss_delta = 1.0706460902373247e-05 | loss_stagnation_count = 1
2026-10-13 15:52:56.218 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 197, smoothed-train-loss = 8.082895 | smoothed-test-loss = 8.085847 | rel_loss_delta = 0.00010011794082967163 | loss_stagnation_count = 0
2026-10-13 15:52:59.419 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 198, smoothed-train-loss = 8.081048 | smoothed-test-loss = 8.096204 | rel_loss_delta = -0.0002284852827461179 | loss_stagnation_count = 0
2026-10-13 15:53:02.804 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 199, smoothed-train-loss = 8.081937 | smoothed-test-loss = 8.096515 | rel_loss_delta = 0.00011004557792430029 | loss_stagnation_count = 0
2026-10-13 15:53:06.109 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:118 - @epoch = 200, smoothed-train-loss = 8.080594 | smoothed-test-loss = 8.111398 | rel_loss_delta = -0.00016625007286013234 | loss_stagnation_count = 0
2026-10-13 15:53:06.114 | INFO     | ttflow.training.nn.nn_trainer_utils:train_nn:128 - MSELoss(test) after training:8.113051414489746
2026-10-13 15:53:06.115 | INFO     | __main__:<module>:288 - Given num_reflow_iterations = 0 and current reflow_iteration = 0, no further reflows are needed, exiting the reflow loop
2026-10-13 15:53:06.136 | INFO     | __main__:<module>:346 - Output model artifact saved to = models/experimental/vt_dataset_gaussian_mixture_2d_data_params_num_components_4_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-10-13T15:42:51.962203.pkl
2026-10-13 15:53:06.136 | INFO     | __main__:<module>:347 - vt training process finished successfully.
```
# Inference

```log
2026-10-13 21:49:46.449 | INFO     | __main__:<module>:202 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: 0.0257 | Std: 0.0093 |Inference time for total number of inference samples: 21.59822 seconds
2026-10-13 22:07:41.153 | INFO     | __main__:<module>:202 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: 0.0255 | Std: 0.0075 |Inference time for total number of inference samples: 21.453684 seconds
2026-10-13 22:08:38.405 | INFO     | __main__:<module>:202 - Sinkhorn(x1_hat, x1) summary | Iterations: 10 | Inference samples: 200000 | Subsamples: 50000 | Mean: 0.0319 | Std: 0.0091 |Inference time for total number of inference samples: 21.764509 seconds
```