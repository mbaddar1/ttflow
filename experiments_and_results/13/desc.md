Data

MW totdim 8 manywell dim 8 

file


data-snapshots/experimental/data_snapshot_many_well_data_dim_8_many_well_dim_8_n_train_500000_n_test_200000_n_eval_200000_timestamp_2026-11-18T20:31:13.605688.pkl


Model

Arch 
RegNet 

files 

1-RecFlow

models/experimental/vt_dataset_many_well_data_params_many_well_total_data_dim_8_many_well_dim_8_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-11-18T20:33:08.975763.pkl

2-RecFlow

models/experimental/vt_dataset_many_well_data_params_many_well_total_data_dim_8_many_well_dim_8_trajectory_model_global_func_model_nn_model_hyperparams_hidden_dim_128_depth_2_opt_adam_nn_hidden_dim_128_timestamp_2026-11-18T20:56:55.175022.pkl

Inference 

1 RF

euler 



==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : euler
  VT Trajectory Model  : global
  Num Reflows          : 0
  Inference Samples    : 200,000
  Inference Time       : 6.77s
  Evaluations (t)      : 1000
  Eval Results         : {0.01: {'real_generated': {'mean': 0.42408815026283264, 'stddev': 0.021673431023289204}, 'real_real': {'mean': 0.3305719196796417, 'stddev': 0.00837827849173997}, 'generated_generated': {'mean': 0.3721270263195038, 'stddev': 0.008465477077695984}}, 0.02: {'real_generated': {'mean': 0.42653924226760864, 'stddev': 0.028429645570041386}, 'real_real': {'mean': 0.3442757248878479, 'stddev': 0.013723093257208998}, 'generated_generated': {'mean': 0.36820383071899415, 'stddev': 0.004216749902731718}}, 0.05: {'real_generated': {'mean': 0.42463473677635194, 'stddev': 0.02911661043983621}, 'real_real': {'mean': 0.3359416961669922, 'stddev': 0.0051278770728755115}, 'generated_generated': {'mean': 0.36875003576278687, 'stddev': 0.007451023995586956}}, 0.06582111120223999: {'real_generated': {'mean': 0.4174959182739258, 'stddev': 0.014860340227784513}, 'real_real': {'mean': 0.3356427788734436, 'stddev': 0.0172072444827476}, 'generated_generated': {'mean': 0.36991791129112245, 'stddev': 0.006222086857314009}}, 0.1: {'real_generated': {'mean': 0.41361294984817504, 'stddev': 0.017749139085172814}, 'real_real': {'mean': 0.336428564786911, 'stddev': 0.01076345994042909}, 'generated_generated': {'mean': 0.3668728172779083, 'stddev': 0.011413963993995717}}, 0.16455277800559998: {'real_generated': {'mean': 0.405450040102005, 'stddev': 0.0149903521888442}, 'real_real': {'mean': 0.3142800807952881, 'stddev': 0.006996620972445781}, 'generated_generated': {'mean': 0.35174904465675355, 'stddev': 0.005911581951187253}}, 0.2: {'real_generated': {'mean': 0.3879145920276642, 'stddev': 0.018592587836579265}, 'real_real': {'mean': 0.2986115634441376, 'stddev': 0.010408740602432147}, 'generated_generated': {'mean': 0.33825171589851377, 'stddev': 0.007065745619223288}}, 0.32910555601119995: {'real_generated': {'mean': 0.29791214466094973, 'stddev': 0.010706265224695985}, 'real_real': {'mean': 0.22597254812717438, 'stddev': 0.013496042429767312}, 'generated_generated': {'mean': 0.25796470046043396, 'stddev': 0.0018770100453575109}}, 0.5: {'real_generated': {'mean': 0.21752161681652069, 'stddev': 0.015225670847597635}, 'real_real': {'mean': 0.147263565659523, 'stddev': 0.010667496095145673}, 'generated_generated': {'mean': 0.16540301740169525, 'stddev': 0.005186307017842842}}, 0.6582111120223999: {'real_generated': {'mean': 0.15660589933395386, 'stddev': 0.01627413804287158}, 'real_real': {'mean': 0.10945306122303008, 'stddev': 0.009587289261733582}, 'generated_generated': {'mean': 0.11049723327159881, 'stddev': 0.005309454853528573}}, 1.0: {'real_generated': {'mean': 0.07992027401924133, 'stddev': 0.006124370370220852}, 'real_real': {'mean': 0.05664513781666756, 'stddev': 0.006686509615038437}, 'generated_generated': {'mean': 0.04766307100653648, 'stddev': 0.0031213401512384478}}, 1.3164222240447998: {'real_generated': {'mean': 0.03361270427703857, 'stddev': 0.003031669391178198}, 'real_real': {'mean': 0.02507961131632328, 'stddev': 0.0020752117973032733}, 'generated_generated': {'mean': 0.02295493707060814, 'stddev': 0.0026898583026770903}}, 2.0: {'real_generated': {'mean': 0.010704659670591355, 'stddev': 0.0029050889347410585}, 'real_real': {'mean': 0.006285845395177602, 'stddev': 0.001533004129979235}, 'generated_generated': {'mean': 0.005041961371898651, 'stddev': 0.0009954011205338378}}, 5.0: {'real_generated': {'mean': 0.0031380340456962585, 'stddev': 0.0008986116077707558}, 'real_real': {'mean': 0.0013287192210555076, 'stddev': 0.0007649449893336786}, 'generated_generated': {'mean': 0.0010804824531078338, 'stddev': 0.00017148538258017615}}, 10.0: {'real_generated': {'mean': 0.004318474978208542, 'stddev': 0.0014561136709857246}, 'real_real': {'mean': 0.0013258866500109434, 'stddev': 0.0005259107336364202}, 'generated_generated': {'mean': 0.0010058932006359101, 'stddev': 0.0003557877321418984}}}
==================================================
2026-11-19 16:20:03.447 | INFO     | __main__:<module>:356 - Staring visualization...
2026-11-19 16:20:03.447 | INFO     | __main__:<module>:362 - Visualizing many_well
2026-11-19 16:20:03.447 | WARNING  | __main__:<module>:369 - Setting plot range for x1 and x1_hat to be None and auto-detected by plot method. Set it manually via experimentation
2026-11-19 16:20:03.447 | INFO     | __main__:<module>:371 - Visualizing many_well and saving figure to figures/inference/experimental/inference_many_well_model_arch_VelocityFieldRegressionNet_data_params_many_well_total_data_dim_8_many_well_dim_8_timestamp_2026-11-19T16:11:28.657796_trajectory_model_global_ode_solver_method_euler_num_reflows_0.png...
2026-11-19 16:20:09.632 | INFO     | __main__:<module>:386 - Inference figure saved in: figures/inference/experimental/inference_many_well_model_arch_VelocityFieldRegressionNet_data_params_many_well_total_data_dim_8_many_well_dim_8_timestamp_2026-11-19T16:11:28.657796_trajectory_model_global_ode_solver_method_euler_num_reflows_0.png
2026-11-19 16:20:09.632 | INFO     | __main__:<module>:387 - Inference and evaluation process finished successfully!

Process finished with exit code 139 (interrupted by signal 11:SIGSEGV)


rk4 

2026-11-19 15:17:02.844 | INFO     | __main__:<module>:329 - 
==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : rk4
  VT Trajectory Model  : global
  Num Reflows          : 0
  Inference Samples    : 200,000
  Inference Time       : 24.88s
  Evaluations (t)      : 1000
  Eval Results         : {0.01: {'real_generated': {'mean': 0.4245616316795349, 'stddev': 0.019997442295702324}, 'real_real': {'mean': 0.3305719196796417, 'stddev': 0.00837827849173997}, 'generated_generated': {'mean': 0.3714672148227692, 'stddev': 0.011077132722066835}}, 0.02: {'real_generated': {'mean': 0.4259879529476166, 'stddev': 0.02670318170804522}, 'real_real': {'mean': 0.3442757248878479, 'stddev': 0.013723093257208998}, 'generated_generated': {'mean': 0.36853315234184264, 'stddev': 0.006414635412019426}}, 0.05: {'real_generated': {'mean': 0.4213906407356262, 'stddev': 0.04155111238012517}, 'real_real': {'mean': 0.3359416961669922, 'stddev': 0.0051278770728755115}, 'generated_generated': {'mean': 0.36119561195373534, 'stddev': 0.005168147355409668}}, 0.06582111120223999: {'real_generated': {'mean': 0.4114681541919708, 'stddev': 0.022585678810223307}, 'real_real': {'mean': 0.3356427788734436, 'stddev': 0.0172072444827476}, 'generated_generated': {'mean': 0.36527512669563295, 'stddev': 0.003036683805025536}}, 0.1: {'real_generated': {'mean': 0.39962305426597594, 'stddev': 0.012083660828501515}, 'real_real': {'mean': 0.336428564786911, 'stddev': 0.01076345994042909}, 'generated_generated': {'mean': 0.3579124867916107, 'stddev': 0.005200847205387293}}, 0.16455277800559998: {'real_generated': {'mean': 0.39743772149086, 'stddev': 0.02105596081310649}, 'real_real': {'mean': 0.3142800807952881, 'stddev': 0.006996620972445781}, 'generated_generated': {'mean': 0.3485938489437103, 'stddev': 0.006878948699994825}}, 0.2: {'real_generated': {'mean': 0.38507079482078554, 'stddev': 0.011765789160791006}, 'real_real': {'mean': 0.2986115634441376, 'stddev': 0.010408740602432147}, 'generated_generated': {'mean': 0.33100942969322206, 'stddev': 0.005701680982684405}}, 0.32910555601119995: {'real_generated': {'mean': 0.29367470145225527, 'stddev': 0.010813175100544865}, 'real_real': {'mean': 0.22597254812717438, 'stddev': 0.013496042429767312}, 'generated_generated': {'mean': 0.25607859492301943, 'stddev': 0.0057317519585190135}}, 0.5: {'real_generated': {'mean': 0.22343556880950927, 'stddev': 0.02225169364285069}, 'real_real': {'mean': 0.147263565659523, 'stddev': 0.010667496095145673}, 'generated_generated': {'mean': 0.16455971598625183, 'stddev': 0.004784740276091808}}, 0.6582111120223999: {'real_generated': {'mean': 0.1550311863422394, 'stddev': 0.00830278445531527}, 'real_real': {'mean': 0.10945306122303008, 'stddev': 0.009587289261733582}, 'generated_generated': {'mean': 0.11156771779060363, 'stddev': 0.0038821251687191833}}, 1.0: {'real_generated': {'mean': 0.07490844577550888, 'stddev': 0.008761795729809858}, 'real_real': {'mean': 0.05664513781666756, 'stddev': 0.006686509615038437}, 'generated_generated': {'mean': 0.052397508174180984, 'stddev': 0.005584609484808267}}, 1.3164222240447998: {'real_generated': {'mean': 0.034490635991096495, 'stddev': 0.006246184159520471}, 'real_real': {'mean': 0.02507961131632328, 'stddev': 0.0020752117973032733}, 'generated_generated': {'mean': 0.021871818602085112, 'stddev': 0.0015313775394880037}}, 2.0: {'real_generated': {'mean': 0.01064971201121807, 'stddev': 0.0017110719723992679}, 'real_real': {'mean': 0.006285845395177602, 'stddev': 0.001533004129979235}, 'generated_generated': {'mean': 0.005013356171548367, 'stddev': 0.000568767051887922}}, 5.0: {'real_generated': {'mean': 0.0029595703817903996, 'stddev': 0.0006498624260593694}, 'real_real': {'mean': 0.0013287192210555076, 'stddev': 0.0007649449893336786}, 'generated_generated': {'mean': 0.001258153491653502, 'stddev': 0.00041433883222210844}}, 10.0: {'real_generated': {'mean': 0.0031596817076206206, 'stddev': 0.0006338102194121107}, 'real_real': {'mean': 0.0013258866500109434, 'stddev': 0.0005259107336364202}, 'generated_generated': {'mean': 0.0010562930256128311, 'stddev': 0.00043276598202485677}}}
==================================================
2026-11-19 15:17:02.844 | INFO     | __main__:<module>:356 - Staring visualization...
2026-11-19 15:17:02.844 | INFO     | __main__:<module>:362 - Visualizing many_well
2026-11-19 15:17:02.844 | WARNING  | __main__:<module>:369 - Setting plot range for x1 and x1_hat to be None and auto-detected by plot method. Set it manually via experimentation
2026-11-19 15:17:02.844 | INFO     | __main__:<module>:371 - Visualizing many_well and saving figure to figures/inference/experimental/inference_many_well_model_arch_VelocityFieldRegressionNet_data_params_many_well_total_data_dim_8_many_well_dim_8_timestamp_2026-11-19T15:08:04.736096_trajectory_model_global_ode_solver_method_rk4_num_reflows_0.png...
2026-11-19 15:17:09.112 | INFO     | __main__:<module>:386 - Inference figure saved in: figures/inference/experimental/inference_many_well_model_arch_VelocityFieldRegressionNet_data_params_many_well_total_data_dim_8_many_well_dim_8_timestamp_2026-11-19T15:08:04.736096_trajectory_model_global_ode_solver_method_rk4_num_reflows_0.png
2026-11-19 15:17:09.112 | INFO     | __main__:<module>:387 - Inference and evaluation process finished successfully!

Process finished with exit code 139 (interrupted by signal 11:SIGSEGV)

dopri8

==================================================
             Sinkhorn Results Summary             
==================================================
  ODE Solver Method    : dopri8
  VT Trajectory Model  : global
  Num Reflows          : 0
  Inference Samples    : 200,000
  Inference Time       : 60.79s
  Evaluations (t)      : 1000
  Eval Results         : {0.01: {'real_generated': {'mean': 0.4319257438182831, 'stddev': 0.034747163946127216}, 'real_real': {'mean': 0.3305719196796417, 'stddev': 0.00837827849173997}, 'generated_generated': {'mean': 0.36791062355041504, 'stddev': 0.004818962000177965}}, 0.02: {'real_generated': {'mean': 0.4386791944503784, 'stddev': 0.020220745238866567}, 'real_real': {'mean': 0.3442757248878479, 'stddev': 0.013723093257208998}, 'generated_generated': {'mean': 0.3686182737350464, 'stddev': 0.006336664471527963}}, 0.05: {'real_generated': {'mean': 0.43254849314689636, 'stddev': 0.02271663663322092}, 'real_real': {'mean': 0.3359416961669922, 'stddev': 0.0051278770728755115}, 'generated_generated': {'mean': 0.3703997373580933, 'stddev': 0.005487809854895947}}, 0.06582111120223999: {'real_generated': {'mean': 0.4063165307044983, 'stddev': 0.025581636555647286}, 'real_real': {'mean': 0.3356427788734436, 'stddev': 0.0172072444827476}, 'generated_generated': {'mean': 0.3618675351142883, 'stddev': 0.006236911870136231}}, 0.1: {'real_generated': {'mean': 0.41211929321289065, 'stddev': 0.01830005650166623}, 'real_real': {'mean': 0.336428564786911, 'stddev': 0.01076345994042909}, 'generated_generated': {'mean': 0.3620638072490692, 'stddev': 0.008441708888521464}}, 0.16455277800559998: {'real_generated': {'mean': 0.400242418050766, 'stddev': 0.011656559381827658}, 'real_real': {'mean': 0.3142800807952881, 'stddev': 0.006996620972445781}, 'generated_generated': {'mean': 0.35152947902679443, 'stddev': 0.0043279650396217135}}, 0.2: {'real_generated': {'mean': 0.39011451601982117, 'stddev': 0.01884693499450136}, 'real_real': {'mean': 0.2986115634441376, 'stddev': 0.010408740602432147}, 'generated_generated': {'mean': 0.33686813712120056, 'stddev': 0.006144008243810487}}, 0.32910555601119995: {'real_generated': {'mean': 0.3151640653610229, 'stddev': 0.015500323553252801}, 'real_real': {'mean': 0.22597254812717438, 'stddev': 0.013496042429767312}, 'generated_generated': {'mean': 0.2634911000728607, 'stddev': 0.00587226001779301}}, 0.5: {'real_generated': {'mean': 0.22272399961948394, 'stddev': 0.014705531418029879}, 'real_real': {'mean': 0.147263565659523, 'stddev': 0.010667496095145673}, 'generated_generated': {'mean': 0.17156751453876495, 'stddev': 0.0038077496704177777}}, 0.6582111120223999: {'real_generated': {'mean': 0.15876861661672592, 'stddev': 0.032791986387460115}, 'real_real': {'mean': 0.10945306122303008, 'stddev': 0.009587289261733582}, 'generated_generated': {'mean': 0.1114700898528099, 'stddev': 0.0026649904182601833}}, 1.0: {'real_generated': {'mean': 0.08219045996665955, 'stddev': 0.01149584761183975}, 'real_real': {'mean': 0.05664513781666756, 'stddev': 0.006686509615038437}, 'generated_generated': {'mean': 0.05481864735484123, 'stddev': 0.0032095130671700852}}, 1.3164222240447998: {'real_generated': {'mean': 0.035352347791194914, 'stddev': 0.005177865657409304}, 'real_real': {'mean': 0.02507961131632328, 'stddev': 0.0020752117973032733}, 'generated_generated': {'mean': 0.02430693432688713, 'stddev': 0.002228404955519747}}, 2.0: {'real_generated': {'mean': 0.01015339493751526, 'stddev': 0.0018324904662044789}, 'real_real': {'mean': 0.006285845395177602, 'stddev': 0.001533004129979235}, 'generated_generated': {'mean': 0.006218974944204092, 'stddev': 0.0008356024200716634}}, 5.0: {'real_generated': {'mean': 0.003316325694322586, 'stddev': 0.0006847598591291257}, 'real_real': {'mean': 0.0013287192210555076, 'stddev': 0.0007649449893336786}, 'generated_generated': {'mean': 0.0014129254966974258, 'stddev': 0.0003141298553726828}}, 10.0: {'real_generated': {'mean': 0.0037813551723957064, 'stddev': 0.0019218709470651258}, 'real_real': {'mean': 0.0013258866500109434, 'stddev': 0.0005259107336364202}, 'generated_generated': {'mean': 0.0006717165466398001, 'stddev': 0.00029034067964839403}}}
==================================================
2026-11-19 17:25:25.510 | INFO     | __main__:<module>:356 - Staring visualization...
2026-11-19 17:25:25.510 | INFO     | __main__:<module>:362 - Visualizing many_well
2026-11-19 17:25:25.510 | WARNING  | __main__:<module>:369 - Setting plot range for x1 and x1_hat to be None and auto-detected by plot method. Set it manually via experimentation
2026-11-19 17:25:25.510 | INFO     | __main__:<module>:371 - Visualizing many_well and saving figure to figures/inference/experimental/inference_many_well_model_arch_VelocityFieldRegressionNet_data_params_many_well_total_data_dim_8_many_well_dim_8_timestamp_2026-11-19T17:15:55.496969_trajectory_model_global_ode_solver_method_dopri8_num_reflows_0.png...
2026-11-19 17:25:31.788 | INFO     | __main__:<module>:386 - Inference figure saved in: figures/inference/experimental/inference_many_well_model_arch_VelocityFieldRegressionNet_data_params_many_well_total_data_dim_8_many_well_dim_8_timestamp_2026-11-19T17:15:55.496969_trajectory_model_global_ode_solver_method_dopri8_num_reflows_0.png
2026-11-19 17:25:31.788 | INFO     | __main__:<module>:387 - Inference and evaluation process finished successfully!

Process finished with exit code 139 (interrupted by signal 11:SIGSEGV)