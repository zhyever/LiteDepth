
runtime = 0.07
# si_rmse = 0.328
# si_rmse = 0.320
si_rmse = 0.30
# 0.0015140673297407044
# 0.00169164795369464

# runtime = 0.105
# si_rmse = 0.328
# si_rmse = 0.292
# 0.0015140673297407044
# 0.00169164795369464

score = 0.01 / (2**(20*si_rmse)*runtime)
print(score)