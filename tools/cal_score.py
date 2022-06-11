
runtime = 151
si_rmse = 0.37


runtime = 102
si_rmse = 0.39

score = 640000 / (2**(20*si_rmse)*runtime)
print(score)