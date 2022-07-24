
runtime = 0.051
# si_rmse = 0.27 # 0.0050389513514308435
si_rmse = 0.303 # 0.003324468085106383

# runtime = 0.049
# si_rmse = 0.309 # 0.002814741697355405

score = 0.01 / (2**(20*si_rmse)*runtime)
print(score)


