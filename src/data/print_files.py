import os
import pickle

files_with_acc = []
for i in range(35):
    file, analysis_evaluator = pickle.load(open(f"./outputs/analysis/{i}.pkl", "rb"))
    acc = analysis_evaluator.sum_correct_foot_classifications / analysis_evaluator.sum_timesteps if analysis_evaluator.sum_timesteps else -1
    files_with_acc.append((file, acc))


sorted_files = sorted(files_with_acc, key=lambda x: x[1])
print(sorted_files)