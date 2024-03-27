# from datasets import load_dataset
import csv
import time
from neu_img_finder import neu_find
from histogram_img_finder import hist_find

if __name__ == '__main__':
    start = time.time()
    with open('result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'video_pred', 'minutage_pred', "evaluation"]) # Evaluation is TP, FP, TN, FN
        hist_find(writer)
    print(f"Hist Execution time: {time.time() - start} seconds")


    start = time.time()
    with open('result_neu.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'video_pred', 'minutage_pred', "evaluation"]) # Evaluation is TP, FP, TN, FN
        neu_find(writer)
    print(f"Neural Execution time: {time.time() - start} seconds")

#  Bin evaluation               F1 score eval
#  Bin = 256 -> 70.1% TP
#  Bin = 85 -> 73.9% TP
#  Bin = 4 -> 76.3% TP
#  Bin = 8 -> 78.3% TP
    

#  Bin = 12 -> 78.5% TP    --> 0.9 : 85%
    


#  Bin = 16 -> 77.9% TP
#  Bin = 32 -> 76.4% TP
#  Bin = 64 -> 
    
# Goal: 80.5% 