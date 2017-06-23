# Importing numpy to calculate mean, standard deviation, get arrays from numpy array i.e. matrix columns
import numpy as np
# importing csv module to read the given input csv file
import csv
# Opening file in read mode
input_csv=open("input2.csv","r")
# Getting the input csv text content
input_csv_text=csv.reader(input_csv)
# Converting csv object into list
input_data = list(input_csv_text)
# Given alpha values and +1 free value i.e. 0.8
alpha_vals =  [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.8]
# Converting inpur list of lists to numpy array
input_data_arr = np.array(input_data)
# Retrieving label - y height
y_height = input_data_arr[:,2]
# Converting string to float using list comprehensions
y_height = [float(el) for el in y_height]
# Retrieving feature - x age
x_age = input_data_arr[:,0]
x_age = [float(el) for el in x_age]
# Calculating mean and standard deviataion to get the scaled values of features
x_age_mean = np.mean(x_age)
x_age_std = np.std(x_age)
# Calculating scaled values
x_age_scaled = [(el - x_age_mean)/x_age_std for el in x_age]
# Performing same as above for feature - x weight
x_weight = input_data_arr[:,1]
x_weight = [float(el) for el in x_weight]
x_weight_mean = np.mean(x_weight)
x_weight_std = np.std(x_weight)
x_weight_scaled = [(el - x_weight_mean)/x_weight_std for el in x_weight]
print_list = []
# Looping through alpha vals 
for alpha in alpha_vals:
    # initializing intercept b_0, other weights to 0
    b_0, b_age, b_weight = 0, 0, 0
    # Iterating for 100 times
    for _ in range(100):
        # Adjusting values of weights as per the formula.
        common_list = [(b_0 + b_age *x_a + b_weight * x_w - y_h)  for x_a, x_w, y_h in zip(x_age_scaled, x_weight_scaled, y_height)]
        b_0 += -(alpha / len(x_weight_scaled)) * sum(common_list)
        b_age += -(alpha / len(x_weight_scaled)) * sum([cel*x_a for cel, x_a in zip(common_list, x_age_scaled)])
        b_weight += -(alpha / len(x_weight_scaled)) * sum([cel*x_w for cel, x_w in zip(common_list, x_weight_scaled)])
    # Adding weights to print list
    print_list.append([alpha, 100, b_0, b_age, b_weight])
input_csv.close()
# Writing the weights to output csv file.
output_csv=open("output2.csv","w")
for el in print_list:
    output_csv.write(",".join([str(ind) for ind in el]))
    output_csv.write("\n")
output_csv.close()