# importing csv module to read the given input csv file
import csv
# Opening file in read mode
input_csv=open("input1.csv","r")
# Getting the input csv text content
input_csv_text=csv.reader(input_csv)
# Converting csv object into list
# and taking into new variable since we have to loop on the data till the convergence
input_data = list(input_csv_text)
# Initializing weights(wj) and bias(w0) to 0
w1 , w2 , b = 0, 0, 0
# Maintaining weights list to check the convergence
weights_list = [[w1, w2, b]]
# Running infinite loop
while True:
    # Taking copy of input data to process it
    input_to_process = input_data[:]
    # Lopping through each sample of data
    for row in input_to_process:
        # Calculating fxi i.e sigsum of wjxij where j -> 0 to d dimenstions, xij is ith sample at dimension d
        fxi = b + w1 * int(row[0]) + w2 * int(row[1])
        # As per perceptron algorithm the classification as 1 when sigsum value > 0 else classified as -1
        fxi = 1 if fxi > 0 else -1
        # Checking whether the classification is right or wrong.
        # if there is wrong classification, i.e. an error, need to adjust the weights
        # wj := wj + yi*xi
        if int(row[2]) * fxi <= 0:
            # Adjusting the weights
            b += int(row[2])
            w1 += int(row[0]) * int(row[2])
            w2 += int(row[1]) * int(row[2])
    weights_list.append([w1, w2, b])
    # checking for convergence i.e. no change in weights, breaking the loop
    if weights_list[-1] == weights_list[-2]:
        break
input_csv.close()
# Writing the weights to output csv file.
output_csv=open("output1.csv","w")
for el in weights_list:
    output_csv.write(",".join([str(ind) for ind in el]))
    output_csv.write("\n")
output_csv.close()