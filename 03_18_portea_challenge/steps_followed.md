# Steps Followed

## Training

* Loaded the data into dataframes
* Removed columns
	* less data % compared to total records
	* dependent variables. Ex: service_id vs service_name
* Renaming columns to have a unique name to help in performing merge (join)
* Merge all the dataframes into single dataframe, by following left side data should be more than right side data.
* Finding the missing values count across the columns
* Replace the missing values with 0.(correct this - may be removing the rows does help?)
* Milestone - Saved the all merged data to CSV file.
* Converting all the object type data to string for perfroming encoding.
* Encode all the string datatype columns data categorical values to numerical values
* Keep the encoders, which needed for converting test data
* Milestone - Saved the processed, cleaned data to CSV file.
* Drop duplicates if any
* Removed patient_ids from data, separated out data to input(X_train) and output variables(Y_train, Y_train_R)
	* Y_train - this is the output variable for classifier.
	* Y_train_R - this is the output variable for Regressor.
* Converting the dataframes to matrices (numpy arrays)
* Running the classifer and Regressor
* Milestone - Saving the classifer, regressor, encoders to pickle files.

## Validating

* Load the processed, cleaned training file
* Get sample(tail) data from the whole dataset for validating
* Have the original values of patient_ids vs Bucket vs Revenue
* Separate out Bucket, Revenue column values to separate dataframes
* Load the classifer we trained above during training
* Run the predictions and capture it with classifier, compare values of original vs the predicted.
* Load the Regressor we trained above during training
* Run the predictions and capture it with Regressor, compare values of original vs the predicted.

## Testing

The steps are more or less similar to combined of training and validation

TODO: Pending documentation.