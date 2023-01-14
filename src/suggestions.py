import pandas as pd
import knn_distance
import utils

data_file = "./data/preprocessed_data.csv"

improvable_columns = {
	"DailyRate": {"column_index": 3, "rounding": 0, "greater_is_better": True},
	"DistanceFromHome": {"column_index": 5, "rounding": 2, "greater_is_better": False},
	"HourlyRate": {"column_index": 10, "rounding": 0, "greater_is_better": True},
	"MonthlyIncome": {"column_index": 16, "rounding": 0, "greater_is_better": True},
	"MonthlyRate": {"column_index": 17, "rounding": 0, "greater_is_better": True},
	"PercentSalaryHike": {"column_index": 20, "rounding": 2, "greater_is_better": True},
	"TrainingTimesLastYear": {"column_index": 24, "rounding": 0, "greater_is_better": True},
	"YearsInCurrentRole": {"column_index": 27, "rounding": 0, "greater_is_better": False},
	"YearsSinceLastPromotion": {"column_index": 28, "rounding": 0, "greater_is_better": False},
	"YearsWithCurrManager": {"column_index": 29, "rounding": 0, "greater_is_better": False}
}

categorical_columns_index = [2, 4, 7, 9, 13, 15, 19, 6, 12, 22]

def reverse_preprocessing(current_value, parameter, rounding_factor):
	value = round(current_value * parameter["stdev"] + parameter["mean"], rounding_factor)
	return int(value) if rounding_factor == 0 else value

def give_suggestions(at_risk_employees, cluster_centroids, preprocessing_parameters):
	# get data for at risk employees
	original_data = pd.read_csv(data_file)
	at_risk_employee_ids = at_risk_employees.keys()
	at_risk_employee_data = original_data[original_data["EmployeeID"].isin(at_risk_employee_ids)]
	at_risk_employee_data.drop(["Attrition"], axis=1, inplace=True)
	
	# iterate for each required column
	updated_values = {}
	for _, employee in at_risk_employee_data.iterrows():
		current_employee_id = employee["EmployeeID"]
		updated_values[current_employee_id] = {}
		print("\nSuggestions to retain employee ", int(current_employee_id))

		for column_name, column_config in improvable_columns.items():
			
			index_position = column_config["column_index"]
			rounding_factor = column_config["rounding"]
			greater_is_better = column_config["greater_is_better"]

			# finding nearest cluster
			min_cluster = -1
			min_distance = float("inf")

			for cluster_number in range(len(cluster_centroids)):
				distance = 0
				centroid = cluster_centroids[cluster_number]
				for column_number in range(29):
					if column_number != index_position-1 and column_number+1 not in categorical_columns_index:
						distance += (centroid[column_number] - employee[column_number+1])**2
					elif column_number != index_position-1 and column_number+1 in categorical_columns_index:
						if centroid[column_number] != employee[column_number+1]:
							distance += 1
				if distance < min_distance:
					min_distance = distance
					min_cluster = cluster_number

			final_centroid = cluster_centroids[min_cluster]
			if (greater_is_better and final_centroid[index_position-1] > employee[index_position]) or (not(greater_is_better) and final_centroid[index_position-1] < employee[index_position]):
				old_value = employee[index_position]
				ideal_value = final_centroid[index_position-1]
				actual_old_value = reverse_preprocessing(old_value, preprocessing_parameters[column_name], rounding_factor)
				actual_ideal_value = reverse_preprocessing(ideal_value, preprocessing_parameters[column_name], rounding_factor)
				updated_values[current_employee_id][column_name] = ideal_value
				print("Change ", column_name, " from ", actual_old_value, " to ", actual_ideal_value)

	return updated_values

def validate_suggestions(at_risk_employees, updated_parameters, exitted_employee):
	print("\nValidating Suggestions")
	print("Old at-risk employees: ", at_risk_employees)
	value = utils.get_preprocessed_dataset()
	# value = original_data[original_data["EmployeeID"].isin([1709918, 1355585])]

	for employee in at_risk_employees.keys():
		for column, new_value in updated_parameters[employee].items():
			value.loc[value.EmployeeID == employee, column] = new_value

	# value.to_csv("data/suggestion_updated_data.csv", encoding='utf-8', index=False)

	# finding max distance of current at risk employees
	max_distance = max(at_risk_employees.values())

	print("Distance threshold: ", max_distance)
	new_at_risk_employees = knn_distance.KNN_distance(value, exitted_employee, max_distance)
	print("New at-risk employees: ",  new_at_risk_employees)
