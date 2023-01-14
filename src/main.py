import utils
import classification
import clustering
import suggestions
import knn_distance

def main():
	print("Preprocessing initialized...")
	utils.generate_normalized_dataset()
	utils.generate_preprocessed_dataset()
	print("Preprocessing complete!")

	preprocessing_parameters = utils.get_improvable_features()

	classification.classify_for_best_model()
	clusters = clustering.generate_clusters()

	exitted_employee = 1420391

	# find at risk employees based on given employee id
	at_risk_employees = knn_distance.KNN_distance(utils.get_preprocessed_dataset(), exitted_employee)
	print("\nAt risk employees: ", at_risk_employees)

	# provide suggestions for at risk employees
	updated_parameters = suggestions.give_suggestions(at_risk_employees, clusters, preprocessing_parameters)

	# validating after updates
	suggestions.validate_suggestions(at_risk_employees, updated_parameters, exitted_employee)

if __name__ == '__main__':
	main()