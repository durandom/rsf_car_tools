To build a robust polynomial regression model for predicting FFB sensitivity, it’s crucial to carefully select a representative subset of cars from your dataset for manual tuning. Here’s how to determine the minimum number of cars and the criteria for selection:

1. Minimum Number of Cars

The minimum sample size depends on:
	•	Number of Features: You have at least three features (weight, drivetrain, lock-to-lock rotation). A polynomial regression with degree 2 or 3 will introduce more terms (e.g., squares, interactions), so you’ll need a larger sample.
	•	General Rule:
	•	For each polynomial term, ensure at least 10x the number of unique cars. For example:
	•	Degree 2 polynomial: ~6 terms (weight, drivetrain, rotation, weight², weight × rotation, rotation²).
	•	Minimum sample size = 60 cars.
	•	Degree 3 polynomial: ~10 terms.
	•	Minimum sample size = ~100 cars (you might need more than 80 for higher degrees).

To avoid overfitting, aim for 20%-30% of your dataset (e.g., ~16–24 cars) for degree 2. Adjust as needed based on model performance.

2. How to Select Cars

To maximize the diversity of the training sample and ensure the regression generalizes well:

a. Stratify by Key Attributes

	•	Weight: Include cars spanning the full range of weights (light, medium, heavy).
	•	Drivetrain: Include all drivetrain types (RWD, FWD, AWD). Ideally, an equal number of each type.
	•	Lock-to-Lock Rotation: Cover the range from low (modern cars) to high (older cars).

b. Stratify by Surface Type

	•	Ensure that cars are selected for all surfaces (gravel, tarmac, snow) to capture differences in FFB sensitivity across surfaces.

c. Avoid Overlap

	•	Avoid clustering too many cars with similar attribute values (e.g., same drivetrain, similar weight) to prevent bias in the model.

d. Include Edge Cases

	•	Select extreme examples (e.g., the lightest car, the car with the highest rotation) to help the model learn boundary behaviors.

3. Sampling Strategy

Use a stratified sampling approach:
	1.	Cluster the Cars:
	•	Group cars into buckets based on their attributes (e.g., weight ranges, drivetrain type, and rotation ranges).
	2.	Random Sampling Within Clusters:
	•	Randomly select cars from each cluster, ensuring even representation across all clusters.

4. Practical Example

Assume you have the following attribute distributions for 80 cars:
	•	Weight: Ranges from 800 kg to 1600 kg.
	•	Drivetrain: 30 RWD, 30 FWD, 20 AWD.
	•	Lock-to-Lock Rotation: Ranges from 400° to 900°.

Select ~20 Cars:

	•	Weight: 6 light cars (<1000 kg), 8 medium (1000–1300 kg), 6 heavy (>1300 kg).
	•	Drivetrain: 7 RWD, 7 FWD, 6 AWD.
	•	Lock-to-Lock Rotation: Ensure at least 6 cars at the lower end (<540°), 6 at the upper end (>720°), and 8 in the middle range.

5. Evaluate Model Performance

After training, validate the model on the remaining cars. If the predictions are not accurate:
	1.	Add more cars to the training set, focusing on underrepresented areas (e.g., missing drivetrain types or weight ranges).
	2.	Adjust the polynomial degree or features.

This balanced approach ensures that your sample is representative and minimizes the risk of bias or overfitting in the regression model. Let me know if you’d like further clarification!