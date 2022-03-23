1. PBRE extracts rules for iris, Wisconsin breast cancer, sonar, German credit, ionosphere, hear disease datasets as shown in the following the files:

	v3/{xxx}_data_model.py: train the model for each dataset

	{xxx}/rule_generation_{xxx}.py: generate rules for seen dataset containing  "Generate instance rules"+"Generalize instance rules"+"Combine rules"  (including 		files: {xxx}/sample_enumerate_abstraction_pedagogical_ing_{xxx}.py and {xxx}/tree_ing_{xxx}.py)

	{xxx}/logicalRuleEvaluation_{xxx}.py: generate rules for seen dataset including  Refine rules":

		- execute cell "B" to obtain "{xxx}_test_acc_max.npy" and "{xxx}_not_significant_B.npy" based on the test(unseen) dataset 
		- execute cell "F" to refine the rules and obtain the final insignificant states: "{xxx}_not_signigicant_F.npy"
		- execute cel "Check" to load "{xxx}_not_signigicant_F.npy" and check the accuracy on test(unseen) dataset or train(seen) dataset.




2. RxNCM extracts rules for iris, Wisconsin breast cancer, sonar, German credit, ionosphere, hear disease datasets as shown in the following the file:

	{xxx}/rule_generation_{xxx}_rxncm_.py: extract rules using RxNCM and calculate the predefined metric values 




3. PBRE extracts rules for light service simulated by the DQN: 

	v3/light_service_structure_{xxx}.py: train the DQNs

	light_services/generate_testing_dataSet_{xxx}.py: generate seen or unseen dataset

	light_services/rule_generation_{xxx}.py: generate rules for seen dataset. Since the number of input states for each DQN is small, there is no need to refine 		rules. (including: light_services/sample_enumerate_abstraction_pedagogical_ing_{xxx}.py and light_services/tree_ing_{xxx}.py)

	evaluate rules:

		- light_services/logicalRuleEvaluation_{xxx}.py: calculate replayMemory and total rewards of rules
		- light_services/pedagogicalRuleEvaluation_{xxx}.py: refine replayMemory and total rewards of DQNs
		- light_services/draw_pedaExplainer.py: compare replayMemory and total rewards of rules and DQNs










