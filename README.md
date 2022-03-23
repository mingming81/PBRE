1. PBRE to extract rules for iris, Wisconsin breast cancer, sonar, German credit, ionosphere, hear disease datasets contains the following the files:

v3/{xxx}_data_model.py: train the model for each dataset

{xxx}/rule_generation_{xxx}.py: generate rules for seen dataset including  "Generate instance rules"+"Generalize instance rules"+"Combine rules"  (including: {xxx}/sample_enumerate_abstraction_pedagogical_ing_{xxx}.py and {xxx}/tree_ing_{xxx}.py)

{xxx}/logicalRuleEvaluation_{xxx}.py: generate rules for seen dataset including  Refine rules":

	- execute cell "B" to obtain "{xxx}_test_acc_max.npy" and "{xxx}_not_significant_B.npy" based on the test(unseen) dataset 
	- execute cell "F" to refine the rules and obtain the final insignificant states: "{xxx}_not_signigicant_F.npy"
	- execute cel "Check" to load "{xxx}_not_signigicant_F.npy" and check the accuracy on test(unseen) dataset or train(seen) dataset.


2. RxNCMto extract rules for iris, Wisconsin breast cancer, sonar, German credit, ionosphere, hear disease datasets contains the following the files:
{xxx}/rule_generation_{xxx}_rxncm_.py


3. PBRE to extract rules for light service simulated by the DQN: (the correspondance between the verions in this folder and the light experiments in the paper: 2_v9 -> DQN_v1, 2_v7-> DQN_v2, lstm->DQN_v3)

v3/light_service_structure_{xxx}.py: train the DQNs

light_services/generate_testing_dataSet_{xxx}.py: generate training or testing dataset

light_services/rule_generation_{xxx}.py: generate rules for seen dataset. Since the number of input states for each DQN is small, there is no need to refine rules. (including: light_services/sample_enumerate_abstraction_pedagogical_ing_{xxx}.py and light_services/tree_ing_{xxx}.py)

evaluate rules:

	- light_services/logicalRuleEvaluation_{xxx}.py: calculate replayMemory and total rewards of rules
	- light_services/pedagogicalRuleEvaluation_{xxx}.py: refine replayMemory and total rewards of DQNs
	- light_services/draw_pedaExplainer.py: compare replayMemory and total rewards of rules and DQNs










