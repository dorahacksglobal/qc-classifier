# qc-classifier
AI-inspired Classification of Quantum Computers
Current benchmark is 63% with the current combination of preprocessing features. When four lines are concatenated together, it reaches 79%. When fourty lines are concatenated, it reaches >95% acuracy. (However, we eventually want to get the best accuracy with no input data concatenation, so just using training and test data of 100 bit QNRG input lines, in other words, to ensure practical use cases). 

This configuration has six feature extraction tests before we run the model - 'spectral_test', 'shannon_entropy', 'frequency_test', 'runs_test', 'autocorrelation', and min_entropy. Many other combinations of tests have not been tested, as well as certain adjustments to the models or different prediction models altogether. The current two highest results come from the Gradient Booster model, followed by the Random Forest. 

Feel free to email sid@dorahacks.biz or shrey@dorahacks.biz if you have any questions!
