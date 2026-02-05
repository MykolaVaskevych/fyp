# Benchmark methodology

Initial prototyping of DQN were made in a way to push algorithm with setted hyperparameters to its limit and reach overfitting.
But after research of current state of evaluationg perfomence of RL algorithms I must admit that that way is wrong.

Recent papers suggest that many works related to RL algorithms are underperforming and published results of many papers which involved evaluation of RL algorithms were impossible to reporoduce. Recent study shows that many algorithms actual perfomence is from 10 to 20 percent lower than stated.

To terminate this discrepancy, recent papers suggest alternative approaches of evaluation.

some of them involve constant hyperparameters without finetuning them to spesific envoriment,
as some algorithms may perform better than others out of the box while others may get an extra few percents after very difficalt finetuning and many iterations, which is not a fair comparison.

other method that i review from paper PAPER NAME, is to test set of algorithms on the set of enviroments where one agent picks the best algorithms and another worst enviroments.

some papers also suggest to include hyperparameter tuning into benchmarking algorithms iteself to make tuning fair and part of evaluation process.

its also suggested to do less iterations as the difference between 1k and 100k/1m is not significant and needed only for spoting small differences.

due to my reserch focusing on comparison of 2 different algorithm families, with mainly involving basic vaniala interpretation...

# notes

certain seeds can benefit one algorithm more than another. paper 1

so use many seeds

---

randomeness is a bitch. state my pc. state all shit.

note 95 % confidence, its randomness shit.
