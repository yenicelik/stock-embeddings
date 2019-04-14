Lukas also set out a bunch of feedback and items to work on.
The rawtext is the following.

IDEAS
Literature
-	I can provide an extended literature review (especially of the papers in the literature review longlist)
                                                                                                   
Methodology & Testing
-	Number of Trials (Number of NNs with different weight initializations tested)?
-	More benchmarks, especially against RNNs (LSTMs/GRUs)
-	Direct comparison with other feature engineering approaches – PCA, IG, RC, LC, etc....
-	Significance testing could be done with a different time series – significance would, in this case, indicate generalizability to different asset classes etc. or, alternatively, with multiple ‘samples’/trained networks that are compared to each other with, e.g., a Wilcoxon rank sum test
-	Explain how feature selection method (temporal correlation) compares to methods based on linear correlation, rank correlation, etc (maybe it’s a good idea to actually benchmark against these)
-	Other possible accuracy metrics: Accuracy in predicting positive changes (HR+), accuracy in predicting negative changes (HR-). This may be important as certain algorithms have shown to exhibit bias in one or another direction. Employing these metrics may also uncover bias in the input data itself. 
-	Ideal Classifier & Ideal Profit Ratio – It might be interesting to show what kind of profit would be possible in employing the feature selection method described in the paper (e.g., for a fixed number of portfolio sizes, fixed rebalancing periods) and comparing that to a system which makes perfect forecasts (thus yielding the maximum profit for a given trading framework))


Possible journals for publication from the top of my head:
♣	Expert Systems with Applications
♣	Neurocomputing
♣	Applied Soft Computing
♣	IEEE Transactions on Neural Networks and Learning Systems
♣	Quantitative Finance
♣	...
