Here are some actionable items.
Once we work on these items, we can probably hand it in for publication at a conference or journal as given by Lukas.

- [ ] Section: Literature research
- [ ] Stock embeddings require more details. 
    - [ ] Specifically what is the input? 
    - [ ] What do the dimensions look like?
    - [ ] Elaborate on the motivation of the stock embeddings
    - [ ] Explain intuitively, why stock embeddings should help
    - [ ] What is the motivation to use stock embeddings? 
    - [ ] What is the intuitive explanation behind it?
    - [ ] Mathematically, why would stock embeddings capture more information?
    - [ ] Provide a more detailed analysis on the stock embeddings. (i.e. take code from word embeddings to visualize, and just visualize them). 
    - [ ] What happens when you interpolate the stock embeddings in the stock embedding space?
    - [ ] Can you learn something about stock similarity from the embeddings?
    - [ ] Dimensions of embedding space. Why was this dimension chosen?
- [ ] Elaborate on how much "signal" there is in the raw numbers data. Specifically, refering to the comment: deep learning is successful whenever there is a signal in the data, e.g. input-output correlations. Now I can see that the ability to predict stock prices is a tempting task, but do you know how much "signal" there is in financial data? You could have elaborated also on applying deep learning to secondary data e.g. twitter, news articles etc, as opposed to raw stock prices.
- [ ] As a side note: By focusing on next day returns, you limit to high-frequency trading (in which case your algorithm has to at least outperform transaction costs to be profitable). I would have found it interesting if you elaborated a bit on the challenges and potential of deep nets in high-frequency trading vs. long-term investing.
- [ ] Clarify: Two main contributions are (i) stock-embeddings, and (ii) feature selection using correlation measure.
- [ ] Include training details: i.e. loss curves 
- [ ] Compare to RNNs

- [ ] Below Fig 3, Fig 2 is referenced where I think Table I was meant
- [ ] e.g. f_0 in Table I is left undefined (I am guessing it is 0.0087) and unmotivated
- [ ] Also in Table I the time period is arbitrary
- [ ] it is said that the task is classification but is it binary (up or down) or multiclass (up by 1-5 points, up by 6-10 points etc.)? ->Finally mentioned in V.B
- [ ] loss function is left undefined.
- [ ] The reshuffling and r* stuff is not explained very well. 
- [ ] A verb is missing in the sentence "On the other hand by the reshuffling..."
- [ ] Specify how NaN values are removed

- [ ] Selected features are not given 0_0???? what does this mean?
- [ ] Add reference in Section V.A
- [ ] Do not apply earthquake selection on features, but on input (i.e. no feature-engineering)!

- [ ] Visualize a time-series model for the benchmarks (i.e. tickers which say "up"/"down")
- [ ] Check if there's survivorship bias in the data.
