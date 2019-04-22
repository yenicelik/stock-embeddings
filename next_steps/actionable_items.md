Here are some actionable items.
Once we work on these items, we can probably hand it in for publication at a conference or journal as given by Lukas.

### Still TODO

- [ ] Stock embeddings require more details.
    - [ ] Mathematically, why would stock embeddings capture more information? (Thomas)
    - [ ] What happens when you interpolate the stock embeddings in the stock embedding space? [I'd suggest we skip this one] (Thomas)
    - [ ] Can you learn something about stock similarity from the embeddings? (Thomas)

- [ ] Elaborate on how much "signal" there is in the raw numbers data. 
Specifically, referring to the comment: deep learning is successful whenever there is a signal in the data, e.g. input-output correlations. 
Now I can see that the ability to predict stock prices is a tempting task, but do you know how much "signal" there is in financial data? 
You could have elaborated also on applying deep learning to secondary data e.g. twitter, news articles etc, as opposed to raw stock prices. (Not sure who? And to what extent?)
- [ ] Elaborate on long-term investing, swing-trading, day-trading, high-frequency trading. (Lukas, Thomas)
- [ ] Compare to RNNs (David) [Will not do this]

- [ ] e.g. f_0 in Table I is left undefined (I am guessing it is 0.0087) and unmotivated (Kostas, Thomas)
- [ ] The reshuffling and r* stuff is not explained very well. (Thomas, Kostas, Miller) 

- [ ] Selected features are not given 0_0???? what does this mean? --> Unclear what the TA meant by this I think. 

- [ ] Visualize a time-series model for the benchmarks (i.e. tickers which say "up"/"down") (David, Lukas)
- [ ] Check if there's survivorship bias in the data. (Kostas, Thomas)

### Done

- [x] Add a section: Literature research. It exists already here and there, but a more elaborate version would be nice (Lukas)

- [x] Extract trained embeddings? (David)  
- [x] Provide a more detailed analysis on the stock embeddings. (i.e. take code from word embeddings to visualize, and just visualize them). (David) 

- [x] Check what different embedding sizes produce (i.e. another table?) (David)
- [x] it is said that the task is classification but is it binary (up or down) or multiclass (up by 1-5 points, up by 6-10 points etc.)? ->Finally mentioned in V.B (David)
- [x] Implement model saving and model restore? (David)
- [x] Specifically what is the input? (David)
- [x] What do the dimensions look like? (David)
- [x] Elaborate on the motivation of the stock embeddings. Explain intuitively, why stock embeddings should help. What is the motivation to use stock embeddings? (David) 
- [x] Dimensions of embedding space. Why was this dimension chosen? (David)

- [x] Clarify: Two main contributions are (i) stock-embeddings, and (ii) feature selection using correlation measure. (David)
- [x] Below Fig 3, Fig 2 is referenced where I think Table I was meant (David)
- [x] Also in Table I the time period is arbitrary (David)
- [x] loss function is left undefined. (David)
- [x] Specify how NaN values are removed (David, Thomas)
- [x] Add reference in Section V.A (Lukas)

- [x] A verb is missing in the sentence "On the other hand by the reshuffling..." (Thomas, Kostas, Miller)
- [ ] Include training details: i.e. loss curves  (David) [Will not do this as loss curves are too narrow]
- [x] Do not apply earthquake selection on features, but on input (i.e. no feature-engineering)! (Thomas, David) Be more explicit that this is done on both input and on other features, and that the correlaiton measure can choose amongst those! [I think is covered! ]
