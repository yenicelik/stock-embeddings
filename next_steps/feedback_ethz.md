Two reviewers reviewed this paper, and added comments.
Please see the comments below, I will create a list of actionable items below!

Reviewer 1

#### 
3. Quality of Paper (40%)
5.0
+/- report is mostly well-written but quite minimalistic when it comes to deep learning
- no details are given regarding the stock-embeddings. The only part where you mention them is: "The model has two inputs [...]. These are split into two different streams. Stream 1 and it’s underlying sequential sub-model for the numerical items, as well as stream 2 and it’s underlying sequential sub-model for the embedded items." However, no details are given regarding stream 2.
- in general: deep learning is successful whenever there is a signal in the data, e.g. input-output correlations. Now I can see that the ability to predict stock prices is a tempting task, but do you know how much "signal" there is in financial data? You could have elaborated also on applying deep learning to secondary data e.g. twitter, news articles etc, as opposed to raw stock prices. 
+/- As a side note: By focusing on next day returns, you limit to high-frequency trading (in which case your algorithm has to at least outperform transaction costs to be profitable). I would have found it interesting if you elaborated a bit on the challenges and potential of deep nets in high-frequency trading vs. long-term investing.
4. Creativity of Idea (30%)
4.75
+/- two main contribution are (i) stock-embeddings, (ii) feature selection
- no details are given regarding the stock-embeddings: this makes half of your main contributions practically useless. See also points above.
- Stock Embeddings is itself an interesting subject, even before using them to predict stock prices. You could have approached this topic from a more scientific perspective, by providing an in-depth analysis of the learned embeddings. E.g. what happens when you interpolate in embedding space? Can you learn something about stock similarity from the embeddings?
- feature selection was found not to help: in Table IV you report "significance", which I guess is meant as synonym for "feature selection/significance".
- authors also introduce some additional features, such as relative returns. These additional features are strictly speaking totally redundant. The core methodology of deep learning is not to introduce such hand-crafted features. Hence, apart from the fact that this is kind of against the principle of deep learning, I would really have liked to see whether you gain anything by introducing them, or whether the deep net actually simply ignores these hand-crafted features.
- you essentially propose a non-recurrent architecture, i.e. you try to predict the return one day ahead from a couple of past returns, but you don't leverage the full time series. In the context of time series analysis, you should really have compared against and discussed recurrent neural network solutions to the stock price prediction problem.
5. Execution of Idea (20%)
5.0
- approach is very minimalistic
- learning embeddings entails a lot of design choices that you should have elaborated on, e.g. dimension of embedding space.
- include training details, e.g. loss curves
- should at least compare with RNNs
+ code looks good


Reviewer 2

#### Questions
3. Quality of Paper (40%)
5
+ The paper is well written and well structured in general

- Some details are not 100% precise though:
> Below Fig 3, Fig 2 is referenced where I think Table I was meant
> e.g. f_0 in Table I is left undefined (I am guessing it is 0.0087) and unmotivated
> Also in Table I the time period is arbitrary
> it is said that the task is classification but is it binary (up or down) or multiclass (up by 1-5 points, up by 6-10 points etc.)? ->Finally mentioned in V.B
> loss function is left undefined.
- The reshuffling and r* stuff is not explained very well. Also a verb is missing in the sentence "On the other hand by the reshuffling..."
- "NaN values are removed" but it is unclear how? Are they replace by means or zeros or is that day left out completely?

- The selected features are not given 0_0
- missing reference in Section V.A 
4. Creativity of Idea (30%)
5
+ The analogy to other power-law distributed events such as earthquakes is valid and it is thus a nice idea to incorporate feature selection tools from that field

- Yet, this novel approach does not necessarily yield novel features since the proposed features in the beginning of Chapter III are themselves not novel. E.g. the feature presented in Table 1 resembles a classic momentum strategy which quants have been using for ages. Consequently it does not make any difference in the final network (table IV)

- Stock embedding might be a good idea but the approach is nowhere motivated and it is left completely unexplained why it helps

5. Execution of Idea (20%)
5
+ benchmark is comprehensive (even though a time series model would have been nice)
- Looking ahead: Is there a survivor bias in the data?