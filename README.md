# reid_baseline
reid baseline model for exploring softmax and triplet hard loss's influence.

### Classification
<img src='https://ws3.sinaimg.cn/large/006tKfTcly1fs76ysvu3zj30kg0d60t5.jpg' height='200'>  <img src='https://ws2.sinaimg.cn/large/006tKfTcly1fs76zbtfxcj30js0d674m.jpg' height='200'>


### Triplet Hard
<img src='https://ws2.sinaimg.cn/large/006tNc79ly1fs3sxc54xjj30ka0d6dgd.jpg' height='200'> <img src='https://ws2.sinaimg.cn/large/006tNc79ly1fs3tpat6emj30k00d2t93.jpg' height=200>


### Classification + Triplet Hard

<img src='https://ws2.sinaimg.cn/large/006tKfTcly1fs79doog34j30ja0cudg5.jpg' height='200'> <img src='https://ws2.sinaimg.cn/large/006tNc79ly1fs3tpat6emj30k00d2t93.jpg' height=200>


## Results

| loss | rank1 | map |
| --- | --| ---|
| triplet hard | 89.9% | 76.8% | 
| softmax | 87% | 65% |
|triplet + softmax | 89.7% | 76.2% |


