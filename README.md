# ReID_baseline
Baseline model (with bottleneck) for person ReID (using softmax and triplet loss).

## Learning rate
This is the warmpup strategy learning rate

<img src='https://ws3.sinaimg.cn/large/006tNc79ly1fthmcjwoaaj31kw0natad.jpg' height='200'>

## Results

| loss | rank1 | map |
| --- | --| ---|
| softmax | 87.9% | 70.1% |
| triplet | 88.8% | 74.8% | 
|triplet + softmax | 92.0% | 78.1% |

