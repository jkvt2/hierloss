Implementation in Tensorflow of WordTree-style Hierarchical Classification Loss as used in YOLO9000: Better, Faster, Stronger (https://arxiv.org/abs/1612.08242).

I needed but couldn't find an implementation of the loss function described in the paper so I made this.

Hierarchical classification loss allows you to train classification with labels of varying specificity. I'll leave it to the authors to describe the benefits of such a hierarchical loss:

> To perform classification with WordTree we predict conditional probabilities at every node for the probability of each hyponym of that synset given that synset. For example, at the “terrier” node we predict:<br/><br/>
> Pr(Norfolk terrier|terrier)<br/>
> Pr(Yorkshire terrier|terrier)<br/>
> Pr(Bedlington terrier|terrier)<br/>
> ...<br/><br/>
> If we want to compute the absolute probability for a particular node we simply follow the path through the tree to the root node and multiply to conditional probabilities. So if we want to know if a picture is of a Norfolk terrier we compute:<br/>
> Pr(Norfolk terrier) = Pr(Norfolk terrier|terrier)<br/> ∗ Pr(terrier|hunting dog)<br/> ∗ . . . <br/>∗ Pr(mammal|animal)<br/>∗ Pr(animal|physical object)<br/><br/>
> For classification purposes we assume that the the image contains an object: Pr(physical object) = 1. To validate this approach we train the Darknet-19 model on WordTree built using the 1000 class ImageNet. To build WordTree1k we add in all of the intermediate nodes which expands the label space from 1000 to 1369. During training we propagate ground truth labels up the tree so that if an image is labelled as a “Norfolk terrier” it also gets labelled as a “dog” and a “mammal”, etc. To compute the conditional probabilities our model predicts a vector of 1369 values and we compute the softmax over all sysnsets that are hyponyms of the same concept, see Figure 5.<br/>
> Using the same training parameters as before, our hierarchical Darknet-19 achieves 71.9% top-1 accuracy and 90.4% top-5 accuracy. Despite adding 369 additional concepts and having our network predict a tree structure our accuracy only drops marginally. Performing classification in this manner also has some benefits. Performance degrades gracefully on new or unknown object categories. For example, if the network sees a picture of a dog but is uncertain what type of dog it is, it will still predict “dog” with high confidence but have lower confidences spread out among the hyponyms.

Running hierarchical_loss.py runs a small example on a toy dataset of 23 nodes. With only one example per class, this does not train to 100% test accuracy. However, even where the predictions are wrong, they are typically wrong only in the final branch split(s) (i.e. elements/base/strong/naoh vs elements/base/strong/koh).
