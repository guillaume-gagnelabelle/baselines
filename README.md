# Baselines
### Run command
python [algo] --dataset [dataset] --num_labeled [num_labeled]
- algo -> [main_fixmatch.py, main_mt.py, main_pham.py, main_simCLR.py]
- dataset -> [cifar, mnist]
- num_labeled -> integer

### Architecture
- MNIST: inferenceModel as we defined it *(+ a linear layer for SimCLR)*
- CIFAR: Resnet50 *(+ a linear layer for SimCLR)*

*For Mean Teacher, the algorithm requires to add some noise in the model (e.g. dropouts)*

### EMA models
Both FixMatch and Pham have the alternative to use an exponential moving average of the model to assess performance with the tag *--use_ema* on the command line. This tag was not use to produce the results as it provides a fairer comparison of the algorithms with ours.

### Hyperparameters search
No hyperparameters search was made on any of those algorithms. They are supposed to be optimized for CIFAR and therefore the results on MNIST might be suboptimal.






