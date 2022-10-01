# gograd

## What is it?
This is a partial port of [capmangrad](https://github.com/dsprn/capmangrad) to the Go programming language, used here primarily to revise my rough understanding of the language.
As of the time the initial commit was made this implementation lacked some features capmangrad added on top of [micrograd](https://github.com/karpathy/micrograd), such as the capability to import/export a trained model to a json file and a few others. For a complete list take a look at the TODOS section below.
It's worth noting that one of these features, namely cross validation (used here to look for the L2 regularization lambda hyperparameter), takes adavantage, in this implementation, of the concurrent capabilities of the Go programming language using goroutines and channels to complete its task.

## How to use it
This project can be run from the command line with the following command (once positioned in the project folder)
```
go run github.com/dsprn/gograd
```
This will produce an output pretty much like the following as it will certainly differ from it because of the random weights choice when creating each model, like those used in cross validation and the one trained just after that.
```
[dsprn@xps gograd]$ go run github.com/dsprn/gograd
==> Using Cross Validation to look for the best L2 lambda hyperparameter in values ranging from 0.0000 to 0.0100
hyperpar=0.0000, accuracy=52%
hyperpar=0.0005, accuracy=50%
hyperpar=0.0010, accuracy=50%
hyperpar=0.0015, accuracy=51%
hyperpar=0.0020, accuracy=50%
hyperpar=0.0025, accuracy=48%
hyperpar=0.0030, accuracy=53%
hyperpar=0.0035, accuracy=51%
hyperpar=0.0040, accuracy=53%
hyperpar=0.0045, accuracy=48%
hyperpar=0.0050, accuracy=48%
hyperpar=0.0055, accuracy=47%
hyperpar=0.0060, accuracy=48%
hyperpar=0.0065, accuracy=50%
hyperpar=0.0070, accuracy=51%
hyperpar=0.0075, accuracy=48%
hyperpar=0.0080, accuracy=50%
hyperpar=0.0085, accuracy=51%
hyperpar=0.0090, accuracy=44%
hyperpar=0.0095, accuracy=47%
==> L2 lambda value=0.0030

==> Choosing inputs and relative label from a preloaded dataset...
==> Getting inputs at index 28 and relative label
==> Input values=[1.02550753 -0.754465849]
==> Expected value=1.000000

==> Start training the model...
pass=1, predicted=1.348178, expected=1.0, loss=0.121228, reg=0.306901, tot_loss=0.428129
pass=2, predicted=1.265511, expected=1.0, loss=0.070496, reg=0.306891, tot_loss=0.377387
pass=3, predicted=1.202535, expected=1.0, loss=0.041020, reg=0.306883, tot_loss=0.347904
pass=4, predicted=1.154532, expected=1.0, loss=0.023880, reg=0.306877, tot_loss=0.330757
pass=5, predicted=1.117926, expected=1.0, loss=0.013907, reg=0.306872, tot_loss=0.320778
pass=6, predicted=1.090002, expected=1.0, loss=0.008100, reg=0.306868, tot_loss=0.314968
pass=7, predicted=1.068695, expected=1.0, loss=0.004719, reg=0.306864, tot_loss=0.311583
pass=8, predicted=1.052434, expected=1.0, loss=0.002749, reg=0.306861, tot_loss=0.309610
pass=9, predicted=1.040023, expected=1.0, loss=0.001602, reg=0.306858, tot_loss=0.308460
pass=10, predicted=1.030549, expected=1.0, loss=0.000933, reg=0.306856, tot_loss=0.307789
pass=11, predicted=1.023316, expected=1.0, loss=0.000544, reg=0.306853, tot_loss=0.307397
pass=12, predicted=1.017794, expected=1.0, loss=0.000317, reg=0.306851, tot_loss=0.307167
pass=13, predicted=1.013578, expected=1.0, loss=0.000184, reg=0.306849, tot_loss=0.307033
pass=14, predicted=1.010359, expected=1.0, loss=0.000107, reg=0.306847, tot_loss=0.306954
pass=15, predicted=1.007901, expected=1.0, loss=0.000062, reg=0.306845, tot_loss=0.306907
==> DONE
```

To compile the project run the following in your terminal
```
go build github.com/dsprn/gograd
```

## Tests
To run the tests written for this very preliminary version of the project go to the grad directory, where they are present, and run the following command
```
go test
```
or for a more verbose output
```
go test -v
```

## Todos
Following are the features that are present in capmangrad but are missing in this version.
Listed here in no particolar order:
* saving model to a json file
* getting a visualization of the computational graph
