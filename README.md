
* Transform stock prices to graphs

```
python price_graph.py
```

* Calculate collective influence (CI) for graph nodes

```
python price_ci.py
```

* Price graph embedding

```
python price_embedding.py
```

* Generate dataset for train and test

```
python dataset.py
```

## Trainer Usage

Train:

```
usage: trainer.py [-h] [-e EPOCH] [-b BATCH] [-ts TIMESTEP] [-hs HIDDENSIZE] [-y YEARS [YEARS ...]] [-sn SEASON] [-dr DROPRATIO] [-s SPLIT] [-i INTERVAL] [-l LRATE] [-l2 L2RATE] [-t]

Train the price graph model on stock

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCH, --epoch EPOCH
                        the number of epochs
  -b BATCH, --batch BATCH
                        the mini-batch size
  -ts TIMESTEP, --timestep TIMESTEP
                        the length of time_step
  -hs HIDDENSIZE, --hiddensize HIDDENSIZE
                        the length of hidden size
  -y YEARS [YEARS ...], --years YEARS [YEARS ...]
                        an integer for the accumulator
  -sn SEASON, --season SEASON
                        the test season of 2019
  -dr DROPRATIO, --dropratio DROPRATIO
                        the ratio of drop
  -s SPLIT, --split SPLIT
                        the split ratio of validation set
  -i INTERVAL, --interval INTERVAL
                        save models every interval epoch
  -l LRATE, --lrate LRATE
                        learning rate
  -l2 L2RATE, --l2rate L2RATE
                        L2 penalty lambda
  -t, --test            train or test
```

