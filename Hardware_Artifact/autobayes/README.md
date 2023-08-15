# Experiment
All the code below should be run in `AutoBayesNN/experiment` directory.

## 1. Conversion time report
To generate the time report for the AutoBayes Converter, old reports should be cleaned first:
```
rm -r timing/*
```
Then run the script:
```
./time_cost.sh
```
Reports will be generated in `timing/` folder.

## 2. Synthesis reports for different number of dropout layers
To generate the synthesis report, old reports should be cleaned first:
```
rm -r diff_dropouts/*
```
Or if you would like to keep the old reports, run:
```
mv diff_dropouts old_diff_dropouts
```
Then run the script:
```
./diff_dropouts.sh
```
Synthesis project will be generated in `diff_dropouts` folder. For example, you could open `diff_dropouts/LeNet-1/myproject_prj/solution1/syn/report/myproject_csynth.rpt` to see the report.
