# psmatching
Propensity Score Matching Python Package

- Calculation of propensity scores based on a LR model
- Matching of k controls to each case patient
- Use of a caliper to control the maximum difference between propensity scores

## Install psmatching

```
python setup.py install

pip install git+https://github.com/rickydangc/psmatching
```

## Usage

```
import psmatching.match as psm

path = "./sample.csv"
model = "CASE ~ AGE + ENCODED_SEX + ENCODED_RACE + ENCODED_CCI_GROUP"
gap = 180
k = "5"

ps = psm.PSMatch(path, model, k, gap)
ps.prepare_data()
caliper = ps.set_caliper('logit', 0.01)
ps.match_by_neighbor(caliper)
```

## Simple Run
```
ps.run()
```
