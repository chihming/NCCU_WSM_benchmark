# 1042 NCCU WSM - Benchmark Usage:
**Competition Website**: [NCCU WSM - User Mood Prediction](https://inclass.kaggle.com/c/nccu-wsm-user-mood-prediction)

**Default Environment**: This is the benchmark of [**VSM TF + Bernoulli Naive Bayes**]
- python version: 2.7 (you can modify the code to fit the python 3.x)
- scikit-learn: http://scikit-learn.org/stable/install.html

**Step 1**. Change the file *path* & *folder* in **1042_WSM_benchmark.py**
```python
  8 train_folder = './data/train/' # the folder containing the training data
  9 test_folder = './data/test/' # folder containing the testing data
 10 label_fname = './data/train_labels.csv' # the path of train_labels.csv
```

**Step 2**. Run the program:
```python
python 1042_WSM_benchmark.py
```
It will create a solution file named *submission_NB.csv*.

**Step 3**. Submit the *submission_NB.csv* to the competition website.
