from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB

# FIXME change your data path/folder here 

train_folder = './data/train/' # folder
test_folder = './data/test/' # folder
label_fname = './data/train_labels.csv' # path

pred_fname = './submission_NB.csv' # predicitons

##########################################

print 'load label data ...'
label = {}
with open(label_fname, 'rb') as f:
    next(f)
    for line in f:
        fidx, ans = line.rstrip('\n').split(',')
        label[fidx] = ans

print 'load training data ...'
tr_ids = []
tr_ans = []
tr_raw = []
cnt = 0
for fname in listdir(train_folder):
    filepath = join(train_folder, fname)
    if isfile(filepath):
        cnt += 1
        if cnt % 10000 == 0: print 'finished:', cnt # line counter
        tr_ids.append(fname[:-4])
        tr_ans.append(label[fname[:-4]])
        with open(filepath, 'rb') as f:
            tr_raw.append( f.read() )

print 'load testing data ...'
te_ids = []
te_raw = []
cnt = 0
for fname in listdir(test_folder):
    filepath = join(test_folder, fname)
    if isfile(filepath):
        cnt += 1
        if cnt % 10000 == 0: print 'finished:', cnt # line counter
        te_ids.append(fname[:-4])
        with open(filepath, 'rb') as f:
            te_raw.append( f.read() )

print 'initilize tf vectorizer ...'
vectorizer = TfidfVectorizer(use_idf=False, stop_words='english')
vectorizer.fit(tr_raw+te_raw)

print 'transform data to tfidf vector ...'
tr_vec = vectorizer.transform(tr_raw)
te_vec = vectorizer.transform(te_raw)

print 'build Naive Bayes classifier ...'
clf = BernoulliNB()
clf.fit(tr_vec, tr_ans)

print 'make predictions ...'
clf_predictions = clf.predict_proba(te_vec)

print 'store predictions in <%s>', pred_fname
pred_out = ["Id,predictions"]
num_pred = range(30)
for fid, pred in zip(te_ids, clf_predictions):
    top_rec = sorted(num_pred, key=lambda k: pred[k], reverse=True)[:3]
    pred_out.append("%s,%s" % (fid, ' '.join( [clf.classes_[rec] for rec in top_rec] )))
with open(pred_fname, 'w') as f:
    f.write('%s\n' % ('\n'.join(pred_out)))

