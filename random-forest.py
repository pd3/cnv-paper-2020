#!/usr/bin/env python3
#
# The memory requirements are bigger because the whole input
# is kept in memory twice: once as the original text to append
# the final scores too and once converted to numeric values.
#
# If this is a problem, the script could read the input values
# twice from a file instead reading it from stdin.
#

import sys, re, gzip

def usage(msg=None):
    global sys
    if msg==None:
        print('Usage: cat data.txt | random-forest.py [OPTIONS]')
        print('Options:')
        print('   -f, --features            list of covariates, the first indicates the training column with TP, FP, or .')
        print('   -n, --ntrees INT          number of trees in the forest [100]')
        print('   -o, --out FILE            the output file (bgzipped)')
    else:
        print >> sys.stderr, msg
    sys.exit(1)

def parse_args():
    global sys
    out = { 'ntrees':100 }
    args = sys.argv[1:]
    if len(args) < 1: usage()
    while len(args):
        if args[0]=='-h' or args[0]=='-?' or args[0]=='--help': 
            usage()
        elif args[0]=='-f' or args[0]=='--features': 
            args = args[1:]
            out['features'] = args[0]
        elif args[0]=='-o' or args[0]=='--out': 
            args = args[1:]
            out['out'] = args[0]
        elif args[0]=='-n' or args[0]=='--ntrees': 
            args = args[1:]
            out['ntrees'] = int(args[0])
        else:
            usage("The argument is not recognised: "+args[0])
        args = args[1:]
    if 'features' not in out: usage('Missing the -f option')
    if 'out' not in out: usage('Missing the -o option')
    out['features'] = out['features'].split(',')
    out['log'] = out['out']+'.log'
    return out

def parse_header(line):
    global re
    hdr = {}
    row = line.split('\t')
    for i in range(len(row)):
        key = re.sub(r'^#\s*', '', row[i])
        key = re.sub(r'^\[\d+\]\s*','',key)
        hdr[key] = i
    return hdr

def parse_value(value, dflt=float('NaN')):
    if value=='.' or value=='' or value=='NA': return dflt
    return float(value)

def replace_nan(values, avg=None):
    global np
    if avg==None:
        avg  = []
        navg = []
        for i in range(len(values[0])):
            avg.append(0)
            navg.append(0)
        for i in range(len(values)):
            for j in range(len(values[i])):
                if np.isnan(values[i][j]): continue
                avg[j]  += values[i][j]
                navg[j] += 1
        for i in range(len(avg)): avg[i] /= navg[i]
        for i in range(len(values)):
            for j in range(len(values[i])):
                if np.isnan(values[i][j]): values[i][j] = avg[j]
    return avg

def read_data(args):
    import sys
    rows = []
    dat  = []
    train_dat   = []
    train_class = []
    hdr = None
    tp_fp_key = args['features'][0]
    features = args['features'][1:]
    nfp = 0
    ntp = 0
    for line in sys.stdin:
        line = line.rstrip('\n')
        if line[0]=='#': 
            args['hdr_line'] = line
            hdr = parse_header(line)
            continue
        elif hdr==None:
            print("Error: no header line found (the first line must start with the hash character)")
            sys.exit(1)
        row = line.split('\t')
        if tp_fp_key not in hdr:
            print("The key \"%s\" is not in the header!" % tp_fp_key)
            sys.exit(1)
        tp_fp = row[hdr[tp_fp_key]]
        vals = []
        try:
            for key in features:
                if key not in hdr:
                    print("The key \"%s\" is not in the header!" % key)
                    sys.exit(1)
                vals.append( parse_value(row[hdr[key]]) )
        except:
            print(row)
            print(sys.exc_info())
            sys.exit(1)
        rows.append(line)
        dat.append(vals)
        if tp_fp!='.':
            train_dat.append(vals)
            if tp_fp=='TP':
                train_class.append(1)
                ntp += 1
            elif tp_fp=='FP':
                train_class.append(0)
                nfp += 1
            else:
                print("Unknown FP/TP class: \"%s\"" % tp_fp)
                sys.exit(1)
    print("Number of TP, FP, UNK: %d,%d,%d" % (ntp,nfp,len(dat)))
    avg = replace_nan(dat)
    replace_nan(train_dat,avg)
    args['rows'] = rows
    args['ntp']  = ntp
    args['nfp']  = nfp
    return dat,train_dat,train_class

def run_models(args):
    global sys,np
    import pipes
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn import clone
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier)
    from sklearn.tree import DecisionTreeClassifier

    import matplotlib.pyplot as plt
    import sklearn
    from sklearn import model_selection
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC

    print('The scikit-learn version is {}.'.format(sklearn.__version__))

    dat,train_dat,train_class = read_data(args)

    models = []

    #   #   LogisticRegression:  0.832309 (0.301088)
    #   #   LDA:                 0.869074 (0.222952)
    #   #   KNN:                 0.810250 (0.317192)
    #   #   SVM:                 0.822015 (0.332504)
    #   #   GaussNaiveBayes:     0.877524 (0.056489)
    #   #   DecisionTree:        0.992647 (0.013558)
    #   #   RandomForest:        0.994118 (0.013478)
    #   #   ExtraTrees:          0.983802 (0.031847)
    #   #   AdaBoost:            0.994118 (0.013478)
    #   #   models.append(('LogisticRegression', LogisticRegression()))
    #   #   models.append(('LDA', LinearDiscriminantAnalysis()))
    #   #   models.append(('KNN', KNeighborsClassifier()))
    #   #   models.append(('SVM', SVC()))
    #   #   models.append(('GaussNaiveBayes', GaussianNB()))
    #   #   models.append(('DecisionTree', DecisionTreeClassifier(max_depth=None)))
    #   #   models.append(('ExtraTrees', ExtraTreesClassifier(n_estimators=n_estimators)))
    #   #   models.append(('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=n_estimators)))

    models.append(('RandomForest', RandomForestClassifier(n_estimators=args['ntrees'],oob_score=True)))

    log = sys.stderr
    if 'log' in args: log = open(args['log'],"w")
    log.write("NTP_FP_UNKN\t%d\t%d\t%d\n" % (args['ntp'],args['nfp'],len(dat)))

    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
        cv_results = model_selection.cross_val_score(model, train_dat, train_class, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        log.write("CV_MEAN_DEV\t%s\t%f\t%f\n" % (name,cv_results.mean(), cv_results.std()))

    importances = []
    predictions = []
    oob_scores  = []
    for name, model in models:
        model.fit(train_dat, train_class)
        oob_scores.append(model.oob_score_)
        importances.append(model.feature_importances_)
        pred = model.predict_proba(dat)
        predictions.append(pred)

    fh = sys.stdout
    if 'out' in args:
        pipe = pipes.Template() 
        pipe.append("bgzip -c",'--')
        fh = pipe.open(args['out'],'w')
    hdr  = args['hdr_line']
    nhdr = len(hdr.split('\t')) + 1
    fh.write(args['hdr_line']+"\t["+str(nhdr)+"]RandomForestScore"+'\n')
    for i in range(len(args['rows'])):
        score = 0
        for pred in predictions: score += pred[i][1]
        fh.write("%s\t%.3f\n" % (args['rows'][i],score))
    fh.close()

    # cat file.log | grep ^IMP | cut -f3,4 | sort -k2,2gr | mplot barplot -o rmme-barplot.png,pdf +type lbl-cnt +adj bottom=0.2 -c -F +wd 0.8 +la "rotation=35,ha='right',ma='center',fontsize=9" +sty mystyle +title 'Feature importance'
    for m in range(len(models)):
        model = models[m]
        log.write("OOB\t%f\n" % oob_scores[m])
        log.write("# Importances:\n")
        for i in range(len(importances[m])):
            log.write("IMP\t%s\t%s\t%f\n" % (model[0],args['features'][i+1],importances[m][i]))

    if 'log' in args:
        log.close()  


def main():
    args = parse_args()
    run_models(args)

if __name__ == "__main__":
    main()


