#python tester.py <path_to_testing_data> <path_to_classifier> <path_to_results>

#################################################################
#
# Write for COMP9318 project
# tester.py is used to predict a word is title or not
# modified by Nijie Sun from ipython Notebooks examples in COMP9318 website
#
##################################################################

import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold

import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

from itertools import tee, islice, chain
#%matplotlib inline


# add features into original data
def add_features(obj, is_tester=True):
    def previous_and_next(o):
        empty_ = [None]
        pre2_, pre1_, item_, next1_, next2_ = tee(o, 5)
        pre2_ = chain(empty_, empty_, pre2_)
        pre1_ = chain(empty_, pre1_)
        next1_ = chain(islice(next1_, 1, None), empty_)
        next2_ = chain(islice(next2_, 2, None), empty_, empty_)
        return zip(pre2_, pre1_, item_, next1_, next2_)

    title_list = ['Abbess', 'Abbot', 'Acolyte', 'Admiral', 'Advocate', 'Agent', 'Alderman', 'Almoner', 'Ambassador',
                      'Archdeacon', 'Archduchess', 'Archduke', 'Attaché', 'Attorney', 'Bailiff', 'Baron', 'Baroness',
                      'Barrister', 'Bishop', 'Blessed', 'Brigadier', 'Brother', 'Burgess', 'CAO', 'CCO', 'CDO', 'CEO',
                      'CFO', 'CIO', 'CMO', 'CNO', 'COO', 'CRO', 'CSO', 'CTO', 'Canon', 'Captain', 'Cardinal',
                      'Catholicos', 'Chairman', 'Chancellor', 'Chaplain', 'Chevalier', 'Chief', 'Christ', 'Coach',
                      'Colonel', 'Commander', 'Commodore', 'Comrade', 'Constable', 'Councillor', 'Count', 'Countess',
                      'Dame', 'Deacon', 'Dean', 'Delegate', 'Director', 'Doctor', 'Dom', 'Dr.', 'Duchess', 'Duke',
                      'Earl', 'Elder', 'Emperor', 'Empress', 'Envoy', 'Father', 'Foreman', 'Fr.', 'Friar', 'General',
                      'Governor', 'Hon', 'Judge', 'Justice', 'King', 'Lady', 'Leader', 'Lictor', 'Lieutenant', 'Lord',
                      'MP', 'MYP', 'Madam', 'Madame', 'Magistrate', 'Maid', 'Major', 'Marchioness', 'Marquess',
                      'Marquis', 'Marquise', 'Master', 'Mayor', 'Mayoress', 'Member', 'Minister', 'Miss', 'Monsignor',
                      'Mother', 'Mr', 'Mrs', 'Ms', 'Mufti', 'Mx', 'Officer', 'PM', 'Pastor', 'Patriarch', 'Pope',
                      'Popess', 'Prefect', 'Prelate', 'Premier', 'Presbyter', 'President', 'Priest', 'Priestess',
                      'Primate', 'Prince', 'Princess', 'Principal', 'Private', 'Prof.', 'Professor', 'Promagistrate',
                      'Provost', 'Queen', 'Reader', 'Reeve', 'Referee', 'Representative', 'Saint', 'Secretary',
                      'Selectman', 'Senator', 'Seneschal', 'Sergeant', 'Sir', 'Sister', 'Solicitor', 'Speaker',
                      'Superintendent', 'Supervisor', 'Ter', 'Treasurer', 'Tribune', 'Tsar', 'Tsarina', 'Tsaritsa',
                      'Umpire', 'Venerable', 'Vicar', 'Viscount', 'Viscountess', 'abbess', 'abbot', 'acolyte',
                      'admiral', 'advocate', 'agent', 'alderman', 'almoner', 'ambassador', 'archdeacon', 'archduchess',
                      'archduke', 'attorney', 'bailiff', 'baron', 'baroness', 'barrister', 'bishop', 'brigadier',
                      'burgess', 'captain', 'cardinal', 'catholicos', 'chairman', 'chancellor', 'chaplain', 'chevalier',
                      'colonel', 'commander', 'commodore', 'comrade', 'constable', 'councillor', 'dame', 'deacon',
                      'dean', 'delegate', 'director', 'doctor', 'dom', 'duchess', 'duke', 'earl', 'emperor', 'empress',
                      'envoy', 'foreman', 'friar', 'governor', 'lictor', 'lieutenant', 'lord', 'magistrate', 'maid',
                      'marchioness', 'marquess', 'marquis', 'marquise', 'mayoress', 'minister', 'monsignor', 'mufti',
                      'pastor', 'patriarch', 'pope', 'popess', 'prelate', 'premier', 'presbyter', 'president', 'priest',
                      'priestess', 'primate', 'prince', 'princess', 'principal', 'professor', 'promagistrate',
                      'provost', 'queen', 'reeve', 'representative', 'secretary', 'selectman', 'senator', 'seneschal',
                      'sergeant', 'solicitor', 'superintendent', 'supervisor', 'treasurer', 'tribune', 'tsarina',
                      'tsaritsa', 'umpire', 'venerable', 'vicar', 'viscount', 'viscountess']
    title_list1 = ['Abbess', 'Abbot', 'Acolyte', 'Administrator', 'Admiral', 'Advocate', 'Agent', 'Alderman', 'Almoner',
                   'Ambassador', 'Analyst', 'Archbishop', 'Archdeacon', 'Archduchess', 'Archduke', 'Assemblywoman',
                   'Assistant', 'Associate', 'Attaché', 'Attorney', 'Atty.', 'Bailiff', 'Baron', 'Baroness',
                   'Barrister', 'Billionaire', 'Bishop', 'Blessed', 'Brigadier', 'Brother', 'Burgess', 'CAO', 'CCO',
                   'CDO', 'CEO', 'CFO', 'CIO', 'CMO', 'CNO', 'COO', 'CRO', 'CSO', 'CTO', 'Cabinet', 'Campaign', 'Canon',
                   'Captain', 'Cardinal', 'Catholicos', 'Chair', 'Chairman', 'Chairperson', 'Chairwoman', 'Chancellor',
                   'Chaplain', 'Chevalier', 'Chief', 'Christ', 'Coach', 'Colonel', 'Columbia', 'Commander',
                   'Commissioner', 'Commodore', 'Comrade', 'Congregation', 'Congressman', 'Constable', 'Councillor',
                   'Counsel', 'Count', 'Countess', 'Crown', 'Culture', 'Dalai', 'Dame', 'Deacon', 'Dean', 'Delegate',
                   'Democrat', 'Deputy', 'Dikgang', 'Director', 'Directorate', 'District', 'Doctor', 'Dom', 'Dr.',
                   'Duchess', 'Duke', 'Earl', 'Editor', 'Elder', 'Emperor', 'Empress', 'Envoy', 'Executive', 'Father',
                   'Fellow', 'Foreman', 'Fr.', 'Friar', 'Gen', 'Gen.', 'General', 'Goodluck', 'Gov.', 'Governor', 'Hon',
                   'Honorary', 'Internist', 'Judge', 'Justice', 'King', 'Labor', 'Lady', 'Lama', 'Leader',
                   'Legislative', 'Lictor', 'Lieutenant', 'Lifetime', 'Lord', 'MEP', 'MEPs', 'MP', 'MPs', 'MYP',
                   'Madam', 'Madame', 'Magistrate', 'Maid', 'Major', 'Manager', 'Managing', 'Marchioness', 'Marquess',
                   'Marquis', 'Marquise', 'Master', 'Mathilde', 'Mayor', 'Mayoress', 'Member', 'Minister', 'Ministers',
                   'Ministership', 'Miss', 'Monsignor', 'Moseneke', 'Mother', 'Mr', 'Mr.', 'Mrs', 'Mrs.', 'Ms', 'Mufti',
                   'Mx', 'NCPO', 'Officer', 'Officers', 'Oilman', 'Olympian', 'Operating', 'Opposition', 'Ordinance',
                   'PM', 'Parliamentary', 'Pastor', 'Patriarch', 'Physician', 'Pier', 'Plenipotentiary', 'Pope',
                   'Popess', 'Prefect', 'Prelate', 'Premier', 'Presbyter', 'Presidency', 'President', 'Presidential',
                   'Presidents', 'Priest', 'Priestess', 'Primate', 'Prime', 'Prince', 'Princess', 'Principal',
                   'Private', 'Prof.', 'Professor', 'Project', 'Promagistrate', 'Prophet', 'Prosecutions', 'Prosecutor',
                   'Provincial', 'Provost', 'Queen', 'Reader', 'Rector', 'Reeve', 'Referee', 'Rep.', 'Representative',
                   'Royal', 'Rt', 'Saint', 'Science', 'Second', 'Secretary', 'Selectman', 'Sen.', 'Senator',
                   'Seneschal', 'Senior', 'Sergeant', 'Shadow', 'Shaikh', 'Sheikh', 'Sir', 'Sister', 'Solicitor',
                   'Speaker', 'Special', 'Sportswear', 'Staff', 'Sultan', 'Superintendent', 'Supervisor', 'Supreme',
                   'Ter', 'Tigre', 'Treasurer', 'Tribune', 'Tsar', 'Tsarina', 'Tsaritsa', 'Typhoon', 'Umpire', 'Uncle',
                   'Venerable', 'Vicar', 'Vice', 'Viscount', 'Viscountess', 'Whip', 'abbess', 'abbot', 'acolyte',
                   'administration', 'admiral', 'advocate', 'aide', 'alderman', 'almoner', 'ambassador', 'ambassadors',
                   'analyst', 'announcer', 'archbishop', 'archdeacon', 'archduchess', 'archduke', 'architect',
                   'assemblyman', 'assistant', 'associate', 'attorney', 'attorneys', 'author', 'backer', 'bailiff',
                   'baron', 'baroness', 'barrister', 'bishop', 'bishops', 'blogger', 'brigadier', 'burgess',
                   'businesswoman', 'cabinet', 'campaign', 'candidate', 'captain', 'cardinal', 'catholicos', 'chairman',
                   'chairperson', 'chairwoman', 'chancellor', 'chaplain', 'chevalier', 'chief', 'cleric', 'colonel',
                   'columnist', 'commander', 'commentator', 'commissioner', 'commodore', 'communications',
                   'competitors', 'comrade', 'congressman', 'constable', 'councillor', 'counsel', 'dame', 'deacon',
                   'dean', 'delegate', 'deputy', 'developer', 'director', 'directors', 'disciplinary', 'doctor', 'dom',
                   'duchess', 'duke', 'earl', 'economist', 'editor', 'emperor', 'empress', 'engineer', 'envoy',
                   'envoys', 'executive', 'fascist', 'foreman', 'friar', 'general', 'governor', 'industrialist',
                   'journalist', 'judge', 'king', 'lawyer', 'leader', 'lictor', 'lieutenant', 'lord', 'magistrate',
                   'maid', 'manager', 'marchioness', 'marquess', 'marquis', 'marquise', 'mayor', 'mayoralty',
                   'mayoress', 'minister', 'ministers', 'monsignor', 'mufti', 'negotiator', 'officer', 'painter',
                   'pastor', 'patriarch', 'pontiff', 'pope', 'popess', 'postmaster', 'practitioner', 'prelate',
                   'premier', 'presbyter', 'president', 'priest', 'priestess', 'primate', 'prince', 'princess',
                   'principal', 'producer', 'professor', 'promagistrate', 'prosecutor', 'provost', 'quarterback',
                   'queen', 'rabbi', 'rector', 'reeve', 'representative', 'scientist', 'screenwriter', 'secretary',
                   'selectman', 'senator', 'seneschal', 'sergeant', 'singer', 'solicitor', 'speaker', 'sultan',
                   'superintendent', 'supervisor', 'treasurer', 'tribune', 'tsarina', 'tsaritsa', 'umpire',
                   'valedictorian', 'venerable', 'vicar', 'viscount', 'viscountess', 'writer']
    # lemmatizer used to convert a plural word to singular mode, no change if it is already a singular word
    lemmatizer = WordNetLemmatizer()
    empty = ('$', '$', 'O')
    #empty = ('$', '$') if is_tester else ('$', '$', 'O')

    new_list = []
    for line in obj:
        new_line = []
        for pre2, pre1, item, next1, next2 in previous_and_next(line):
            pre2 = pre2 or empty
            pre1 = pre1 or empty
            next1 = next1 or empty
            next2 = next2 or empty

            new_item = []
            new_item.extend(item[:2])
            new_item.append(lemmatizer.lemmatize(item[0]))
            new_item.extend(pre1[:2])
            new_item.extend(pre2[:2])
            new_item.extend(next1[:2])
            new_item.extend(next2[:2])
            #new_item.append(pre1[1])
            #new_item.append(pre2[1])
            #new_item.append(next1[1])
            #new_item.append(next2[1])
            #new_item.extend(pre1[1:])
            #new_item.extend(pre2[1:])
            #new_item.extend(next1[1:])
            #new_item.extend(next2[1:])

            #new_item.append(
            #    1 if pre1[-1] == 'TITLE' and next1[-1] == 'TITLE' else 0)  # Upper and lower are both 'TITLE'
            #new_item.append(1 if pre1[-1] == 'TITLE' else 0) # previous is 'TITLE'
            #new_item.append(1 if next1[-1] == 'TITLE' else 0) # next is 'TITLE'
            new_item.append(1 if item[0][0].isupper() else 0)  # Word starts with uppercase
            #new_item.append(1 if item[0] in title_list or lemmatizer.lemmatize(item[0]) in title_list else 0)  # If it's in titlelist
            #new_item.append(1 if next1[0] in title_list or lemmatizer.lemmatize(next1[0]) in title_list else 0)  # If next1 is in titlelist
            #new_item.append(1 if next2[0] in title_list or lemmatizer.lemmatize(next2[0]) in title_list else 0)  # If next2 is in titlelist
            new_item.append(1 if item[0] in title_list or item[0][:-1] in title_list else 0)  # If it's in titlelist
            new_item.append(1 if next1[0] in title_list or next1[0][:-1] in title_list else 0)  # If next1 is in titlelist
            new_item.append(1 if next2[0] in title_list or next2[0][:-1] in title_list else 0)  # If next2 is in titlelist

            new_item.append(item[-1])

            #if (is_tester != True):
            #    new_item.append(item[-1])

            new_line.append(new_item)
        new_list.append(new_line)
        #print(new_line)
    return new_list

if __name__ == '__main__':
    if (4 != len(sys.argv)):
        sys.stderr.write('Usage:' + '\n')
        sys.stderr.write(sys.argv[0] + ' path_to_testing_data path_to_classifier path_to_results' + '\n')
        exit()

    # read test data
    data_path = sys.argv[1]
    with open(data_path, 'rb') as f:
        training_set = pickle.load(f)
    #print(training_set[0])

    # add features in test data
    training_set_features = add_features(training_set)
    #print(training_set_features[0])
    training_data = training_set_features[0][:]

    # convert line list [[line [word]]] to word list [[word]]
    for i in range(len(training_set_features)):
        if i != 0:
            training_data.extend(training_set_features[i])

    #df = pd.DataFrame(training_data,
    #                  columns=['token', 'tag', 'pre1tag', 'pre1result', 'pre2tag', 'pre2result', 'next1tag',
    #                           'next1result', 'next2tag', 'next2result', 'pre1isTITLE', 'next1isTITLE', 'uppercase', 'inTitlelist', 'result'])
    #df = pd.DataFrame(training_data,
    #                  columns=['token', 'tag', 'pre1tkoen', 'pre1tag', 'pre2tag', 'next1tag',
    #                            'next2tag', 'uppercase',
    #                           'inTitlelist', 'result'])
    df = pd.DataFrame(training_data,
                      columns=['token', 'tag',  'tokenissingular', 'pre1token', 'pre1tag', 'pre2token', 'pre2tag', 'next1token', 'next1tag', 'next2token', 'next2tag',
                               'uppercase',
                               'inTitlelist',
                               'next1inTitlelist', 'next2inTitlelist' ,
                               'result'
                               ])

    #print(df)

    X = df.drop('result', 1)

    # transform X to dictionary by use pandas to_dict()
    X = X.T.to_dict().values()

    # Verctirize X to vectors
    # read vectorizer create by trainer to make sure has same number of features
    # here use transform() instead of fix_transform() which used in trainer
    #vectorizer = DictVectorizer(dtype=float, sparse=True)
    #X = vectorizer.fit_transform(X)
    with open('./vectorizer.dat', 'rb') as vectorizer_f:
        vectorizer = pickle.load(vectorizer_f)
    X = vectorizer.transform(X)

    #y = df['result']
    # for b in y:
    #    print(b)
    # print(y)

    #encode 'O' 'TITLE' to '0' '1'
    #encoder = LabelEncoder()
    #y = encoder.fit_transform(y)

    # load the saved classifier
    classifier_path = sys.argv[2]
    with open(classifier_path, 'rb') as classifier_f:
        classifier = pickle.load(classifier_f)

    #print(classifier)
    #classifier.score(X, y)

    #  predict result
    predict_y = classifier.predict(X)
    #print(predict_y, ' ', len(predict_y))
    #for b in predict_y:
    #    print(b)

    # write predict result to result file
    path_to_results = sys.argv[3]
    with open(path_to_results, 'wb') as result_f:
        pickle.dump(predict_y, result_f)

    #with open(path_to_results, 'rb') as result_f:
    #    y_result = pickle.load(result_f)
    #print(y_result, ' ', len(y_result))
    #print(y)
    #f1_scorer = make_scorer(f1_score, pos_label="TITLE")
    #print("f1:", f1_score(y_true=y, y_pred=predict_y, pos_label=None, average='micro'))
    #print("f1:", f1_score(y_true=y, y_pred=predict_y, average='binary'))