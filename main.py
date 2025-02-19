import numpy as np
import os
import xgboost
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import r2pipe
import joblib as jb
from utils import parameter_parser
from utils import write_output


def Extraction(filename):
    # OpenFile
    r2 = r2pipe.open(filename)
    # Analyze  
    r2.cmd('aaaa')
    # Extraction
    OpcodeSequence=[]
    DisassembleJ=r2.cmdj('pdj $s')
    if DisassembleJ:
        for instruction in DisassembleJ:
            try:
                if instruction['opcode'].split(' ')[0]:
                    if instruction['opcode'].split(' ')[0] != "invalid":
                        OpcodeSequence.append(instruction['opcode'].split(' ')[0])
            except:
                pass
    return OpcodeSequence

def vectorize(sequence):
    seq = ''
    for i in sequence:
        seq += str(i + ' ')
    X = list()
    X.append(seq)
    vectorizer_count = CountVectorizer(ngram_range = (2, 4))
    x_count = vectorizer_count.fit_transform(X)
    feature_list = vectorizer_count.get_feature_names()
    top_feature= np.load("./top_features_1.npy")
    tmp = []
    for j in range(len(top_feature)):
        if(top_feature[j] in feature_list):
            tmp.append(x_count[:,feature_list.index(top_feature[j])][0,0])
        else:
            tmp.append(0)
    tmp = np.asarray(tmp)
    tmp = tmp.reshape(1,-1)
    return(tmp)
    
    
def Predict(X,clf):
    '''
    param: X (feature vector)
    return: y (label), 1 for benign, 0 for malware
    '''

    model = jb.load('./MD_Model/' + clf)
    # label = model.predict_proba(X)
    # return label
    result = model.predict_proba(X).tolist()[0]
    return result    

def Predict_classification(X,clf):
    '''
    param: X (feature vector)
    return: y (label)
    '''
    #model = pickle.load('./FC_Model/SVM.pickle')
    model = jb.load('./FC_Model/' + clf)
    # label = model.predict_proba(X)
    # return label
    result = model.predict_proba(X).tolist()[0]
    return result

def main(args):
    result = [-1]
    
    # table = {0:'Mirai',1:'Bashlite',2:'Unknown',3:'Android',4:'Tsunami',5:'Hajime',6:'Dofloo',7:'Xorddos',8:'Pnscan',9:'BenignWare'}
    # table_1 = {0:'Malware',1:'Benignware'}
    labels = ['Malware','Benignware']
    if args.classify:
        labels = ['Mirai','Bashlite','Unknown','Android','Tsunami','Hajime','Dofloo','Xorddos','Pnscan','BenignWare']
    try:
        opcode_sequence = Extraction(args.input_path)
    except:
        print('fail to extract the Opcode.')
        print(result)
        write_output(args.input_path, args.output_path, result, labels)
        return result
    
    try:
        feature = vectorize(opcode_sequence)
    except:
        print('fail to extract the feature.')
        print(result)
        write_output(args.input_path, args.output_path, result, labels)
        return result


    if(args.classify == False):
        result = Predict(feature,args.model)
        # result = int(result[0])
        # print(table_1[result])
        # return table_1[result]
        print(result)
        write_output(args.input_path, args.output_path, result, labels)
        return result
    if(args.classify == True):
        result = Predict_classification(feature,args.model)
        # result = int(result[0])
        # print(table[result])
        # return table[result]
        print(result)
        write_output(args.input_path, args.output_path, result, labels)  
        return result


if __name__=='__main__':
    args = parameter_parser()
    main(args)