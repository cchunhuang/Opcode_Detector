import r2pipe
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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

def Vectorize(sequence, top_features_path="./top_features_1.npy"):
    seq = ' '.join(str(i) for i in sequence)  # Improved string concatenation
    X = [seq]
    
    vectorizer_count = CountVectorizer(ngram_range=(2, 4))
    x_count = vectorizer_count.fit_transform(X)
    
    feature_list = vectorizer_count.get_feature_names_out().tolist()  # Convert to list

    top_feature = np.load(top_features_path)
    tmp = []
    
    for j in range(len(top_feature)):
        if top_feature[j] in feature_list:
            tmp.append(x_count[:, feature_list.index(top_feature[j])][0, 0])
        else:
            tmp.append(0)
    
    tmp = np.asarray(tmp).reshape(1, -1)
    return tmp
