'''
Created on 2014-10-27

@author: IRISBEST
'''
import numpy as np



positive = 1
negtive = 0

def change_real2class(real_res_matrix):
    res_matrix = np.zeros_like(real_res_matrix, dtype=int)
    max_indexs = np.argmax(real_res_matrix, 1)
    for i in xrange(len(max_indexs)):
        res_matrix[i][max_indexs[i]] = 1
        
    return res_matrix

def change_real2class_vec(real_res_vec):
    res_vec = np.zeros_like(real_res_vec, dtype=int)
    max_index = np.argmax(real_res_vec)
    res_vec[max_index] = 1
    return res_vec

def change_5class_2_2class(five_class_matrix):
    res_matrix = np.zeros((five_class_matrix.shape[0],2), dtype=int)
    max_indexs = np.argmax(five_class_matrix, 1)
    for i in xrange(len(max_indexs)):
        if max_indexs[i]  <= 3:
            res_matrix[i][0] = 1
        else:
            res_matrix[i][1] = 1
        
    return res_matrix
    
 
def eval_mulclass4(ans_matrix, res_matrix,print_lag=False, real=True):
   
    #db:[214,298,94,278] ml:[7,62,2,24] both:[221,360,96,302]
    if ans_matrix.shape[1] == 5 or ans_matrix.shape[1] == 4:
        positives = [221,360,96,302]
    else:
        positives = [979]

    
    confuse_matrixs = np.zeros((ans_matrix.shape[1], 4))
    
    if real == True:
        res_matrix = change_real2class(res_matrix)
#        res_matrix = change_5class_2_2class(res_matrix)
    
#    FileUtil.writeFloatMatrix(res_matrix, './step2/both.res')
    
    class_indexs = np.argmax(ans_matrix, 1)
    for class_index in range(confuse_matrixs.shape[0]):
        for i in range(ans_matrix.shape[0]):
            if np.allclose(ans_matrix[i], np.zeros(4)):
                class_indexs[i] = -1
            if class_index == class_indexs[i]: #positive entry
                if res_matrix[i][class_index] == positive:
                    confuse_matrixs[class_index][0] += 1 #TP
                else:
                    confuse_matrixs[class_index][1] += 1 #FN
            else: #negtive entry
                if res_matrix[i][class_index] == positive:
                    confuse_matrixs[class_index][2] += 1 #FP
                else:
                    confuse_matrixs[class_index][3] += 1 #TN

    
    P, R = .0, .0    
    for i in range(confuse_matrixs.shape[0]):
        print confuse_matrixs[i]
        p = confuse_matrixs[i][0]/(confuse_matrixs[i][0] + confuse_matrixs[i][2])
#        r = confuse_matrixs[i][0]/(confuse_matrixs[i][0] + confuse_matrixs[i][1] + loss_positive[i])
        r = confuse_matrixs[i][0]/(positives[i])
        P += p
        R += r
        print 'Evaluation for the ' + str(i + 1) + 'th class'
        print 'P:    ', p
        print 'R:    ', r
        print 'F1:    ', 2*p*r/(p+r)
        print        
    P /= (confuse_matrixs.shape[0])
    R /= (confuse_matrixs.shape[0])
    F1 = 2*P*R/(P+R)
    print 'Evaluation for all the class'
    print 'P:    ', P
    print 'R:    ', R
    print 'F1:    ', F1
    print
    
    return P,R,F1

def eval_mulclass(ans_matrix, res_matrix,print_flag=False, real=True):
   
  
    confuse_matrixs = np.zeros((ans_matrix.shape[1], 4))
    
    if real == True:
        res_matrix = change_real2class(res_matrix)
    
    class_indexs = np.argmax(ans_matrix, 1)
    for class_index in range(confuse_matrixs.shape[0]):
        for i in range(ans_matrix.shape[0]):
            if class_index == class_indexs[i]: #positive entry
                if res_matrix[i][class_index] == positive:
                    confuse_matrixs[class_index][0] += 1 #TP
                else:
                    confuse_matrixs[class_index][1] += 1 #FN
            else: #negtive entry
                if res_matrix[i][class_index] == positive:
                    confuse_matrixs[class_index][2] += 1 #FP
                else:
                    confuse_matrixs[class_index][3] += 1 #TN

    
    P, R = .0, .0    
    for i in range(confuse_matrixs.shape[0]-1):
        if print_flag==True:
            print
	    print confuse_matrixs[i]
        p = confuse_matrixs[i][0]/(confuse_matrixs[i][0] + confuse_matrixs[i][2])
        r = confuse_matrixs[i][0]/(confuse_matrixs[i][0] + confuse_matrixs[i][1])
        P += p
        R += r
         
    P /= (confuse_matrixs.shape[0]-1)
    R /= (confuse_matrixs.shape[0]-1)
    F1 = 2*P*R/(P+R)
    Acc=(confuse_matrixs[0][0]+confuse_matrixs[0][3])/(confuse_matrixs[0][0]+confuse_matrixs[0][1]+confuse_matrixs[0][2]+confuse_matrixs[0][3])
    if print_flag==True:
        print 'Evaluation for all the class'
        print 'P=%.5f, R=%.5f, F=%.5f, ACC=%.5f' % (P,R,F1,Acc)
        print
    
    return [P,R,F1,Acc]





if __name__ == '__main__':
    
    '''
        compute the extra filtered instances of test set
    '''
#    test_file = 'C:/Users/IRISBEST/Desktop/DDIs/DDICorpus/test.data'
#    test_set = [line.strip() for line in open(test_file)]
#    filtered = set()
#    id_list = []
#    for elem in test_set:
#        filtered.add(elem.split(' ')[0])    
#        id_list.append(elem.split(' ')[0])
#         
#    
#    test_file = 'C:/Users/IRISBEST/Desktop/DDIs/DDICorpus/test.raw.data'
#    test_set = [line.strip() for line in open(test_file)]
#    id_list2 = []
#    label_list = []
#    for elem in test_set:
#        id_list2.append(elem.split(' ')[0])
#        label_list.append(elem.split(' ')[1])
#    
#        
#    ans_list = []
#    for line in open('C:/Users/IRISBEST/Desktop/Major Revision/results/onestage.lr.0.0015.nounder.res'):
#        ans_list.append(line.strip())
#    
#    ans_map = {}
#    for id, ans in zip(id_list2, ans_list):
#        ans_map[id] = ans
#    
#    list = []
#    f1 = 0
#    f2 = 0
#    for id, label in zip(id_list2, label_list):
#        if id not in filtered:
#            list.append(label + '\t' + get_dditype_of_array5(change_real2class_vec(np.array(str_list_2_float_list(ans_map[id])))) +\
#                        '\t' + id)
#            if label == 'false':
#                f1 += 1
#            if get_dditype_of_array5(change_real2class_vec(np.array(str_list_2_float_list(ans_map[id])))) == 'false':
#                f2 += 1
#    print f1, f2
#    FileUtil.writeStrLines('C:/Users/IRISBEST/Desktop/Major Revision/results/comb.nounder.filtered.res', list)




    '''
        compute under-sampling result from by filtering the nounder-sampling result
        onestage
    '''
#    test_file = 'C:/Users/IRISBEST/Desktop/DDIs/DDICorpus/test.data'
#    test_set = [line.strip() for line in open(test_file)]
#    filtered = set()
#    id_list = []
#    for elem in test_set:
#        filtered.add(elem.split(' ')[0])    
#        id_list.append(elem.split(' ')[0])
#
#    test_file = 'C:/Users/IRISBEST/Desktop/DDIs/DDICorpus/test.raw.data'
#    test_set = [line.strip() for line in open(test_file)]
#    id_list2 = []
#    label_list = []
#    for elem in test_set:
#        id_list2.append(elem.split(' ')[0])
#        label_list.append(elem.split(' ')[1])
#    
#        
#    ans_list = []
#    for line in open('C:/Users/IRISBEST/Desktop/Major Revision/results/step1.raw.res'):
#        ans_list.append(line.strip())
#    
#    ans_map = {}
#    for id, ans in zip(id_list2, ans_list):
#        ans_map[id] = ans
#    
#    ans_list = []
#    for id in id_list:
#        ans_list.extend(str_list_2_float_list(ans_map[id]))
#    
#    ans_mat = np.array(ans_list)
#    ans_mat.resize((3055,2))
#    
#
#    test_file = 'C:/Users/IRISBEST/Desktop/DDIs/DDICorpus/test.data'
#    test_set = [line.strip() for line in open(test_file)]
#    lab_list = []
#    for elem in test_set:
#        lab_list.extend(get_array_of_dditype2(elem.split(' ')[1]))
#    lab_mat = np.array(lab_list)
#    lab_mat.resize((len(test_set),2))
#
#    FileUtil.writeFloatMatrix(ans_mat, 'C:/Users/IRISBEST/Desktop/Major Revision2/step1.raw.under.res')
#
#    eval_mulclass(lab_mat[:], ans_mat[:], True)



    '''
        compute under-sampling result from by filtering the nounder-sampling result
        twostage
    '''
#    test_file = 'C:/Users/IRISBEST/Desktop/DDIs/DDICorpus/test.raw.step2.data'
#    test_set = [line.strip() for line in open(test_file)]
#    lab_list = []
#    id_list = []
#    for elem in test_set:
#        lab_list.extend(get_array_of_dditype4(elem.split(' ')[1]))
#        id_list.append(elem.split(' ')[0])
#    lab_mat = np.array(lab_list)
#    lab_mat.resize((len(test_set),4))
#
#    ans_mat = loader.loadFullMatrix('C:/Users/IRISBEST/Desktop/Major Revision/results/twostage.step2.raw.res', 946, 4)
#
#    test_file = 'C:/Users/IRISBEST/Desktop/DDIs/DDICorpus/test.data'
#    test_set = [line.strip() for line in open(test_file)]
#    filtered = set()
#    for elem in test_set:
#        filtered.add(elem.split(' ')[0])    
#    
#    id_label_list = []
#    condition = np.zeros(len(id_list))
#    for i in range(len(id_list)):
#        if id_list[i] in filtered:
#            condition[i] = 1
#    
#    eval_mulclass(np.compress(condition,lab_mat, axis=0),\
#                  np.compress(condition,ans_mat, axis=0), True)


    '''
        compute normal
    '''

    test_file = 'C:/Users/IRISBEST/Desktop/DDIs/DDICorpus/test.data'
    test_set = [line.strip() for line in open(test_file)]
    lab_list = []
    for elem in test_set:
        lab_list.extend(get_array_of_dditype2(elem.split(' ')[1]))

#    test_file = 'C:/Users/IRISBEST/Desktop/Major Revision2/results/step2.raw.filtered'
#    test_set = [line.strip() for line in open(test_file)]
#    for elem in test_set:
#        lab_list.extend(get_array_of_dditype4('false'))
    
    lab_mat = np.array(lab_list)
    lab_mat.resize((len(lab_list)/2, 2))
#
    ans_mat1 = loader.loadFullMatrix('C:/Users/IRISBEST/Desktop/Major Revision2/results/step1.tfidf.res', 3055, 2)
#    ans_mat2 = loader.loadFullMatrix('C:/Users/IRISBEST/Desktop/Major Revision2/results/step2.raw.filtered', 76, 4)
    
#    eval_mulclass4(lab_mat[:], np.concatenate((ans_mat1, ans_mat2), axis=0))
    eval_mulclass(lab_mat, ans_mat1)




#    map = {'P1':'advise','P2':'effect', 'P3':'int', 'P4':'mechanism', 'N':'false'}
#    test_file = 'C:/Users/IRISBEST/Desktop/Major Revision2/final_result.result'
#    test_set = [line.strip() for line in open(test_file)]
#    ans_list = []
#    lab_list = []
#    for elem in test_set:
#        lab_list.extend(get_array_of_dditype2(map[elem.split('\t')[0]]))
#        ans_list.extend(get_array_of_dditype2(map[elem.split('\t')[1]]))
#
#    lab_mat = np.array(lab_list)
#    lab_mat.resize((len(lab_list)/2, 2))
#
#    ans_mat = np.array(ans_list)
#    ans_mat.resize((len(ans_list)/2, 2))
#
#    eval_mulclass(lab_mat, ans_mat)


    '''
        compute for the FPs generated by filtered
        
    '''


#    removed_id = set()
#    id_2_label = {}
#    for line in open('C:/Users/IRISBEST/Desktop/Major Revision2/results/test.filter.data'):
#        splited = line.split(' ')
#        removed_id.add(splited[0])
#        id_2_label[splited[0]] = splited[1]
#    
#    id_list = []
#    for line in open('C:/Users/IRISBEST/Desktop/Major Revision2/results/test.raw.step2.data'):
#        id_list.append(line.split(' ')[0])
#
#    res_list = []
#    for line in open('C:/Users/IRISBEST/Desktop/Major Revision2/results/step2.raw.res'):
#        splited = line.strip().split(' ')
#        array = np.array([float(elem) for elem in splited])
#        res_list.append(get_dditype_of_array4(array))
#    
#    assert len(id_list) == len(res_list)
#    
#    for id, res in zip(id_list, res_list):
#        if id in removed_id:
#            print id, res
            
    
    

    '''
        compute for the Uturku result
        
    '''

#    test_file = 'C:/Users/IRISBEST/Desktop/DDIs/DDICorpus/test.data'
#    test_set = [line.strip() for line in open(test_file)]
#    id_lab_list = []
#    lab_list = []
#    for elem in test_set:
#        id_lab_list.append(elem.split(' ')[0])
#        lab_list.extend(get_array_of_dditype2(elem.split(' ')[1]))
#
#    class_number = 2
#
#    lab_mat = np.array(lab_list)
#    lab_mat.resize((len(lab_list)/class_number, class_number))
#
#    id_2_label = {}
#    for line in open('C:/Users/IRISBEST/Desktop/Major Revision2/Uturku/test-pre.result'):
#        splited = line.strip().replace('x','p').split(' ')
#        id_2_label[splited[0]] = splited[3]
#    
#    res_list = []
#    for id in id_lab_list:
#        if id_2_label.has_key(id):
#            res_list.extend(get_array_of_dditype2(id_2_label[id]))
#        else:
#            res_list.extend(get_array_of_dditype2('false'))
#    
#    res_mat = np.array(res_list)
#    res_mat.resize((len(res_list)/class_number, class_number)) 
#    
#    eval_mulclass(lab_mat, res_mat)   
    
    

