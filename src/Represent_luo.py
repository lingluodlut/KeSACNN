'''
Created on 2014-10-14

@author: IRISBEST
'''
import os, sys
import numpy as np


class RepresentationLayer(object):
    
    
    def __init__(self, wordvec_file, POS_list=None, \
                 vec_size=50, frequency=10000, scale=1):
        
        '''
        vec_size        :    the dimension size of word vector 
                             learned by word2vec tool
        
        frequency       :    the threshold for the words left according to
                             their frequency appeared in the text
                             for example, when frequency is 10000, the most
                             frequent appeared 10000 words are considered
        
        scheme          :    the NER label scheme like IOB,IOBE, IOBES
                                     
        scale           :    the scaling for the vectors' each real value
                             when the vectors are scaled up it will accelerate
                             the training process
        
        vec_talbe        :    a matrix each row stands for a vector of a word

        index_map        :    the map from word to corresponding index in vec_table
        
        pos_2_index    :    the map from a POS to corresponding vector's index
        
        chunk_2_index    : the map from a Chunk to corresponding vector's index
        
        shape_2_index    : the map from a word's shape to corresponding vector's index

        orthograph_2_index    : the map from a word's orthograph to corresponding vector's index

        affix_2_index    : the map from a word's affix to corresponding vector's index

        terminology_2_index    : the map from a word's terminology to corresponding vector's index

        length_2_index    : the map from a word's length to corresponding vector's index

        semantic_mat    :    the matrix contains disease semantic information
        
        '''
        self.frequency = frequency
        self.vec_size = vec_size
        self.scale = scale
        
        self.vec_table = np.zeros((self.frequency + 1, self.vec_size))
        self.word_2_index = {}
        self.load_wordvecs(wordvec_file)
        

    
    
    def load_wordvecs(self, wordvec_file):
        
        file = open(wordvec_file)
        first_line = file.readline()
        word_count = int(first_line.split()[0])
        
        row = 0
        for line in file:
            if row < self.frequency:
                line_split = line[:-1].split()
                self.word_2_index[line_split[0]] = row
                for col in xrange(self.vec_size):
                    self.vec_table[row][col] = float(line_split[col + 1])
                row += 1
            else:
                break
        
        self.word_2_index['sparse_vectors'] = row
        sparse_vectors = np.zeros(self.vec_size)

        if word_count > self.frequency:
            for line in file:
                line_split = line[:-1].split()[1:]
                for i in xrange(self.vec_size):
                    sparse_vectors[i] += float(line_split[i])
            sparse_vectors /= (word_count - self.frequency)
        
        for col in xrange(self.vec_size):
            self.vec_table[row][col] = sparse_vectors[col]
        
     
        self.vec_table *= self.scale    
        
        file.close()

    


    def word_2_vec(self, word):
        if self.word_2_index.has_key(word):
            return self.vec_table[self.word_2_index[word]]
        else:
            return self.vec_table[self.word_2_index['sparse_vectors']]


    def labelindex_2_vec(self, label_index):
        if label_index == 0:
            return [1, 0, 0.]
        elif label_index == 1:
            return [0, 1, 0.]
        elif label_index == 2:
            return [0, 0, 1.]
        else:
            print 'Unexcepted label index'
            return None

    

    def indexs_2_labels(self, indexs):
        labels = []
        
        for index in indexs:
            labels.append(self.index_2_label(index))
        
        return labels

    def get_array_of_type2(self, type):
        if type == 'false':
            return [0.,1.]
        else:
            return [1.,0.]
    

    def represent_instance(self, instance):
        
        word_indexs = []
        splited = instance.split(' ')
        for word in splited[1:]:
            if self.word_2_index.has_key(word):
                word_indexs.append(self.word_2_index[word])
            else:
                word_indexs.append(self.word_2_index['sparse_vectors'])
        
      
        type = self.get_array_of_type2(splited[0])

        
        return type, word_indexs          



    '''
        represent instances that will pass to
        the recurrent neural networks.
        the input format are the word sequence
        and corresponding label sequence
    
    '''
    def represent_instances(self, instances, max_len=None):
        labels_list = []
        words_list = []
        
        for instance in instances:
            label_array, word_indexs = self.represent_instance(instance)
            labels_list.extend(label_array)
            if len(word_indexs) > max_len:
                words_list.extend(word_indexs[0:max_len])
            else:
                for i in range(max_len-len(word_indexs)):
                    word_indexs.append(0)
                words_list.extend(word_indexs)
        
        label_mat =  np.array(labels_list)
        label_mat = label_mat.reshape((len(instances), 2))
        
        x_mat = np.array(words_list)
        x_mat = x_mat.reshape((len(instances), max_len))
        
        return x_mat, label_mat



if __name__ == '__main__':
    pass
    
#    array_x, array_y = rep.generate_train_unit_rnn(test_x, test_y, 7)
#
#    array_3d = rep.generate_emb_based_data(array_x)


#    train = [line.strip() for line in open('C:/Users/IRISBEST/Desktop/CDR/data/test.txt')]
#    labels, words = rep.represent_instances_rnn(train, 'D')
#    for label, word in zip(labels, words):
#        print label
#        print word
#
#    train = [line.strip() for line in open('C:/Users/IRISBEST/Desktop/CDR/data/test.txt')]
    
#    labels_list = []
#    map = {'O_B':0,'O_I':0,'B_I':0,'B_O':0,'I_B':0,'I_O':0, 'I_I':0, 'B_B':0, 'O_O':0}
#    for elem in train:
#        id, labels, words = rep.represent_instance_D(elem)
#        for i in range(len(labels)-1):
#            if labels[i] == O and labels[i+1] == O:
#                map['O_O'] += 1
#            elif labels[i] == O and labels[i+1] == I:
#                map['O_I'] += 1
#            elif labels[i] == O and labels[i+1] == B:
#                map['O_B'] += 1
#            elif labels[i] == I and labels[i+1] == O:
#                map['I_O'] += 1
#            elif labels[i] == I and labels[i+1] == B:
#                map['I_B'] += 1
#            elif labels[i] == I and labels[i+1] == I:
#                map['I_I'] += 1
#            elif labels[i] == B and labels[i+1] == O:
#                map['B_O'] += 1
#            elif labels[i] == B and labels[i+1] == I:
#                map['B_I'] += 1
#            else:
#                map['B_B'] += 1
#    
#    print map
            
