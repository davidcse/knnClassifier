import copy
import random
import math
import numpy
from functools import reduce

debug = 0
verbose=1

class NotEnoughNeighborsError(Exception):
    pass

class KnnModel:

    def __init__(self, kValue, distance_formula):
        self.kValue = kValue
        self.distance_formula = distance_formula
        self.reset()

    # resets all variables except for the kValue
    def reset(self):
        self.positiveFeatures = []
        self.negativeFeatures = []
        self.positiveMatrix = None
        self.negativeMatrix = None
        self.master_vec_keys = None
        self.master_vec = None
        self.unseen_vectors = []


    # store the vectors for later use, depending on label
    def train(self, vector, result):
        if(result < 0):
            self.negativeFeatures.append(vector)
        elif(result > 0):
            self.positiveFeatures.append(vector)


    # prepare to test after training
    def readyTest(self):
        self.normalizeFeatures()
        self.buildNegPosMatrices()


    # after all training vectors added, normalize all training vectors.
    def normalizeFeatures(self):
        print("knn is standardizing size of feature vector...")
        # create a master array of all keys in both neg/pos training vec
        master_vec = self.build_master_vec()
        # add missing keys to all neg vec
        for neg_dict in self.negativeFeatures:
            self.add_master_vec_keys(master_vec,neg_dict)
        # add missing keys to all pos vec
        for pos_dict in self.positiveFeatures:
            self.add_master_vec_keys(master_vec,pos_dict)
        # set master keys for use later
        self.master_vec_keys = master_vec.keys()
        self.master_vec = master_vec



    # converts the normalized training features into numpy matrices
    def convertVectorsToMatrix(self, feature_set):
        matrix = []
        # convert all dictionaries in feature set into ordered 2-D matrix
        for dictionary in feature_set:
            # make sure it follows the order index of the master vector
            for key in self.master_vec_keys:
                matrix.append(dictionary[key])
        matrix = numpy.array(matrix)
        matrix = matrix.reshape(len(feature_set), len(self.master_vec_keys))
        return matrix


    def remove_inf_nan(self,matrix):
        # get rid of infinity and Nan values.
        nan_indices = numpy.isnan(matrix)
        matrix[nan_indices] = 0
        # get rid of infinity and Nan values.
        inf_indices = numpy.isinf(matrix)
        matrix[inf_indices] = 0
        return matrix



    def tf_idf(self,np_matrix):
        print("___________________________________________")
        print("starting tf_idf: Non-normalized matrix is ")
        print(np_matrix)
        print("___________________________________________")
        print("\nNormalized Matrix is ")
        # get (i,j) of matrix, i == number of rows (each row is a document), j is the word features length.
        document_count, vector_indices_count = numpy.shape(np_matrix)
        # sum across each row, to get total count of words in each document.
        sums_of_document_terms = np_matrix.sum(axis=1)
        # convert term count vector into a column matrix. [1,2,3,4] => [[1],[2],[3],[4]]
        sums_of_document_terms = sums_of_document_terms[:,None]
        # divide for each item in every row, by total number of term count for the row.
        tf_matrix = np_matrix / sums_of_document_terms
        # for each column j,representing particular word term,
        # get count of documents that it is seen in (nonzero).
        # same as number of rows that this column was non-zero.
        presence_count_for_words = numpy.count_nonzero(np_matrix,axis=0)
        # count(documents) / [count(documents_present(word))].
        #  Applies scalar division to a row. result is still a row
        idf_row = document_count / presence_count_for_words
        # log_e(x), numpy.log is base e by default.
        idf_row = numpy.log(idf_row)
        idf_row = self.remove_inf_nan(idf_row)
        np_matrix = tf_matrix * idf_row
        # get rid of infinity and Nan values.
        np_matrix = self.remove_inf_nan(np_matrix)
        # assert that dimensions of tf_idf normalization didn't change matrix shape.
        new_rows,new_columns = numpy.shape(np_matrix)
        assert(new_rows == document_count)
        assert(new_columns == vector_indices_count)
        print("done tf_idf")
        print(np_matrix)
        return np_matrix




    # conver the negative and positive training features into matrices.
    def buildNegPosMatrices(self):
        print("knn algorithm is building training vector matrix...")
        negativeMatrix = self.convertVectorsToMatrix(self.negativeFeatures)
        positiveMatrix = self.convertVectorsToMatrix(self.positiveFeatures)
        self.negativeMatrix = self.tf_idf(negativeMatrix)
        self.positiveMatrix = self.tf_idf(positiveMatrix)
        print("knn algorithm finished building training vector matrix.")




    # creates a master vector which contains all keys in all training vectors
    def build_master_vec(self):
        master_vec = {}
        for d in self.negativeFeatures:
            for key in d:
                if key not in master_vec:
                    master_vec[key] = 1
        for d in self.positiveFeatures:
            for key in d:
                if key not in master_vec:
                    master_vec[key]=1
        for d in self.unseen_vectors:
            for key in d:
                if key not in master_vec:
                    master_vec[key]=1
        return master_vec


    # normalizes a training vector to the size of the master vector.
    def add_master_vec_keys(self,master_vec, normal_vec):
        for mvKey in master_vec:
            if mvKey not in normal_vec:
                normal_vec[mvKey] = 0
        # during testing, test vector may contain
        # unseen words. By default the distance will be the size of the value.
        # store the aggregate distance generated by unseen words.
        unseen_words_distance = 0
        for key in normal_vec:
            if key not in master_vec:
                unseen_words_distance += math.pow(normal_vec[key],2)
        return unseen_words_distance



    def manhattan_distances(self,ordered_list):
        # compute the manhattan distances |(Ti - Xi)|
        neg_distance_matrix = self.negativeMatrix - ordered_list
        pos_distance_matrix = self.positiveMatrix - ordered_list
        # ge the absolute distance of the difference
        neg_distance_matrix = abs(neg_distance_matrix)
        pos_distance_matrix = abs(pos_distance_matrix)
        # sum rows of matrices to get the total sum absolute
        neg_distance_list = numpy.sum(neg_distance_matrix, axis=1)
        pos_distance_list = numpy.sum(pos_distance_matrix, axis=1)
        # make sure the pos/neg lists are equivalent to num training vec of both classes
        assert(len(neg_distance_list) == len(self.negativeFeatures))
        assert(len(pos_distance_list) == len(self.positiveFeatures))
        # sort the distance lists, lowest first
        neg_distance_list = sorted(neg_distance_list)
        pos_distance_list = sorted(pos_distance_list)
        if(debug) : print("neg " + str(neg_distance_list))
        if(debug) : print("pos " + str(pos_distance_list))
        return neg_distance_list, pos_distance_list


    def euclidean_distances(self,ordered_list):
        # compute the euclidean distances (Ti - Xi)^2
        neg_distance_matrix = self.negativeMatrix - ordered_list
        pos_distance_matrix = self.positiveMatrix - ordered_list
        neg_distance_matrix = neg_distance_matrix ** 2
        pos_distance_matrix = pos_distance_matrix ** 2
        # sum components to get the distance
        neg_distance_list = numpy.sum(neg_distance_matrix, axis=1)
        pos_distance_list = numpy.sum(pos_distance_matrix, axis=1)
        # sqrt final result
        neg_distance_list = numpy.sqrt(neg_distance_list)
        pos_distance_list = numpy.sqrt(pos_distance_list)
        # make sure the pos/neg lists are equivalent to num training vec of both classes
        assert(len(neg_distance_list) == len(self.negativeFeatures))
        assert(len(pos_distance_list) == len(self.positiveFeatures))
        # sort the distance lists
        neg_distance_list = sorted(neg_distance_list)
        pos_distance_list = sorted(pos_distance_list)
        if(debug) : print("neg " + str(neg_distance_list))
        if(debug) : print("pos " + str(pos_distance_list))
        return neg_distance_list, pos_distance_list



    def classify(self,test_feature):
        # add missing keys to test features.
        unseen_word_distance = self.add_master_vec_keys(self.master_vec, test_feature)
        # create a list that complies with matrix order and format.
        ordered_list = []
        for key in self.master_vec_keys:
            ordered_list.append(test_feature[key])
        ordered_list = numpy.array(ordered_list)
        # choose the distance method to caculate
        if(self.distance_formula.strip().lower() == "m"):
            neg_distance_list, pos_distance_list = self.manhattan_distances(ordered_list)
        else:
            neg_distance_list, pos_distance_list = self.euclidean_distances(ordered_list)
        # get the nearest neighbor based on the distance
        return self.get_knn_class(neg_distance_list,pos_distance_list,self.kValue)



    # checks both lists and see which has the nearest neighbor
    # assumes list is sorted with element being the distance.
    def nearest_neighbor(self,list1,list2):
        if(len(list1) > 0 and len(list2) > 0):
            if list1[0] < list2[0] :
                minElement, minList = list1.pop(0), 1
            else:
                minElement, minList = list2.pop(0), 2
        elif(len(list1) > 0 and len(list2)==0):
            minElement, minList = list1.pop(0), 1
        elif(len(list2) > 0 and len(list1)==0):
            minElement, minList = list2.pop(0), 2
        elif(len(list2)==0 and len(list1)==0):
            minElement, minList = None,None
        return minElement, minList

    # performs neareset neighbor analysis to find the closest class.
    def get_knn_class(self,neg_dist, pos_dist, kValue):
        negClassCount,posClassCount = 0,0
        for i in range(kValue):
            distance,neighborClass = self.nearest_neighbor(neg_dist,pos_dist)
            # nearest neighbor comes from first list
            if neighborClass == 1:
                negClassCount += 1 # increment the class seen
            # nearest neighbor comes from second list
            elif neighborClass == 2:
                posClassCount +=1 # increment the class seen
            else:
                raise NotEnoughNeighborsError
        if(verbose): print("K-neighbors:" + str(kValue) + " K-neg:"+str(negClassCount) + " K-pos:"+str(posClassCount))
        if negClassCount > posClassCount :
            return -1
        elif negClassCount < posClassCount:
            return 1
        else:
            # equally likely, return random
            prob = random.uniform(0,1)
            return 1 if prob > 0.5 else -1
