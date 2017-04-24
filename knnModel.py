import copy
import random
import math
import numpy
from functools import reduce

debug = 1

class NotEnoughNeighborsError(Exception):
    pass

class KnnModel:

    def __init__(self, kValue):
        self.kValue = kValue
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


    # conver the negative and positive training features into matrices.
    def buildNegPosMatrices(self):
        negativeMatrix = self.convertVectorsToMatrix(self.negativeFeatures)
        positiveMatrix = self.convertVectorsToMatrix(self.positiveFeatures)
        self.negativeMatrix = negativeMatrix
        self.positiveMatrix = positiveMatrix



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
        try:
            assert(len(master_vec)==len(normal_vec))
        except:
            # encountered unseen words, rebuild matrices
            self.unseen_vectors.append(normal_vec)
            self.readyTest()



    def classify(self,test_feature):
        # add missing keys to test features.
        self.add_master_vec_keys(self.master_vec, test_feature)

        # create a list that complies with matrix order and format.
        ordered_list = []
        for key in self.master_vec_keys:
            ordered_list.append(test_feature[key])
        ordered_list = numpy.array(ordered_list)

        # compute the euclidean distances (Ti - Xi)^2
        neg_distance_matrix = self.negativeMatrix - ordered_list
        pos_distance_matrix = self.positiveMatrix - ordered_list
        neg_distance_matrix = neg_distance_matrix ** 2
        pos_distance_matrix = pos_distance_matrix ** 2
        # sum components to get the distance
        neg_distance_list = numpy.sum(neg_distance_matrix, axis=1)
        pos_distance_list = numpy.sum(pos_distance_matrix, axis=1)
        # make sure the pos/neg lists are equivalent to num training vec of both classes
        assert(len(neg_distance_list) == len(self.negativeFeatures))
        assert(len(pos_distance_list) == len(self.positiveFeatures))
        # sort the distance lists
        neg_distance_list = sorted(neg_distance_list)
        pos_distance_list = sorted(pos_distance_list)
        if(debug) : print("neg " + str(neg_distance_list))
        if(debug) : print("pos " + str(pos_distance_list))
        return self.get_knn_class(neg_distance_list,pos_distance_list,self.kValue)

    # checks both lists and see which has the nearest neighbor
    # assumes list is sorted with element being the distance.
    def nearest_neighbor(self,list1,list2):
        if(len(list1) > 0 and len(list2) > 0):
            if list1[0] < list2[0] :
                minElement, minList = list1[0], 1
            else:
                minElement, minList = list2[0], 2
        elif(len(list1) > 0):
            minElement, minList = list1[0], 1
        elif(len(list2) > 0):
            minElement, minList = list2[0], 2
        elif(len(list2)==0 and len(list1)==0):
            minElement, minList = None,None
        return minElement, minList

    def get_knn_class(self,neg_dist, pos_dist, kValue):
        negClassCount,posClassCount = 0,0
        for i in range(kValue):
            distance,neighborClass = self.nearest_neighbor(neg_dist,pos_dist)
            # nearest neighbor comes from first list
            if neighborClass == 1:
                negClassCount += 1
            # nearest neighbor comes from second list
            elif neighborClass == 2:
                posClassCount +=1
            else:
                raise NotEnoughNeighborsError
        if negClassCount > posClassCount :
            return 0
        elif negClassCount < posClassCount:
            return 1
        else:
            # equally likely, return random
            prob = random.uniform(0,1)
            return 1 if prob > 0.5 else 0
