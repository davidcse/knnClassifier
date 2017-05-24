import copy
import random
import math
import numpy
from functools import reduce

debug = 0


class CentroidModel:

    def __init__(self, distance_formula):
        self.distance_formula = distance_formula
        self.reset()

    # resets all variables except for the kValue
    def reset(self):
        # stores the positive and negative feature vec
        self.positiveFeatures = []
        self.negativeFeatures = []
        # produced matrices from the features vectors in training
        self.positiveMatrix = None
        self.negativeMatrix = None
        # produced from matrices, the centroids for each features.
        self.negativeCentroidVector = None
        self.positiveCentroidVector = None
        # produced from the
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
        # makes sure that all feature vectors are same length and have same keys.
        self.standardizeFeaturesFormat()
        # use the standardized feature vectors to build matrix of fixed length - width
        self.buildNegPosMatrices()
        # use the matrices to calculate centroid means.
        self.buildCentroids()


    # after all training vectors added, normalize all training vectors.
    def standardizeFeaturesFormat(self):
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


    def buildCentroids(self):
        # sum up all the columns axis of the negative matrices
        negativeCentroid = numpy.sum(self.negativeMatrix, axis=0)
        rowCount,columnCount = numpy.shape(self.negativeMatrix)
        negativeCentroid = negativeCentroid / rowCount

        # sum up all the columns axis of the positive matrices
        positiveCentroid = numpy.sum(self.positiveMatrix, axis=0)
        rowCount,columnCount = numpy.shape(self.positiveMatrix)
        positiveCentroid = positiveCentroid / rowCount

        # store the generated centroids
        self.negativeCentroid = negativeCentroid
        self.positiveCentroid = positiveCentroid



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




    def manhattan_distances(self,ordered_list):
        # compute the manhattan distances |(Ti - Xi)|
        neg_distance_list = self.negativeCentroid - ordered_list
        pos_distance_list = self.positiveCentroid - ordered_list
        # ge the absolute distance of the difference
        neg_distance_list = abs(neg_distance_list)
        pos_distance_list = abs(pos_distance_list)
        # sum row of list to get the total sum absolute
        negative_centroid_distance = numpy.sum(neg_distance_list, axis=1)
        positive_centroid_distance = numpy.sum(pos_distance_list, axis=1)
        if(debug) : print("neg centroid" + str(negative_centroid_distance))
        if(debug) : print("pos centroid" + str(positive_centroid_distance))
        return negative_centroid_distance, positive_centroid_distance


    def euclidean_distances(self,ordered_list):
        # compute the euclidean distances (Ti - Xi)^2
        neg_distance_list = self.negativeCentroid - ordered_list
        pos_distance_list = self.positiveCentroid - ordered_list
        neg_distance_list = neg_distance_list ** 2
        pos_distance_list = pos_distance_list ** 2
        # sum components to get the distance
        negative_centroid_distance = numpy.sum(neg_distance_list)
        positive_centroid_distance = numpy.sum(pos_distance_list)
        # sqrt final result
        negative_centroid_distance = numpy.sqrt(negative_centroid_distance)
        positive_centroid_distance = numpy.sqrt(positive_centroid_distance)
        if(debug) : print("neg centroid" + str(negative_centroid_distance))
        if(debug) : print("pos centroid" + str(positive_centroid_distance))
        return negative_centroid_distance, positive_centroid_distance



    def classify(self,test_feature):
        # add missing keys to test features.
        self.add_master_vec_keys(self.master_vec, test_feature)
        # create a list that complies with matrix order and format.
        ordered_list = []
        for key in self.master_vec_keys:
            ordered_list.append(test_feature[key])
        ordered_list = numpy.array(ordered_list)
        # choose the distance method to caculate
        if(self.distance_formula.strip().lower() == "m"):
            negative_centroid_distance, positive_centroid_distance = self.manhattan_distances(ordered_list)
        else:
            negative_centroid_distance, positive_centroid_distance = self.euclidean_distances(ordered_list)
        # get the nearest neighbor based on the distance
        return self.get_centroid_class(negative_centroid_distance,positive_centroid_distance)


    # checks which centroid is closer to the test feature.
    def get_centroid_class(self,neg_dist, pos_dist):
        if neg_dist > pos_dist :
            return -1
        elif neg_dist < pos_dist:
            return 1
        else:
            # equally likely, return random
            prob = random.uniform(0,1)
            return 1 if prob > 0.5 else -1
