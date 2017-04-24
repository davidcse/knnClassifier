import copy
from sys import argv
import preprocessor as pp
from knnModel import KnnModel

##############################
# SET FLAGS
#############################
debug = 0
verbose = 0
freqOption = True
punctOption = True
kValue  = 3 # default K-value

for i in argv:
    if i == "-d":
        debug = 1
    elif i == "-v":
        verbose = 1
    elif i == "--punct":
        punctOption = True
    elif i == "--nopunct":
        punctOption = False
    elif i == "--frequency":
        freqOption = True
    elif i == "--binary":
        freqOption = False


# trains the knnClassifier on all the training set tuples, pulling feature vectors out of the master_feature dictionaries.
def train_knn_classifer_allfiles(knnClassifier, train_set, master_features_dict_neg, master_features_dict_pos):
    knnClassifier.reset()
    finished = 0
    for train_tuple in train_set:
        # extract tuple elements
        training_file,expected_result = train_tuple[0], train_tuple[1]
        # pull the training_file out of the preprocessed dictionary of features.
        # the labeling of pos(1) or neg(-1) tells you which preprocessed dict to look in.
        if(expected_result == -1):
            training_feature_vector = master_features_dict_neg[training_file]
        elif(expected_result == 1):
            training_feature_vector = master_features_dict_pos[training_file]
        # train on the feature vector and continue.
        knnClassifier.train(training_feature_vector, expected_result)
        finished = finished + 1
        if verbose: print("knnClassifier finished training on \t" + training_file + "\tremaining files:" + str(len(train_set)- finished))
    # ready knnClassifier after training
    knnClassifier.readyTest()

# tests the knnClassifier on all the training set tuples, pulling feature vectors out of the master_feature dictionaries.
def test_knn_classifier_allfiles(knnClassifier, test_set, master_features_dict_neg,  master_features_dict_pos):
    # Test Report variables
    false_neg_count = 0
    false_pos_count = 0
    true_pos_count = 0
    true_neg_count = 0
    # Test the test set.
    tested = 0
    for test_tup in test_set:
        # extract tuple elements
        test_file,expected_result = test_tup[0], test_tup[1]
        # pull the test_file out of the preprocessed dictionary of features.
        # the labeling of pos(1) or neg(-1) tells you which preprocessed dict to look in.
        if(expected_result == -1):
            test_feature_vector = master_features_dict_neg[test_file]
        else:
            test_feature_vector = master_features_dict_pos[test_file]
        # TEST OUR KNN'S VIEW OF THIS TEST FEATURE SET.
        result = knnClassifier.classify(test_feature_vector)

        # ADD TO THE COUNTS FOR THIS RUN.
        #True Positives
        if((expected_result == 1) and (result == 1)): true_pos_count += 1
        #False Positives
        elif( (expected_result== -1) and (result == 1)): false_pos_count += 1
        #True Negatives
        elif((expected_result == -1) and (result == -1)): true_neg_count += 1
        #False Negatives.
        else: false_neg_count += 1
        # VERBOSE PRINTOUT ON PROGRESS
        tested = tested + 1
        output = "CORRECT" if(result==expected_result) else "WRONG"
        prediction = "POS" if result>0 else "NEG"
        if verbose: print("knn tested on \t" + test_file + "\tremaining files:" + str(len(test_set)- tested) + "\tRESULT:"+ output + " PREDICT:" + str(prediction))
    return test_statistics_dict(true_pos_count, true_neg_count, false_pos_count, false_neg_count)


def test_statistics_dict(true_pos_count,true_neg_count,false_pos_count,false_neg_count):
    total_count = true_pos_count + true_neg_count + false_pos_count + false_neg_count
    test_summary = {
        'tp': true_pos_count,
        'tn':true_neg_count,
        'fp':false_pos_count,
        'fn':false_neg_count,
        'total':total_count
    }
    return test_summary


def analyze_performance(accuracy_report, performance):
    for stat_dict in accuracy_report:
        if(debug): print(stat_dict)
        testFold = stat_dict.get("test")
        performance.update({
                testFold:{
                    'precision+' : stat_dict['tp'] / (stat_dict['tp'] + stat_dict['fp']),
                    'precision-' : stat_dict['tn'] / (stat_dict['tn'] + stat_dict['fn']),
                    'recall+' : stat_dict['tp'] / (stat_dict['tp'] + stat_dict['fn']),
                    'recall-' : stat_dict['tn'] / (stat_dict['tn'] + stat_dict['fp']),
                    'accuracy' : (stat_dict['tp'] + stat_dict['tn']) / (stat_dict['tp'] + stat_dict['tn'] + stat_dict['fp'] + stat_dict['fn'])
                }
        })
        positive_precision = performance.get(testFold).get('precision+')
        negative_precision = performance.get(testFold).get('precision-')
        performance[testFold]['precision'] = (positive_precision + negative_precision) / 2
        positive_recall = performance[testFold]['recall+']
        negative_recall = performance[testFold]['recall-']
        performance[testFold]['recall'] = (positive_recall + negative_recall) / 2




# takes the performance dict, and returns  dict containing avg
# of all performance statistics variables.
def analyze_average_performance(performance):
    # average performance variables
    avg_precision_plus = 0
    avg_recall_plus = 0
    avg_accuracy = 0
    avg_precision = 0
    avg_recall = 0
    # get aggregates for all folds in performance dict.
    for fold in performance:
        avg_precision_plus += performance[fold]['precision+']
        avg_recall_plus += performance[fold]['recall+']
        avg_accuracy += performance[fold]['accuracy']
        avg_precision += performance[fold]['precision']
        avg_recall += performance[fold]['recall']
    # normalize for average.
    avg_precision_plus /= len(performance)
    avg_recall_plus /= len(performance)
    avg_accuracy /= len(performance)
    avg_precision /= len(performance)
    avg_recall /= len(performance)
    # store avg statistics in dict
    return {
        "avg_precision_plus" : avg_precision_plus,
        "avg_recall_plus" : avg_recall_plus,
        "avg_accuracy" : avg_accuracy,
        "avg_precision" : avg_precision,
        "avg_recall" : avg_recall
    }

def print_individual_performance(performance):
    print("\n\n-------------------------------------------")
    print("\n\nINDIVIDUAL PERFORMANCE SUMMARY\n\n")
    print("-------------------------------------------")
    for fold in performance:
        print("\nTEST FOLD #" + str(fold))
        print("-----------------")
        print("Precision+ : \t" + str(performance[fold]['precision+'] *100) + "%")
        print("Recall+ : \t"+ str(performance[fold]['recall+'] *100) + "%")
        print("ACCURACY: \t" + str(performance[fold]['accuracy'] *100) + "%")
        print("PRECISION : \t" + str(performance[fold]['precision'] *100) + "%")
        print("RECALL : \t" +  str(performance[fold]['recall'] *100) + "%")


def print_accuracy_variables(accuracy_report):
    for stat_dict in accuracy_report:
        if verbose: print("TEST NUMBER: " + str(stat_dict['test']))
        if verbose: print("_____________________________________________")
        if verbose: print("TRUE POSITIVES: " + str(stat_dict['tp'] / stat_dict['total'] * 100) + "%")
        if verbose: print("FALSE POSITIVES: " + str(stat_dict['fp'] / stat_dict['total'] * 100) + "%")
        if verbose: print("TRUE NEGATIVE: " + str(stat_dict['tn'] / stat_dict['total'] * 100)+ "%")
        if verbose: print("FALSE NEGATIVE: " + str(stat_dict['fn'] / stat_dict['total'] * 100)+ "%")
        if verbose: print("ACCURACY: " + str((stat_dict['tp'] + stat_dict['tn']) / stat_dict['total'] * 100)+ "%")



def print_avg_performance(avg_performance):
    print("\n\n--------------------------------------------------")
    print("\n\nAVERAGE OVERALL PERFORMANCE SUMMARY\n\n")
    print("--------------------------------------------------")
    print("AVERAGE PRECISION + : \t" + str(avg_performance.get("avg_precision_plus") *100) + "%")
    print("AVERAGE RECALL + : \t" + str(avg_performance.get("avg_recall_plus") *100) + "%")
    print("AVERAGE ACCURACY : \t" + str(avg_performance.get("avg_accuracy") *100) + "%")
    print("AVERAGE PRECISION : \t" + str(avg_performance.get("avg_precision") *100) + "%")
    print("AVERAGE RECALL : \t" + str(avg_performance.get("avg_recall") *100) + "%")



###########################
# INIT DATA
###########################
combined_data = pp.combined_data
# DECREASE TRAINING DATA SIZE DURING TESTING DEV.
if(debug):
    combined_data = combined_data[:len(combined_data)//4]
if(debug and verbose):
    for i in combined_data:
        print(i)

################################################
#       TEST KNN
################################################
accuracy_report = []
knnClassifier = KnnModel(kValue)

# TEST WITH FIVE-FOLD CROSS-VALIDATION
for foldIteration in range(5):
    fifth_partition = len(combined_data)//5
    start_partition = foldIteration*fifth_partition
    end_partition = (foldIteration+1) * fifth_partition
    if verbose: print("\n\nTRAINING:")
    if verbose: print("______________________________________________________________________________________________")
    if verbose: print("start_partition:" + str(start_partition))
    if verbose: print("end_partition:" + str(end_partition))
    test_set = combined_data[start_partition:end_partition]
    train_set = combined_data[:start_partition] + combined_data[end_partition:]
    # give the list of training file tuples to be trained
    # and the master features dict, to get file vectors for the training file.
    train_knn_classifer_allfiles(knnClassifier, train_set, pp.master_features_dict_neg, pp.master_features_dict_pos)
    # give the list of test file tuples to be tested
    # and the master features dict, to get file vectors for the test file.
    test_summary = test_knn_classifier_allfiles(knnClassifier, test_set, pp.master_features_dict_neg, pp.master_features_dict_pos)
    test_summary.update({"test" : foldIteration + 1 })
    # store the statistics report.
    accuracy_report.append(test_summary)

##########################################
#  ELEMENTARY STATISTICS FOR ALL FOLDS
#########################################
print_accuracy_variables(accuracy_report)

#############################################
#   PERFORMANCE MEASUREMENTS CALCULATIONS
#############################################
performance ={}
analyze_performance(accuracy_report, performance)
print_individual_performance(performance)
avg_performance = analyze_average_performance(performance)
print_avg_performance(avg_performance)
