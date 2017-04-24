from sys import argv
import preprocessor as pp
import naivebayesModel as naivebayes
import copy
from sys import argv

##############################
# SET FLAGS
#############################
debug = 0
verbose = 0
freqOption = True
punctOption = True

for i in argv:
    if i == "-d":
        debug = 1
    if i == "-v":
        verbose = 1
    if i == "--punct":
        punctOption = True
    if i == "--nopunct":
        punctOption = False
    if i == "--frequency":
        freqOption = True
    if i == "--binary":
        freqOption = False

###########################
# INIT DATA
###########################
combined_data = pp.combined_data
if(debug):
    combined_data = combined_data[:len(combined_data)//4]

################################################################
#       TEST NAIVE BAYES
#################################################################
nb = naivebayes.NaiveBayes()
naive_bayes_accuracy_report = []

# PERFORM FIVE FOLD CROSS-VALIDATION
for i in range(5):
    nb.reset()

    # define partition for training, and testing.
    fifth_partition = len(combined_data)//5
    start_partition = i*fifth_partition
    end_partition = (i+1) * fifth_partition
    test_set = combined_data[start_partition:end_partition]
    train_set = combined_data[:start_partition] + combined_data[end_partition:]

    # debug print.
    if debug: print("\n\nTRAINING NAIVE BAYES:")
    if debug: print("______________________________________________________________________________________________")
    if debug: print("start_partition:" + str(start_partition))
    if debug: print("end_partition:" + str(end_partition))

    # train on training partition
    for train_tuple in train_set:
        training_file = train_tuple[0]
        expected_result = train_tuple[1]
        if(expected_result== -1):
            training_feature_vector = pp.master_features_dict_neg[training_file]
        elif(expected_result == 1):
            training_feature_vector = pp.master_features_dict_pos[training_file]
        nb.train(training_feature_vector,expected_result)
        if verbose: print("naive-bayes finished training on \t" + training_file)

    #######################################
    #   TEST ON TEST SET
    ######################################
    false_neg_count = 0
    false_pos_count = 0
    true_pos_count = 0
    true_neg_count = 0
    tested = 0
    for test_tup in test_set:
        test_file = test_tup[0]
        expected_result = test_tup[1]
        if(expected_result== -1):
            test_feature_vector = pp.master_features_dict_neg[test_file]
        elif(expected_result == 1):
            test_feature_vector = pp.master_features_dict_pos[test_file]
        result = nb.classify(test_feature_vector)
        tested = tested + 1
        #True Positives
        if expected_result==1 and result == 1:
            true_pos_count = true_pos_count + 1
        #False Positives
        elif expected_result== -1 and result ==1:
            false_pos_count = false_pos_count + 1
        #True Negatives
        elif expected_result== -1 and result == -1:
            true_neg_count = true_neg_count + 1
        #False Negatives.
        elif expected_result== 1 and result ==-1:
            false_neg_count = false_neg_count + 1
        #### Print Out Progress ##############
        if verbose:
            if result == test_tup[1]:
                output = "CORRECT"
            else:
                output = "WRONG"
            print("naive-bayes finished testing on \t" + test_tup[0] + "\tremaining files:" + str(len(test_set)- tested) + "\tRESULT:"+ output)

    #######################################
    #   STORE REPORT FOR TEST FOLD.
    ######################################
    # ADD ONE TO SMOOTH. WE CANNOT HAVE ZEROES, BECAUSE LATER DIVISION BY ZERO IN PERFORMANCE STATISTICS ANALYSIS.
    if(true_pos_count==0): true_pos_count+=1
    if(true_neg_count==0): true_neg_count+=1
    if(false_pos_count==0): false_pos_count+=1
    if(false_neg_count==0): false_neg_count+=1

    total_count = true_pos_count + true_neg_count + false_pos_count + false_neg_count
    test_summary = {'tp': true_pos_count, 'tn':true_neg_count,'fp':false_pos_count,'fn':false_neg_count,'total':total_count, 'test': (i+1)}
    naive_bayes_accuracy_report.append(test_summary)


##########################################
#  ELEMENTARY STATISTICS FOR ALL FOLDS
#########################################
if(debug):
    for i in naive_bayes_accuracy_report:
        if verbose : print("\n\n")
        if verbose: print("TEST NUMBER: " + str(i['test']))
        if verbose: print("_____________________________________________")
        if verbose: print("TRUE POSITIVES: " + str(i['tp'] / i['total'] * 100) + "%")
        if verbose: print("FALSE POSITIVES: " + str(i['fp'] / i['total'] * 100) + "%")
        if verbose: print("TRUE NEGATIVE: " + str(i['tn'] / i['total'] * 100)+ "%")
        if verbose: print("FALSE NEGATIVE: " + str(i['fn'] / i['total'] * 100)+ "%")
        if verbose: print("ACCURACY: " + str((i['tp'] + i['tn']) / i['total'] * 100)+ "%")

#############################################
#   PERFORMANCE MEASUREMENTS CALCULATIONS
#############################################
performance ={}
for summary in naive_bayes_accuracy_report:
    performance[summary['test']] = {
        'precision+' : summary['tp'] / (summary['tp'] + summary['fp']),
        'precision-' : summary['tn'] / (summary['tn'] + summary['fn']),
        'recall+' : summary['tp'] / (summary['tp'] + summary['fn']),
        'recall-' : summary['tn'] / (summary['tn'] + summary['fp']),
        'accuracy' : (summary['tp'] + summary['tn']) / (summary['tp'] + summary['tn'] + summary['fp'] + summary['fn'])
    }
    positive_precision = performance[summary['test']]['precision+']
    negative_precision = performance[summary['test']]['precision-']
    performance[summary['test']]['precision'] = (positive_precision + negative_precision) / 2

    positive_recall = performance[summary['test']]['recall+']
    negative_recall = performance[summary['test']]['recall-']
    performance[summary['test']]['recall'] = (positive_recall + negative_recall) / 2

print("\n\n-------------------------------------------")
print("\n\nINDIVIDUAL PERFORMANCE SUMMARY\n\n")
print("-------------------------------------------")
for summary in performance:
    print("\nTEST FOLD #" + str(summary))
    print("-----------------")
    print("Precision+ : \t" + str(performance[summary]['precision+'] *100) + "%")
    print("Recall+ : \t"+ str(performance[summary]['recall+'] *100) + "%")
    print("ACCURACY: \t" + str(performance[summary]['accuracy'] *100) + "%")
    print("PRECISION : \t" + str(performance[summary]['precision'] *100) + "%")
    print("RECALL : \t" +  str(performance[summary]['recall'] *100) + "%")


print("\n\n--------------------------------------------------")
print("\n\nAVERAGE OVERALL PERFORMANCE SUMMARY\n\n")
print("--------------------------------------------------")
avg_precision_plus = 0
avg_recall_plus = 0
avg_accuracy = 0
avg_precision = 0
avg_recall = 0

for summary in performance:
    avg_precision_plus += performance[summary]['precision+']
    avg_recall_plus += performance[summary]['recall+']
    avg_accuracy += performance[summary]['accuracy']
    avg_precision += performance[summary]['precision']
    avg_recall += performance[summary]['recall']

avg_precision_plus /= len(performance)
avg_recall_plus /= len(performance)
avg_accuracy /= len(performance)
avg_precision /= len(performance)
avg_recall /= len(performance)

print("AVERAGE PRECISION + : \t" + str( avg_precision_plus *100) + "%")
print("AVERAGE RECALL + : \t" + str( avg_recall_plus *100) + "%")
print("AVERAGE ACCURACY : \t" + str( avg_accuracy *100) + "%")
print("AVERAGE PRECISION : \t" + str( avg_precision *100) + "%")
print("AVERAGE RECALL : \t" + str( avg_recall *100) + "%")


#
# print("\n\nFOR PDF USAGE --------------")
# recall_plus = []
# recall = []
# precision_plus = []
# precision = []
# accuracy = []
# for fold in performance:
#     precision_plus.append(performance[fold]['precision+'])
#     recall_plus.append(performance[fold]['recall+'])
#     accuracy.append(performance[fold]['accuracy'])
#     precision.append(performance[fold]['precision'])
#     recall.append(performance[fold]['recall'])
#
# print("\n\n Accuracy  --------------")
# print("lowest accuracy : " + str(min(accuracy) * 100))
# print("highest accuracy : " + str(max(accuracy)* 100))
# print("range accuracy : " + str((max(accuracy)- min(accuracy))* 100))
#
# print("\n\n Precision Plus  --------------")
# print("lowest precision plus : " + str(min(precision_plus)* 100 ))
# print("highest precision plus : " + str(max(precision_plus)* 100))
# print("range precision plus : " + str(  (max(precision_plus) - min(precision_plus))* 100))
#
# print("\n\n Precision  --------------")
# print("lowest precision : " + str(min(precision) *100))
# print("highest precision : " + str(max(precision) * 100))
# print("range precision : " + str(  ( max(precision) - min(precision))  * 100 ) )
#
# print("\n\n Recall Plus  --------------")
# print("lowest recall plus: " + str(   min(recall_plus)  * 100)   )
# print("highest recall plus: " + str(   max(recall_plus)   * 100)  )
# print("range recall plus: " + str(   (max(recall_plus) - min(recall_plus))  * 100   ))
#
# print("\n\n Recall  --------------")
# print("lowest recall : " + str(min(recall) * 100))
# print("highest recall : " + str(max(recall) * 100))
# print("range recall : " + str((max(recall) - min(recall))* 100))
