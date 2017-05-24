David Lin
Machine Learning


#######################
# SOFTWARE
#######################
python version 3
Numpy Modules for python

#######################
# RUN COMMANDLINE
######################

Name:
  Knn.py

Synopsis:
  python3 knn.py [--binary|--frequency] [--punct|--nopunct] [--k=<number>] [--metric=euclidean|manhattan]

Description Options :
    argument position 1 : --binary or --frequency
    During preprocessing, deciding whether to count the presence of the word in the document,
    or to count the frequency of how many times it showed up in the document.

    argument position 2 : --punct or --nopunct
    During preprocessing, deciding whether to count punctuations as a feature,
    or to remove the punctuations from the document before collecting feature extraction.

    argument position 3 : --k=<number>
    receives an integer for number of nearest neighbors to check for during execution of the
    k-nearest neighbors algorithm. Applicable during the testing phase.

    argument position 4: --metric=euclidean or --metric=manhattan
    During testing, deciding whether to use the euclidean distance formula or the
    manhattan distance formula to measure the distance between the test vector and all
    training vectors.

    argument position 5: Optional -v
    Print verbose trace of execution

    argument position 6: Optional -d
    Print debug which includes function state variables.


Name:
  centroid.py

Synposis:
  python3 centroid.py [--binary|--frequency] [--punct|--nopunct] [--metric=euclidean|manhattan] -v -d

Description Options :
  argument position 1 : --binary or --frequency
  During preprocessing, deciding whether to count the presence of the word in the document,
  or to count the frequency of how many times it showed up in the document.

  argument position 2 : --punct or --nopunct
  During preprocessing, deciding whether to count punctuations as a feature,
  or to remove the punctuations from the document before collecting feature extraction.

  argument position 3: --metric=euclidean or --metric=manhattan
  During testing phase, deciding whether to use the euclidean distance formula or the
  manhattan distance formula to measure the distance between the test vector and all
  training vectors.

  argument position 4: Optional -v
  Print verbose trace of execution

  argument position 5: Optional -d
  Print debug which includes function state variables.
