from functools import reduce

class NaiveBayes:
    def __init__(self):
        self.neg_class = {}
        self.pos_class = {}

    def calculate_neg_prob(self,feature_dict):
        count_neg_words = reduce(lambda x,y: x+y, self.neg_class.values())
        count_pos_words = reduce(lambda x,y: x+y, self.pos_class.values())
        count_all_words = count_neg_words + count_pos_words
        pr_neg = count_neg_words / count_all_words
        pi_prob = 1
        #count_new_words = reduce(lambda x,y:x+y, feature_dict.values())
        for word in feature_dict:
            if word in self.pos_class:
                count_word_pos = self.pos_class[word]
            else:
                count_word_pos = 0
            if word in self.neg_class:
                count_word_neg = self.neg_class[word]
            else:
                count_word_neg = 0
            count_word = count_word_pos + count_word_neg # + feature_dict[word]
            pr_word = (count_word + 1) / (count_all_words + count_all_words)  # + count_new_words in denominator
            pr_word_given_neg = (count_word_neg + 1) / (count_neg_words + count_neg_words)
            pi_prob *= pr_word_given_neg / pr_word
        pi_prob *= pr_neg
        return pi_prob

    def calculate_pos_prob(self,feature_dict):
        count_neg_words = reduce(lambda x,y: x+y, self.neg_class.values())
        count_pos_words = reduce(lambda x,y: x+y, self.pos_class.values())
        count_all_words = count_neg_words + count_pos_words
        # print("count_neg_words:"+str(count_neg_words)+"\tcount_pos_words" +str(count_pos_words) + "\tcount_all_words:" + str(count_all_words))
        pr_pos = count_pos_words / count_all_words
        pi_prob = 1
        #count_new_words = reduce(lambda x,y:x+y, feature_dict.values())
        for word in feature_dict:
            if word in self.pos_class:
                count_word_pos = self.pos_class[word]
            else:
                count_word_pos = 0
            if word in self.neg_class:
                count_word_neg = self.neg_class[word]
            else:
                count_word_neg = 0
            print("count_all_words:" + str(count_all_words))
            count_word = count_word_pos + count_word_neg # + feature_dict[word]
            print("count_word(50):"+str(count_word))
            num = (count_word + 1)
            denom = (count_all_words + count_all_words)
            pr_word = (num / denom)  # + count_new_words in denominator
            print("pr_word(54):" + str(pr_word) + "\tnum:"+str(num)+"\tdenom:"+str(denom)+"\tpr_word:"+str(num/denom))
            print("test: "+ str(1/2000000))
            pr_word_given_pos = ( count_word_pos + 1) / (count_pos_words + count_pos_words)
            print("pi_prob:"+str(pi_prob)+"\tpr_word_given_pos:"+str(pr_word_given_pos)+"\tpr_word:"+str(pr_word))
            pi_prob = (pi_prob * pr_word_given_pos  / pr_word)
        pi_prob *= pr_pos
        return pi_prob

    def train(self, feature_dict, expected_result):
        if(expected_result == -1):
            for key in feature_dict:
                if key in self.neg_class:
                    self.neg_class[key] = self.neg_class[key] + 1
                else:
                    self.neg_class[key] = 1
        elif(expected_result ==1):
            for key in feature_dict:
                if key in self.pos_class:
                    self.pos_class[key] = self.pos_class[key] + 1
                else:
                    self.pos_class[key] = 1

    def classify(self,feature_dict):
        pr_pos = self.calculate_pos_prob(feature_dict)
        pr_neg = self.calculate_neg_prob(feature_dict)
        if(pr_pos > pr_neg):
            return 1
        else:
            return -1

    def reset(self):
        self.pos_class = {}
        self.neg_class = {}
