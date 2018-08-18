import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from sklearn import tree

# constants
real_file = "clean_real.txt"
fake_file = "clean_fake.txt"
figures_folder = "figures/"

def get_data(file):
    words = {}
    lines = []

    for line in open(file):
        lines.append(line)
        line_words = line.split()
        for w in line_words:
            if w in words:
                words[w] += 1
            else:
                words[w] = 1

    total = len(lines)
    validation_percent = (15 * total) / 100
    test_percent = (15 * total) / 100

    np.random.seed(0)
    np.random.shuffle(lines)
    np.random.shuffle(lines)
    np.random.shuffle(lines)

    return words, lines[:validation_percent], lines[validation_percent:validation_percent + test_percent],\
           lines[validation_percent + test_percent:]

def get_counts(lines, label, words_dict):
    for line in lines:
        line_words = list(set(line.split()))
        for w in line_words:
            if w not in words_dict:
                words_dict[w] = [0, 0]
            words_dict[w][label] += 1

    return words_dict


def get_prediction(p_real, p_fake, words_dict, lst, label, set_type):
    label_names = ["fake", "real"]
    p_headlines = []
    for line in lst:
        line_words = list(set(line.split()))
        p_line_real = 0.
        p_line_fake = 0.

        for word in words_dict:
            if word in line_words:
                p_line_real += np.log(words_dict[word][1])
                p_line_fake += np.log(words_dict[word][0])
            else:
                p_line_real += np.log(1. -  (words_dict[word][1]))
                p_line_fake += np.log(1. -  (words_dict[word][0]))

        p_line_real = np.exp(p_line_real)
        p_line_fake = np.exp(p_line_fake)
        p_real_line = (p_line_real * p_real) / ((p_line_real * p_real) + (p_line_fake * p_fake))
        p_fake_line = (p_line_fake * p_fake) / ((p_line_real * p_real) + (p_line_fake * p_fake))
        p_headlines.append((p_fake_line, p_real_line))

    results = [i.index(max(i)) for i in p_headlines]
    print set_type, ", for label:", label_names[label], \
        ", performance:", (results.count(label) / float(len(lst))) * 100, "%"
    return results

def get_label_given_word(p_real, p_fake, words_dict):
    label_given_word_dict = {}
    label_given_not_word_dict = {}
    p_line_real = 0.
    p_line_fake = 0.

    for curr_word in words_dict:
        for word in words_dict:
            if word == curr_word:
                p_line_real += np.log(words_dict[word][1])
                p_line_fake += np.log(words_dict[word][0])
            else:
                p_line_real += np.log(1. -  (words_dict[word][1]))
                p_line_fake += np.log(1. -  (words_dict[word][0]))

        p_line_real = np.exp(p_line_real)
        p_line_fake = np.exp(p_line_fake)

        label_given_word_dict[curr_word] = [0, 0]
        label_given_word_dict[curr_word][1] = (p_line_real * p_real) / ((p_line_real * p_real) + (p_line_fake * p_fake))
        label_given_word_dict[curr_word][0] = (p_line_fake * p_fake) / ((p_line_real * p_real) + (p_line_fake * p_fake))

        p_not_line_real = 1. - p_line_real
        p_not_line_fake = 1. - p_line_fake
        
        label_given_not_word_dict[curr_word] = [0, 0]
        label_given_not_word_dict[curr_word][1] = (p_not_line_real * p_real) / \
                                                  (1. - ((p_line_real * p_real) + (p_line_fake * p_fake)))
        label_given_not_word_dict[curr_word][0] = (p_not_line_fake * p_fake) / \
                                                  (1. - ((p_line_real * p_real) + (p_line_fake * p_fake)))


    return label_given_word_dict, label_given_not_word_dict

def part1_2_3():

#    part 1, split data
    real_words, real_validation, real_test, real_train = get_data(real_file)
    fake_words, fake_validation, fake_test, fake_train = get_data(fake_file)

#   used for part 1 examples
#    sorted_real_words = sorted(real_words.iteritems(), key=lambda (k,v): (v,k), reverse=True)
#    sorted_fake_words = sorted(fake_words.iteritems(), key=lambda (k,v): (v,k), reverse=True)

#   part 2, NB classifier training
    total = float(len(real_train) + len(fake_train))
    p_real = len(real_train) / total
    p_fake = len(fake_train) / total



    m = 1.

    #for i in range(5):
    p = 0.03
    #    m += 1.
    #    for j in range(10):
    #        p += 0.01
    train_dict = {}
    train_dict = get_counts(real_train, 1, train_dict)
    train_dict = get_counts(fake_train, 0, train_dict)

    for k, v in train_dict.items():
        train_dict[k][0] = (train_dict[k][0] + m * p) / (float(len(fake_train)) + m)
        train_dict[k][1] = (train_dict[k][1] + m * p) / (float(len(real_train)) + m)

#   get performance
#    print "m:", m, ",p:", p
    training_real_results = get_prediction(p_real, p_fake, train_dict, real_train, 1, "training")
    training_fake_results = get_prediction(p_real, p_fake, train_dict, fake_train, 0, "training")
    validation_real_results = get_prediction(p_real, p_fake, train_dict, real_validation, 1, "validation")
    validation_fake_results = get_prediction(p_real, p_fake, train_dict, fake_validation, 0, "validation")
    test_real_results = get_prediction(p_real, p_fake, train_dict, real_test, 1, "test")
    test_fake_results = get_prediction(p_real, p_fake, train_dict, fake_test, 0, "test")
    print "\n"

    print_NB_performance(training_real_results, training_fake_results, validation_real_results
                         ,validation_fake_results, test_real_results, test_fake_results)
    
#   part 3:
    label_given_word_dict, label_given_not_word_dict = get_label_given_word(p_real, p_fake, train_dict)

    presence_real = sorted(label_given_word_dict.items(), key=lambda x: x[1][1], reverse=True)
    absence_real = sorted(label_given_not_word_dict.items(), key=lambda x: x[1][1], reverse=True)
    presence_fake = sorted(label_given_word_dict.items(), key=lambda x: x[1][0], reverse=True)
    absence_fake = sorted(label_given_not_word_dict.items(), key=lambda x: x[1][0], reverse=True)

#   part 3a:
    presence_real1 = presence_real[0:10]
    absence_real1 = absence_real[0:10]
    presence_fake1 = presence_fake[0:10]
    absence_fake1 = absence_fake[0:10]
    
    print "part 3a: Words with stop words"
    
    print_word(presence_real1, 1, "p(real|word)")
    print_word(presence_fake1, 0, "p(fake|word)")
    print_word(absence_real1, 1, "p(real|~word)")
    print_word(absence_fake1, 0, "p(fake|~word)")
    print "\n"
    
#    part 3b:
    presence_real2 = [i for i in presence_real if i[0] not in ENGLISH_STOP_WORDS][0:10]
    absence_real2 = [i for i in absence_real if i[0] not in ENGLISH_STOP_WORDS][0:10]
    presence_fake2 = [i for i in presence_fake if i[0] not in ENGLISH_STOP_WORDS][0:10]
    absence_fake2 = [i for i in absence_fake if i[0] not in ENGLISH_STOP_WORDS][0:10]

    print "part 3b: Words without stop words"
    
    print_word(presence_real2, 1, "p(real|word)")
    print_word(presence_fake2, 0, "p(fake|word)")
    print_word(absence_real2, 1, "p(real|~word)")
    print_word(absence_fake2, 0, "p(fake|~word)")

def print_NB_performance(training_real_results, training_fake_results, validation_real_results,
                         validation_fake_results, test_real_results, test_fake_results):
    
    training_performance = (training_real_results.count(1) + training_fake_results.count(0)) / \
                           float(len(training_real_results) + len(training_fake_results))
    validation_performance = (validation_real_results.count(1) + validation_fake_results.count(0)) / \
                             float(len(validation_real_results) + len(validation_fake_results))
    test_performance = (test_real_results.count(1) + test_fake_results.count(0)) / \
                       float(len(test_real_results) + len(test_fake_results))

    print "training performance:", training_performance * 100, "%"
    print "validation performance:", validation_performance * 100, "%"
    print "test performance:", test_performance * 100, "%"

def print_word(lst, label, lst_type):
    print lst_type
    filtered_lst = [(i[0], i[1][label]) for i in lst]
    print filtered_lst

# part 4:

def part4_6():
    real_words, real_validation, real_test, real_train = get_data(real_file)
    fake_words, fake_validation, fake_test, fake_train = get_data(fake_file)

    all_words = list(set(real_words.keys() + fake_words.keys()))

    training_x, training_y = get_set(all_words, real_train, fake_train)
    validation_x, validation_y = get_set(all_words, real_validation, fake_validation)
    test_x, test_y = get_set(all_words, real_test, fake_test)

    dim_x = int(training_x.shape[1])
    dim_out = 2

    np.random.seed(0)
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_out),
    )
    results = {}
    loss_fn = torch.nn.CrossEntropyLoss()
    l2_reg_lambda = 0.05

    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    for t in range(3000):
        y_pred = model(training_x).data.numpy()
        training_performance = np.mean(np.argmax(y_pred, 1) == training_y.data)
        y_pred = model(validation_x).data.numpy()
        validation_performance = np.mean(np.argmax(y_pred, 1) == validation_y.data)

        results[t] = [training_performance * 100, validation_performance * 100]

        y_pred = model(training_x)

        l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        for W in model.parameters():
            l2_reg = l2_reg + W.norm(2)
        loss = loss_fn(y_pred, training_y) + (l2_reg_lambda * l2_reg)



        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()  # Compute the gradient
        optimizer.step()  # Use the gradient information to
        # make a step

        if t % 100 == 0:
            y_pred = model(training_x).data.numpy()
            print "Performance:", np.mean(np.argmax(y_pred, 1) == training_y.data) * 100
            #print "loss:", loss

    print "\n"
    y_pred = model(training_x).data.numpy()
    print "training performance:%", np.mean(np.argmax(y_pred, 1) == training_y.data) * 100

    y_pred = model(validation_x).data.numpy()
    print "validation performance:%", np.mean(np.argmax(y_pred, 1) == validation_y.data) * 100

    y_pred = model(test_x).data.numpy()
    print "test performance:%", np.mean(np.argmax(y_pred, 1) == test_y.data) * 100

    learning_curve(results, "part4")

#   part 6
    weights = model[0].weight.data.numpy()

    fake_weights = list(weights[0,:].copy())
    real_weights = list(weights[1,:].copy())

    sorted_fake_weights = sorted(fake_weights, reverse=True)
    sorted_real_weights = sorted(real_weights, reverse=True)

#   part 61:
    top_fake1 = [(all_words[fake_weights.index(i)], i) for i in sorted_fake_weights[0:10]]
    top_real1 = [(all_words[real_weights.index(i)], i) for i in sorted_real_weights[0:10]]
    bottom_fake1 = [(all_words[fake_weights.index(i)], i) for i in sorted_fake_weights[-10:]][::-1]
    bottom_real1 = [(all_words[real_weights.index(i)], i) for i in sorted_real_weights[-10:]][::-1]
    
    print "part 6a: top 10 positive and negative thetas"
    print "top 10 positive real thetas:"
    print top_real1
    print "top 10 positive fake thetas:"
    print top_fake1
    print "top 10 negative real thetas:"
    print bottom_real1
    print "top 10 negative fake thetas:"
    print bottom_fake1
    print "\n"
#   part 6 b:
    top_fake2 = [(all_words[fake_weights.index(i)], i) for i in sorted_fake_weights
                 if all_words[fake_weights.index(i)] not in ENGLISH_STOP_WORDS][0:10]
    top_real2 = [(all_words[real_weights.index(i)], i) for i in sorted_real_weights
                 if all_words[real_weights.index(i)] not in ENGLISH_STOP_WORDS][0:10]
    bottom_fake2 = [(all_words[fake_weights.index(i)], i) for i in sorted_fake_weights
                    if all_words[fake_weights.index(i)] not in ENGLISH_STOP_WORDS][-10:][::-1]
    bottom_real2 = [(all_words[real_weights.index(i)], i) for i in sorted_real_weights
                    if all_words[real_weights.index(i)] not in ENGLISH_STOP_WORDS][-10:][::-1]

    print "part 6b: top 10 positive and negative thetas without stop words"
    print "top 10 positive real thetas:"
    print top_real2
    print "top 10 positive fake thetas:"
    print top_fake2
    print "top 10 negative real thetas:"
    print bottom_real2
    print "top 10 negative fake thetas:"
    print bottom_fake2

def part7():
    real_words, real_validation, real_test, real_train = get_data(real_file)
    fake_words, fake_validation, fake_test, fake_train = get_data(fake_file)

    all_words = list(set(real_words.keys() + fake_words.keys()))

    training_x, training_y = get_set(all_words, real_train, fake_train)
    validation_x, validation_y = get_set(all_words, real_validation, fake_validation)
    test_x, test_y = get_set(all_words, real_test, fake_test)

    training_x = training_x.data.numpy()
    training_y = training_y.data.numpy()
    validation_x = validation_x.data.numpy()
    validation_y = validation_y.data.numpy()
    test_x = test_x.data.numpy()
    test_y = test_y.data.numpy()

    depths = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250]
    for depth in depths:
        np.random.seed(0)
        dtc = tree.DecisionTreeClassifier(max_depth=depth, max_features=0.7)
        dtc.fit(training_x, training_y)

        print "Depth: ", depth
        print "training performance: ", dtc.score(training_x, training_y) * 100, "%"
        print "validation performance: ", dtc.score(validation_x, validation_y) * 100, "%"

    print "\n"
    np.random.seed(0)
    dtc = tree.DecisionTreeClassifier(max_depth=100, max_features=0.7)
    dtc.fit(training_x, training_y)
    print "training performance: ", dtc.score(training_x, training_y) * 100, "%"
    print "validation performance: ", dtc.score(validation_x, validation_y) * 100, "%"
    print "test performance: ", dtc.score(test_x, test_y) * 100, "%"

    tree.export_graphviz(dtc, out_file= figures_folder + "tree.dot", max_depth=2, filled=True,
                         rounded=True, feature_names=all_words, class_names=["fake", "real"])

def part8():
    real_words, real_validation, real_test, real_train = get_data(real_file)
    fake_words, fake_validation, fake_test, fake_train = get_data(fake_file)

    total = float(len(real_train) + len(fake_train))
    p_real = len(real_train) / total
    p_fake = len(fake_train) / total

    I_x_i = get_mutual_information(p_fake, p_real, real_train, fake_train, "the")
    I_x_j = get_mutual_information(p_fake, p_real, real_train, fake_train, "hillary")
    
    print "\n"
    print "I(Y, 'the'):", I_x_i
    print "I(Y, 'hillary'):", I_x_j

def get_mutual_information(p_fake, p_real, real_train, fake_train, x):
    train_dict = {}
    train_dict = get_counts(real_train, 1, train_dict)
    train_dict = get_counts(fake_train, 0, train_dict)

    p_x = (train_dict[x][0] + train_dict[x][1]) / float(len(real_train) + len(fake_train))
    h_x = (- p_x * np.log2(p_x)) + (- (1. - p_x) * np.log2(1. - p_x))
    
    m = 1.
    p = 0.03
    train_dict = {}
    train_dict = get_counts(real_train, 1, train_dict)
    train_dict = get_counts(fake_train, 0, train_dict)

    for k, v in train_dict.items():
        train_dict[k][0] = (train_dict[k][0] + m * p) / (float(len(fake_train)) + m)
        train_dict[k][1] = (train_dict[k][1] + m * p) / (float(len(real_train)) + m)

    p_word_real, p_word_fake, p_not_word_real, p_not_word_fake = get_word_p(x, train_dict)

    h_x_given_y = (p_fake * - ( (p_not_word_fake * np.log2(p_not_word_fake)) + (p_word_fake * np.log2(p_word_fake)) )) \
                  + (p_real * - ( (p_not_word_real * np.log2(p_not_word_real)) + (p_word_real * np.log2(p_word_real)) ))

    I_Y_x = h_x - h_x_given_y
    
    return I_Y_x


def get_word_p(x, words_dict):
    p_word_real = 0.
    p_word_fake = 0.
    for word in words_dict:
        if word == x:
            p_word_real += np.log(words_dict[word][1])
            p_word_fake += np.log(words_dict[word][0])
        else:
            p_word_real += np.log(1. -  (words_dict[word][1]))
            p_word_fake += np.log(1. -  (words_dict[word][0]))

    p_word_real = np.exp(p_word_real)
    p_word_fake = np.exp(p_word_fake)

    p_not_word_real = 1. - p_word_real
    p_not_word_fake = 1. - p_word_fake
    
    return p_word_real, p_word_fake, p_not_word_real, p_not_word_fake

def learning_curve(results, name):
    lists = sorted(results.items())
    x, y = zip(*lists)

    plt.figure(int(name[-1]))
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Performance %")
    plt.legend(["Training", "Validation"])
    plt.savefig(figures_folder + name + "_learning_curve.png")

# return validation, training, or test set
def get_set(all_words, real_set, fake_set):


    real_n = len(real_set)
    n = len(all_words)
    lst = real_set + fake_set

    set_x = []
    set_y = np.zeros((len(lst), 2))
    set_y[:real_n - 1, 1] = 1.
    set_y[real_n:, 0] = 1.
    for line in lst:
        line_words = list(set(line.split()))
        x = np.zeros(n)

        for word in line_words:
            x[all_words.index(word)] = 1.
        set_x.append(x)

    set_x = np.vstack(set_x)
    set_y = np.vstack(set_y)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    set_x = Variable(torch.from_numpy(set_x), requires_grad=False).type(dtype_float)
    set_y = Variable(torch.from_numpy(np.argmax(set_y, 1)), requires_grad=False).type(dtype_long)
    return set_x, set_y

# Main Function
if __name__ == '__main__':
    part1_2_3()
    part4_6()
    part7()
    part8()