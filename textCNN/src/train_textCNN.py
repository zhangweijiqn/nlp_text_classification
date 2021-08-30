import os
import sys
from collections import Counter
failed_files = []
label_sep = "、"

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
sys.path.append(root_path)

import tensorflow as tf
from model import TextCNN
from etl import *

# configuration
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("traning_data_path", "data/input",
                           "path of traning data.")  # ../data/sample_multiple_label.txt
# tf.app.flags.DEFINE_integer("vocab_size",100000,"maximum vocab size.")

tf.app.flags.DEFINE_string("cache_file_h5py", "data/input-samples/data.h5",
                           "path of training/validation/test data.")  # ../data/sample_multiple_label.txt
tf.app.flags.DEFINE_string("cache_file_pickle", "data/input-samples/vocab_label.pik",
                           "path of vocabulary and label files")  # ../data/sample_multiple_label.txt

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
tf.app.flags.DEFINE_float("threshold", 0.5, "classification score threhold")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")  # 0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir", "trained_model/text_cnn/checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 200, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.app.flags.DEFINE_boolean("is_training_flag", True, "is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 10, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding", False, "whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters")  # 256--->512
tf.app.flags.DEFINE_string("word2vec_model_path", "word2vec-title-desc.bin", "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")
tf.app.flags.DEFINE_boolean("is_fenci", False, "if need fenci.")
tf.app.flags.DEFINE_string("labels", None, '一/二/三级标签任务')

##############################################################################################################################################
filter_sizes = [3, 4, 5]
##############################################################################################################################################

# 1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    # trainX, trainY, testX, testY = None, None, None, None
    # vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, _= create_vocabulary(FLAGS.traning_data_path,FLAGS.vocab_size,name_scope=FLAGS.name_scope)
    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY = preprocess(FLAGS.traning_data_path)
    # word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY = load_data(FLAGS.cache_file_h5py,FLAGS.cache_file_pickle)
    vocab_size = len(word2index)
    print("cnn_model.vocab_size:", vocab_size)
    num_classes = int(len(label2index))          # 目前发现读取 label2index 会读成双份，为了简单直接除以2
    if FLAGS.multi_label_flag == False:
        num_classes = 1
    print("num_classes:", num_classes, label2index)
    num_examples, FLAGS.sentence_len = trainX.shape
    print("num_examples of training:", num_examples, ";sentence_len:", FLAGS.sentence_len)
    # train, test= load_data_multilabel(FLAGS.traning_data_path,vocabulary_word2index, vocabulary_label2index,FLAGS.sentence_len)
    # trainX, trainY = train;testX, testY = test
    # print some message for debug purpose
    print("trainX[0:10]:", trainX[0:10])
    print("trainY[0]:", trainY[0:10])
    print("train_y_short:", trainY[0])
    print("train_y shape:", trainY.shape)

    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, num_classes, FLAGS.learning_rate, FLAGS.batch_size,
                          FLAGS.decay_steps,
                          FLAGS.decay_rate, FLAGS.sentence_len, vocab_size, FLAGS.embed_size,
                          multi_label_flag=FLAGS.multi_label_flag)
        # Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            # for i in range(3): #decay learning rate if necessary.
            #    print(i,"Going to decay learning rate by half.")
            #    sess.run(textCNN.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # load pre-trained word embedding
                index2word = {v: k for k, v in word2index.items()}
                assign_pretrained_word_embedding(sess, index2word, vocab_size, textCNN, FLAGS.word2vec_model_path)
        curr_epoch = sess.run(textCNN.epoch_step)
        # 3.feed data & training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        iteration = 0
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, counter = 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),
                                  range(batch_size, number_of_training_data, batch_size)):
                iteration = iteration + 1
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])

                feed_dict = {textCNN.input_x: trainX[start:end], textCNN.dropout_keep_prob: 0.8,
                             textCNN.is_training_flag: FLAGS.is_training_flag}
                if not FLAGS.multi_label_flag:
                    feed_dict[textCNN.input_y] = trainY[start:end]
                else:
                    feed_dict[textCNN.input_y_multilabel] = trainY[start:end]
                # print("debug feeddict:", trainY[start:end].shape, trainY[start:end], type(trainY[start:end][0][0]))
                curr_loss, lr, _ = sess.run([textCNN.loss_val, textCNN.learning_rate, textCNN.train_op], feed_dict)
                loss, counter = loss + curr_loss, counter + 1
                if counter % 1 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" % (
                        epoch, counter, loss / float(counter), lr))

                ########################################################################################################
                if start % (10 * FLAGS.batch_size) == 0:  # eval every 10 steps.
                    # print('validX, validY', vaildX, vaildY)
                    eval_loss, f1_score, f1_micro, f1_macro, p, r = do_eval(sess, textCNN, vaildX, vaildY, num_classes)
                    print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f\tPrecision:%.3f\tRecall:%.3f" % (
                        epoch, eval_loss, f1_score, f1_micro, f1_macro, p, r))
                    # save model to checkpoint
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    print("Going to save model..")
                    saver.save(sess, save_path, global_step=epoch)
                ########################################################################################################
            # epoch increment
            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            # 4.validation
            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, f1_score, f1_micro, f1_macro, p, r = do_eval(sess, textCNN, testX, testY, num_classes)
                print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f\tPrecision:%.3f\tRecall:%.3f" % (
                    epoch, eval_loss, f1_score, f1_micro, f1_macro, p, r))
                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, f1_score, f1_micro, f1_macro, p, r = do_eval(sess, textCNN, testX, testY, num_classes)
        print("Test Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f\tPrecision:%.3f\tRecall:%.3f" % (test_loss, f1_score, f1_micro, f1_macro, p, r))
    pass


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, textCNN, evalX, evalY, num_classes):
    # evalX = evalX[0:3000]
    # evalY = evalY[0:3000]
    number_examples = len(evalX)
    eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0, 0.0, 0.0, 0.0
    batch_size = 1
    predict = []

    for start, end in zip(range(0, number_examples, batch_size),
                          range(batch_size, number_examples + batch_size, batch_size)):
        ''' evaluation in one batch '''
        feed_dict = {textCNN.input_x: evalX[start:end],
                     textCNN.input_y_multilabel: evalY[start:end],
                     textCNN.dropout_keep_prob: 1.0,
                     textCNN.is_training_flag: False}
        current_eval_loss, logits = sess.run(
            [textCNN.loss_val, textCNN.logits], feed_dict)
        predict = [*predict, np.argmax(np.array(logits[0]))]
        eval_loss += current_eval_loss
        eval_counter += 1
    evalY = [np.argmax(ii) for ii in evalY]

    if not FLAGS.multi_label_flag:
        predict = [int(ii > FLAGS.threshold) for ii in predict]
    p, r, f1_macro, f1_micro, _ = fastF1(predict, evalY, num_classes)
    f1_score = (f1_micro + f1_macro) / 2.0

    return eval_loss / float(eval_counter), f1_score, f1_micro, f1_macro, p, r


def fastF1(result: list, predict: list, num_classes: int):
    print('result,predict', result, predict)
    ''' f1 score '''
    true_total, r_total, p_total, p, r = 0, 0, 0, 0, 0
    total_list = []
    # import pdb
    # pdb.set_trace()
    for trueValue in range(num_classes):
        trueNum, recallNum, precisionNum = 0, 0, 0
        for index, values in enumerate(result):
            if values == trueValue:
                recallNum += 1
                if values == predict[index]:
                    trueNum += 1
            if predict[index] == trueValue:
                precisionNum += 1
        # print('trueValue,trueNum,recallNum,precisionNum', trueValue, trueNum, recallNum, precisionNum)
        R = trueNum / recallNum if recallNum else 0
        P = trueNum / precisionNum if precisionNum else 0
        true_total += trueNum
        r_total += recallNum
        p_total += precisionNum
        p += P
        r += R
        f1 = (2 * P * R) / (P + R) if (P + R) else 0
        total_list.append([P, R, f1])
    p, r = np.array([p, r]) / num_classes
    micro_r, micro_p = true_total / np.array([r_total, p_total])
    macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) else 0
    accuracy = true_total / len(result)
    print('P: {:.2f}%, R: {:.2f}%, Micro_f1: {:.2f}%, Macro_f1: {:.2f}%, Accuracy: {:.2f}'.format(
        p * 100, r * 100, micro_f1 * 100, macro_f1 * 100, accuracy * 100))
    return p, r, macro_f1, micro_f1, total_list


def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN, word2vec_model_path):
    print("using pre-trained word emebedding.ended...")


def preprocess(traning_data_path):
    """
    load data from csv files, gen vocabulary, trans label to multi-hot, word to index
    :param traning_data_path:
    :return: word2index, index2word
    """
    if not os.path.exists(traning_data_path):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download traning_data_path file, it include training data and vocabulary & labels. ")
    print("INFO: file exists. going to load training file")

    cache_path_h5py = FLAGS.cache_file_h5py
    cache_path_pickle = FLAGS.cache_file_pickle
    training_fenci_path = traning_data_path + "-fenci"
    max_sentence_length = 128

    if FLAGS.is_fenci:
        print('######### starting to fenci ...... ########', FLAGS.is_fenci)
        fenci_dir_pandas(traning_data_path, training_fenci_path, field_label=FLAGS.labels, cn_clean=False)

    # based on fenci result
    label2index, _ = gen_label_vocabulary(training_fenci_path, traning_data_path + '/vocabulary/label_set.txt', field_name='labels', multi_label=FLAGS.multi_label_flag)
    word2index, _ = gen_word_vocabulary(training_fenci_path, traning_data_path + '/vocabulary/word_set.txt')

    df = read_dir(training_fenci_path, is_subdir=False)  # columns: labels, contents
    X, Y = get_X_Y(df, word2index, label2index, FLAGS.multi_label_flag)
    # print("Y after get_X_Y:", Y)
    X = pad_sequences(X, maxlen=max_sentence_length, value=0.)  # padding to max length

    train_X, train_Y, vaild_X, valid_Y, test_X, test_Y = shuffle_and_split(X, Y)

    print("train_X:", train_X.shape, ";train_Y:", train_Y.shape, ";vaild_X.shape:", vaild_X.shape, ";valid_Y:",
          valid_Y.shape, ";test_X:", test_X.shape, ";test_Y:", test_Y.shape)
    print("train_Y:", train_Y)

    # save to file system
    save_data(cache_path_h5py, cache_path_pickle, word2index, label2index, train_X, train_Y, vaild_X, valid_Y, test_X,
              test_Y)
    print("save cache files to file system successfully!")

    return word2index, label2index, train_X, train_Y, vaild_X, valid_Y, test_X, test_Y


def load_data(cache_file_h5py, cache_file_pickle):
    """
    load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download cache file, it include training data and vocabulary & labels. "
                           "link can be found in README.md\n download zip file, unzip it, then put cache files as FLAGS."
                           "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
    print("INFO. cache file exists. going to load cache file")
    f_data = h5py.File(cache_file_h5py, 'r')
    print("f_data.keys:", list(f_data.keys()))
    train_X = f_data['train_X']  # np.array(
    print("train_X.shape:", train_X.shape)
    train_Y = f_data['train_Y']  # np.array(
    print("train_Y.shape:", train_Y.shape, ";")
    vaild_X = f_data['vaild_X']  # np.array(
    valid_Y = f_data['valid_Y']  # np.array(
    test_X = f_data['test_X']  # np.array(
    test_Y = f_data['test_Y']  # np.array(
    # print(train_X)
    # f_data.close()

    word2index, label2index = None, None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index = pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    print("INFO. cache file load successful:", label2index)
    return word2index, label2index, train_X, train_Y, vaild_X, valid_Y, test_X, test_Y

def gen_label_vocabulary(file_path, save_path, field_name='labels', multi_label=True):
    c_labels = Counter()
    df = read_dir(file_path)
    for index, row in df.iterrows():
        try:
            if multi_label:
                label_list = str(row[field_name]).split(label_sep)
                label_list = [l.strip() for l in label_list]
                c_labels.update(label_list)
            else:
                c_labels.update([row[field_name]])
        except Exception as e:
            print("error: " + str(e))
            print(row[field_name])
            continue

    label_list = c_labels.most_common()
    label2index = {}
    index2label = {}
    label_target_object = open(save_path, 'w', encoding='utf-8')
    for i, label_freq in enumerate(label_list):
        label, freq = label_freq
        label2index[label] = i
        index2label[i] = label
        label_target_object.write(str(label) + "," + str(i) + "," + str(freq) + "\n")
        if i < 20: print(label, freq)
    label_target_object.close()
    print("generate label dict successful.")
    print("gen_label_vocabulary label2index:", len(label2index), label2index)
    return label2index, index2label

def get_label_vocabulary(file_path):
    index2label = {}
    label2index = {}
    classes_ = []
    with open(file_path, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            fields = line.split(',')
            label = fields[0]
            index = fields[1]
            index2label[int(index)] = label
            label2index[label] = int(index)
    print("get_label_vocabulary label2index:", len(label2index))
    return index2label, label2index

def gen_word_vocabulary(file_path, save_path):
    c_words = Counter()
    docs = load_dir(file_path, 'seg.txt')
    for row in docs:
        try:
            word_list = row.split(" ")
            c_words.update(word_list)
        except Exception as e:
            print("error: " + str(e))
            print(row)
            continue

    word_list = c_words.most_common()
    word2index = {}
    index2word = {}
    label_target_object = open(save_path, 'w', encoding='utf-8')
    for i, w_freq in enumerate(word_list):
        w, freq = w_freq
        word2index[w] = i
        index2word[i] = w
        label_target_object.write(w + "," + str(i) + "\n")
        if i < 20: print(w, freq)
    label_target_object.close()
    print("generate label dict successful.")
    return word2index, index2word

if __name__ == "__main__":
    tf.app.run()
