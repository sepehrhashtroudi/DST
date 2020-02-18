from keras.models import Sequential,Model
from keras import metrics
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from sklearn.metrics import accuracy_score
from keras.layers import Convolution1D, MaxPooling1D, Bidirectional,Input, Embedding, LSTM, Dense,concatenate
from sklearn.metrics import classification_report
import progressbar
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras as k

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
# sess = tf.Session(config=config) 
# k.backend.set_session(sess)

# import sys
# np.set_printoptions(threshold=sys.maxsize)
import preprocess_data as pre

pre.clean_dataset()
train_data, test_data = pre.load_data()
word_to_index = pre.get_word_to_index(train_data)
n_vocab = len(word_to_index)
print(word_to_index)
dialogs = pre.get_user_utterances(train_data, word_to_index)
test_dialogs = pre.get_user_utterances(test_data, word_to_index)
print(dialogs[0])
for turn in dialogs[0]:
    print(pre.index_to_word(turn, word_to_index))

food_dict, area_dict, pricerange_dict = pre.get_ontology(train_data)
bs, bs_index, bs_onehot = pre.get_belief_states(train_data, food_dict, area_dict, pricerange_dict)
test_bs, test_bs_index, test_bs_onehot = pre.get_belief_states(test_data, food_dict, area_dict, pricerange_dict)
food_nb_classes = len (food_dict)
area_nb_classes = len (area_dict)
price_nb_classes = len (pricerange_dict)
print(bs[0])
print(bs_index[0])
print(bs_onehot[0])
print(food_dict)
print(area_dict)
print(pricerange_dict)
print(word_to_index)

sent_input = Input(shape=(None,), name='main_in')
x = Embedding(output_dim=300, input_dim=n_vocab)(sent_input)

lstm_out = Bidirectional(LSTM(units=150, return_sequences=False))(x)
lstm_out = Dense(150, activation='relu',name='lstm_out')(lstm_out)

bs_input_food = Input(shape=(food_nb_classes,), name='bs_in_food')
bs_input_area = Input(shape=(area_nb_classes,), name='bs_in_area')
bs_input_price = Input(shape=(price_nb_classes,), name='bs_in_price')


bs_out_food_joined = concatenate([lstm_out, bs_input_food])
bs_out_food_joined = Dense(food_nb_classes, activation='relu')(bs_out_food_joined)
bs_out_food_joined = Dense(food_nb_classes, activation='relu')(bs_out_food_joined)
bs_out_food_joined = Dense(food_nb_classes, activation='softmax', name='main_out')(bs_out_food_joined)

bs_out_area = Dense(area_nb_classes, activation='relu')(bs_input_area)
bs_out_area_joined = concatenate([lstm_out, bs_out_area])
bs_out_area_joined = Dense(area_nb_classes, activation='softmax', name='main_out2')(bs_out_area_joined)

bs_out_price = Dense(price_nb_classes, activation='relu')(bs_input_price)
bs_out_price_joined = concatenate([lstm_out, bs_out_price])
bs_out_price_joined = Dense(price_nb_classes, activation='softmax', name='main_out3')(bs_out_price_joined)

# model = Model(inputs=[sent_input, bs_input_food, bs_input_area, bs_input_price], outputs=[bs_out_food_joined, bs_out_area_joined, bs_out_price_joined])
model = Model(inputs=[sent_input, bs_input_food], outputs=[bs_out_food_joined])
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])

### Training
n_epochs = 100

train_f_scores = []
val_f_scores = []
best_val_f1 = 0

for i in range(n_epochs):
    print("Epoch {}".format(i))

    print("Training =>")
    bs_pred = []
    ground_truth = []
    avgLoss = 0
    n_batch = 0
    n_turn = 0

    for n_batch, dialog in enumerate(tqdm(dialogs)):
        for n_turn, turn in enumerate(dialog):
            if(n_turn == 0):
                input_turn_bs = [np.eye(food_nb_classes)[0],np.eye(area_nb_classes)[0],np.eye(price_nb_classes)[0]]
            else:
                input_turn_bs = bs_onehot[n_batch][n_turn-1]

            food_turn_bs_in = np.array(input_turn_bs[0])
            food_turn_bs_in = food_turn_bs_in[np.newaxis, :]
            area_turn_bs_in = np.array(input_turn_bs[1])
            area_turn_bs_in = area_turn_bs_in[np.newaxis, :]
            price_turn_bs_in = np.array(input_turn_bs[2])
            price_turn_bs_in = price_turn_bs_in[np.newaxis, :]
            sent = np.array(turn)
            sent = sent[np.newaxis, :]
            bs_out = bs_onehot[n_batch][n_turn]
            food_out = np.array(bs_out[0])
            food_out = food_out[np.newaxis, :]
            area_out = np.array(bs_out[1])
            area_out = area_out[np.newaxis, :]
            price_out = np.array(bs_out[2])
            price_out = price_out[np.newaxis, :]
            # print(sent.shape)
            # print(input_turn_bs_np.shape)
            # print(bs_out.shape)
            if sent.shape[1] > 1:  # some bug in keras
                # print(input_turn_bs_np,[food_out,area_out,price_out])
                loss = model.train_on_batch([sent,food_turn_bs_in], [food_out])
                # loss = model.train_on_batch([sent,food_turn_bs_in,area_turn_bs_in,price_turn_bs_in], [food_out,area_out,price_out])
                # print(food_turn_bs_np,area_turn_bs_np,price_turn_bs_np)
                # print([food_out, area_out, price_out])
                avgLoss += loss[0]
                # price_acc += loss[6]
            # pred = model.predict_on_batch([sent, food_turn_bs_in, area_turn_bs_in, price_turn_bs_in])
            pred = model.predict_on_batch([sent, food_turn_bs_in])
            food_bs_pred = np.argmax(pred, -1)[0]
            # area_bs_pred = np.argmax(pred[1], -1)[0]
            # price_bs_pred = np.argmax(pred[2], -1)[0]
            # bs_pred.append([food_bs_pred,area_bs_pred,price_bs_pred])
            bs_pred.append([food_bs_pred])
            # bs_pred.append([np.argmax(food_turn_bs_in, -1)[0],np.argmax(area_turn_bs_in, -1)[0],np.argmax(price_turn_bs_in, -1)[0]])
            # ground_truth.append([np.argmax(food_out, -1)[0],np.argmax(area_out, -1)[0],np.argmax(price_out, -1)[0]])
            ground_truth.append([np.argmax(food_out, -1)[0]])


    avgLoss = avgLoss
    print(np.array(bs_pred)[:,0])
    print(np.array(ground_truth)[:,0])
    food_acc = accuracy_score(np.array(ground_truth)[:, 0], np.array(bs_pred)[:, 0])
    # area_acc = accuracy_score(np.array(ground_truth)[:, 1], np.array(bs_pred)[:, 1])
    # price_acc = accuracy_score(np.array(ground_truth)[:, 2], np.array(bs_pred)[:, 2])
    joint_acc = pre.joint_acc(ground_truth, bs_pred)
    print("food_acc:{}".format(food_acc))
    # print("area_acc:{}".format(area_acc))
    # print("price_acc:{}".format(price_acc))
    print("joint_acc:{}".format(joint_acc))

    test_bs_pred = []
    test_ground_truth = []
    for n_batch, dialog in enumerate(tqdm(test_dialogs)):
        for n_turn, turn in enumerate(dialog):
            if(n_turn == 0):
                input_turn_bs = [np.eye(food_nb_classes)[0],np.eye(area_nb_classes)[0],np.eye(price_nb_classes)[0]]
            else:
                input_turn_bs = test_bs_onehot[n_batch][n_turn-1]

            food_turn_bs_in = np.array(input_turn_bs[0])
            food_turn_bs_in = food_turn_bs_in[np.newaxis, :]
            area_turn_bs_in = np.array(input_turn_bs[1])
            area_turn_bs_in = area_turn_bs_in[np.newaxis, :]
            price_turn_bs_in = np.array(input_turn_bs[2])
            price_turn_bs_in = price_turn_bs_in[np.newaxis, :]
            sent = np.array(turn)
            sent = sent[np.newaxis, :]
            bs_out = test_bs_onehot[n_batch][n_turn]
            food_out = np.array(bs_out[0])
            food_out = food_out[np.newaxis, :]
            area_out = np.array(bs_out[1])
            area_out = area_out[np.newaxis, :]
            price_out = np.array(bs_out[2])
            price_out = price_out[np.newaxis, :]
            # print(sent.shape)
            # print(input_turn_bs_np.shape)
            # print(bs_out.shape)
            # pred = model.predict_on_batch([sent, food_turn_bs_in, area_turn_bs_in, price_turn_bs_in])
            pred = model.predict_on_batch([sent, food_turn_bs_in])
            food_bs_pred = np.argmax(pred, -1)[0]
            # area_bs_pred = np.argmax(pred[1], -1)[0]
            # price_bs_pred = np.argmax(pred[2], -1)[0]
            # test_bs_pred.append([food_bs_pred,area_bs_pred,price_bs_pred])
            test_bs_pred.append([food_bs_pred])
            # test_ground_truth.append([np.argmax(food_out, -1)[0],np.argmax(area_out, -1)[0],np.argmax(price_out, -1)[0]])
            test_ground_truth.append([np.argmax(food_out, -1)[0]])

    avgLoss = avgLoss
    print(np.array(test_bs_pred)[:,0])
    print(np.array(test_ground_truth)[:,0])
    joint_acc = pre.joint_acc(test_ground_truth, test_bs_pred)
    food_acc = accuracy_score(np.array(test_ground_truth)[:, 0], np.array(test_bs_pred)[:, 0])
    # area_acc = accuracy_score(np.array(test_ground_truth)[:, 1], np.array(test_bs_pred)[:, 1])
    # price_acc = accuracy_score(np.array(test_ground_truth)[:, 2], np.array(test_bs_pred)[:, 2])
    print("test_food_acc:{}".format(food_acc))
    # print("test_area_acc:{}".format(area_acc))
    # print("test_price_acc:{}".format(price_acc))
    print("test_joint_acc:{}".format(joint_acc))



