import json
import numpy as np
import nltk

def load_data():
    with open("data/woz_train_validate_en_clean.json","r") as f:
        train_data = json.load(f)
    with open("data/woz_test_en_clean.json","r") as f:
        test_data = json.load(f)
    # with open("data/CamRestOTGY.json", "r") as f:
    #     ontology = json.load(f)
    return train_data, test_data #, ontology

def clean_dataset():
    fin = open("data/woz_train_en.json", "rt")
    fout = open("data/woz_train_en_clean.json", "wt")
    for line in fin:
        fout.write(line.replace('centre', 'center'))
    fin.close()
    fout.close()
    fin = open("data/woz_validate_en.json", "rt")
    fout = open("data/woz_validate_en_clean.json", "wt")
    for line in fin:
        fout.write(line.replace('centre', 'center'))
    fin.close()
    fout.close()
    fin = open("data/woz_test_en.json", "rt")
    fout = open("data/woz_test_en_clean.json", "wt")
    for line in fin:
        fout.write(line.replace('centre', 'center'))
    fin.close()
    fout.close()


def get_user_utterances(train_data, word_to_index):
    dialogs = []
    for dialog in train_data:
        user_utterances = []
        for turn in dialog["dialogue"]:
            words = nltk.word_tokenize(turn["transcript"])
            indexd_utterances = []
            for word in words:
                if(word in word_to_index):
                    indexd_utterances.append(word_to_index[word])
                else:
                    indexd_utterances.append(word_to_index["UNK"])
            user_utterances.append(indexd_utterances)
        dialogs.append(user_utterances)
    return dialogs

def get_word_to_index(train_data):
    word_to_index = {"UNK":0}
    word_index = 1
    for dialog in train_data:
        for turn in dialog["dialogue"]:
            words = nltk.word_tokenize(turn["transcript"])
            for word in words:
                if word not in word_to_index:
                    word_to_index[word] = word_index
                    word_index += 1
    return  word_to_index

def index_to_word(turn, word_to_index):
    index_to_word = {v: k for k, v in word_to_index.items()}
    turn_to_text = []
    for index in turn:
        if index in index_to_word:
            turn_to_text.append(index_to_word[index])
        else:
            turn_to_text.append("UNK")
    return  turn_to_text

def get_belief_states(train_data,food_dict, area_dict, pricerange_dict):
    food_max_index = len(food_dict)
    area_max_index = len(area_dict)
    pricerange_max_index = len(pricerange_dict)
    bs =[]
    bs_index = []
    bs_onehot = []
    for dialog in train_data:
        dialog_bs = []
        dialog_bs_index = []
        dialog_bs_onehot = []
        for turn in dialog["dialogue"]:
            turn_bs = ["not mentioned", "not mentioned", "not mentioned"]
            turn_bs_index = [0, 0, 0] # set to not mentioned
            turn_bs_onehot = [np.eye(food_max_index)[0],np.eye(area_max_index)[0],np.eye(pricerange_max_index)[0]]
            for belief in turn["belief_state"]:
                if belief["act"] == "inform":
                    slot_value = belief["slots"][0][0]+" "+belief["slots"][0][1]
                    if slot_value in food_dict:
                        turn_bs[0] = (slot_value)
                        turn_bs_index[0] = (food_dict[slot_value])
                        turn_bs_onehot[0] = np.eye(food_max_index)[food_dict[slot_value]]
                    if slot_value in area_dict:
                        turn_bs[1] = (slot_value)
                        turn_bs_index[1] = (area_dict[slot_value])
                        turn_bs_onehot[1] = np.eye(area_max_index)[area_dict[slot_value]]
                    if slot_value in pricerange_dict:
                        turn_bs[2] = (slot_value)
                        turn_bs_index[2] = (pricerange_dict[slot_value])
                        turn_bs_onehot[2] = np.eye(pricerange_max_index)[pricerange_dict[slot_value]]
            dialog_bs.append(turn_bs)
            dialog_bs_index.append(turn_bs_index)
            dialog_bs_onehot.append(turn_bs_onehot)
        bs.append(dialog_bs)
        bs_index.append(dialog_bs_index)
        bs_onehot.append(dialog_bs_onehot)
    return bs , bs_index, bs_onehot

def get_ontology(train_data):
    food_dict = {"not mentioned":0 , "food dontcare" : 1}
    area_dict = {"not mentioned":0 , "area dontcare" : 1}
    pricerange_dict = {"not mentioned":0 , "price range dontcare" :1}
    food_index = 2
    area_index = 2
    pricerange_index = 2
    for dialog in train_data:
        for turn in dialog["dialogue"]:
            for belief in turn["belief_state"]:
                if belief["act"] == "inform":
                    bs = belief["slots"][0][0] + " " + belief["slots"][0][1]
                    if belief["slots"][0][0] == "food":
                        if bs not in food_dict:
                            food_dict[bs] = food_index
                            food_index += 1
                    if belief["slots"][0][0] == "area":
                        if bs not in area_dict:
                            area_dict[bs] = area_index
                            area_index += 1
                    if belief["slots"][0][0] == "price range":
                        if bs not in pricerange_dict:
                            pricerange_dict[bs] = pricerange_index
                            pricerange_index += 1
    index_to_slot_food = {v: k for k, v in food_dict.items()}
    index_to_slot_area = {v: k for k, v in area_dict.items()}
    index_to_slot_price = {v: k for k, v in pricerange_dict.items()}
    return food_dict, area_dict, pricerange_dict, index_to_slot_food, index_to_slot_area, index_to_slot_price

def joint_acc(y_true,y_pred):
    true = 0
    for i,y in enumerate(y_true):
        if y == y_pred[i]:
            true +=1
    return true/len(y_true)

def print_random_dialog(model, dialogs, bs_onehot, word_to_index, i2s_food, i2s_area, i2s_price):
    random_dialog = np.random.randint(len(dialogs))
    food_nb_classes = len (i2s_food)
    area_nb_classes = len (i2s_area)
    price_nb_classes = len (i2s_price)
    for n_turn, turn in enumerate(dialogs[random_dialog]):
        if(n_turn == 0):
            input_turn_bs = [np.eye(food_nb_classes)[0],np.eye(area_nb_classes)[0],np.eye(price_nb_classes)[0]]
        else:
            input_turn_bs = bs_onehot[random_dialog][n_turn-1]
        food_turn_bs_in = np.array(input_turn_bs[0])
        food_turn_bs_in = food_turn_bs_in[np.newaxis, :]
        area_turn_bs_in = np.array(input_turn_bs[1])
        area_turn_bs_in = area_turn_bs_in[np.newaxis, :]
        price_turn_bs_in = np.array(input_turn_bs[2])
        price_turn_bs_in = price_turn_bs_in[np.newaxis, :]
        sent = np.array(turn)
        sent = sent[np.newaxis, :]
        bs_out = bs_onehot[random_dialog][n_turn]
        food_out = np.array(bs_out[0])
        food_out = food_out[np.newaxis, :]
        area_out = np.array(bs_out[1])
        area_out = area_out[np.newaxis, :]
        price_out = np.array(bs_out[2])
        price_out = price_out[np.newaxis, :]
        pred = model.predict_on_batch([sent, food_turn_bs_in, area_turn_bs_in, price_turn_bs_in])
        print(index_to_word(turn, word_to_index))
        print("prediction: {}, {}, {}".format(i2s_food[ np.argmax(pred[0], -1)[0] ], i2s_area[ np.argmax(pred[1], -1)[0] ], i2s_price[ np.argmax(pred[2], -1)[0] ]))
        print("truth:      {}, {}, {}".format(i2s_food[ np.argmax(food_out, -1)[0]],i2s_area[ np.argmax(area_out, -1)[0] ],i2s_price[ np.argmax(price_out, -1)[0] ]))

def print_false_dialogs(model, dialogs, bs_onehot, word_to_index, i2s_food, i2s_area, i2s_price):
    food_nb_classes = len (i2s_food)
    area_nb_classes = len (i2s_area)
    price_nb_classes = len (i2s_price)
    for n_batch, dialog in enumerate(dialogs):
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
            pred = model.predict_on_batch([sent, food_turn_bs_in, area_turn_bs_in, price_turn_bs_in])
            prediction_lable = [i2s_food[ np.argmax(pred[0], -1)[0] ], i2s_area[ np.argmax(pred[1], -1)[0] ], i2s_price[ np.argmax(pred[2], -1)[0] ]]
            truth_lable = [i2s_food[ np.argmax(food_out, -1)[0]],i2s_area[ np.argmax(area_out, -1)[0] ],i2s_price[ np.argmax(price_out, -1)[0] ]]
            if  prediction_lable != truth_lable:
                print(index_to_word(turn, word_to_index))
                print("prediction: {}, {}, {}".format(prediction_lable[0],prediction_lable[1],prediction_lable[2]))
                print("truth:      {}, {}, {}".format(truth_lable[0],truth_lable[1],truth_lable[2]))



if __name__ == "__main__":
    # clean_dataset()
    train_data, test_data = load_data()
    word_to_index = get_word_to_index(train_data)
    print(word_to_index)
    dialogs = get_user_utterances(train_data, word_to_index)
    print(dialogs[0])
    for turn in dialogs[0]:
        print(index_to_word(turn,word_to_index))

    food_dict, area_dict, pricerange_dict,_,_,_ = get_ontology(train_data)
    bs, bs_index,bs_onehot = get_belief_states(train_data,food_dict, area_dict, pricerange_dict)
    print(bs[0])
    print(bs_index[0])
    print(bs_onehot[0])
    print(food_dict)
    print(area_dict)
    print(pricerange_dict)
    print(word_to_index)