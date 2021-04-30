import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import models,embeddings,helpers

early_stopping = EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=20,
        mode='min',
        restore_best_weights=True)

def LR_model_training(train_ds, test_ds, textField, labelField, max_features,
                      batch_size,no_epochs,saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    # LR Model training
    AUC_scores = []
    F1_scores = []
    micro_F1_scores = []
    macro_F1_scores = []
    training_time = []

    #y_train = train_ds[labelField]
    #y_test = test_ds[labelField]

    for i in range (5):
        print("iteration" + str(i))
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3, stratify=train_ds[labelField])

        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range = (1,3))
        vectorizer = vectorizer.fit(train_ds[textField])

        X_train = vectorizer.transform(X_train)
        X_valid = vectorizer.transform(X_valid)
        X_test = vectorizer.transform(test_ds[textField])

        saver = ModelCheckpoint(saver_path + "/" + saver_name)
        # Logistic regression
        LR_model = models.LR(X_train.shape[1])
        print("model created")
        LR_training_history = LR_model.fit(
            X_train,
            y_train,
            epochs = no_epochs,
            batch_size = batch_size,
            validation_data = [X_valid, y_valid],
            callbacks=[earlystopping,saver],
            verbose=0)

        predicted_labels = LR_model.predict(X_test)
        print("AUC score", roc_auc_score(test_ds[labelField],predicted_labels))
        print("F1 score", f1_score(test_ds[labelField],np.rint(predicted_labels)))
        print("micro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        print("macro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField],predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        macro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        micro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration"+str(i))

    test_ds["LR_prediction"] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    print("macro_f1_avg", np.mean(macro_F1_scores))
    print("micro_f1_avg", np.mean(micro_F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("macro_F1_mean: " + str(np.mean(macro_F1_scores)))
    f.write("\n")
    f.write("micro_F1_mean: " + str(np.mean(micro_F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")
def MLP_model_training(train_ds, test_ds, textField, labelField,maxlen,embed_size
                      ,batch_size,no_epochs, saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    AUC_scores = []
    F1_scores = []
    training_time = []
    micro_F1_scores = []
    macro_F1_scores = []
    for i in range(5):
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3, stratify=train_ds[labelField])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_ds[textField])
        vocab_size = len(tokenizer.word_index) + 1

        X_train = tokenizer.texts_to_sequences(X_train)
        X_valid = tokenizer.texts_to_sequences(X_valid)
        X_test = tokenizer.texts_to_sequences(test_ds[textField])

        x_train = pad_sequences(X_train, maxlen=maxlen)
        x_valid = pad_sequences(X_valid, maxlen=maxlen)
        x_test =  pad_sequences(X_test, maxlen=maxlen)

        print(x_train.shape, "padded Training sequences")
        print(x_valid.shape, "padded validation sequences")
        print(x_test.shape, "padded testing sequences")

        x_train = np.asmatrix(x_train)
        x_valid = np.asmatrix(x_valid)

        training_generator = helpers._data_generator(
            x_train, y_train, maxlen, batch_size)
        validation_generator = helpers._data_generator(
            x_valid, y_valid, maxlen, batch_size)

        # Get number of training steps. This indicated the number of steps it takes
        # to cover all samples in one epoch.
        steps_per_epoch = x_train.shape[0] // batch_size
        if x_train.shape[0] % batch_size:
            steps_per_epoch += 1
        # Get number of validation steps.
        validation_steps = x_valid.shape[0] // batch_size
        if x_valid.shape[0] % batch_size:
            validation_steps += 1

        saver = ModelCheckpoint(saver_path +"/"+saver_name)
        # Logistic regression
        MLP_model = models.MLP(x_train.shape[1],vocab_size,embed_size,False)
        MLP_training_history = MLP_model.fit_generator(
            generator=training_generator,
            validation_data = validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[earlystopping,saver],
            epochs=no_epochs,
            verbose=0)

        predicted_labels = MLP_model.predict(x_test)
        print("iteration" + str(i))
        print("AUC score", roc_auc_score(test_ds[labelField], predicted_labels))
        print("F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels)))
        print("micro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        print("macro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))

        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField], predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        macro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        micro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration" + str(i))

    test_ds["MLP_prediction"] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    print("macro_f1_avg", np.mean(macro_F1_scores))
    print("micro_f1_avg", np.mean(micro_F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("macro_F1_mean: " + str(np.mean(macro_F1_scores)))
    f.write("\n")
    f.write("micro_F1_mean: " + str(np.mean(micro_F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")
def MLP_embeddings_model_training(train_ds, test_ds, textField, labelField,maxlen
                      ,embed_size, embedding_matrix,embedding_type, trainable
                      , batch_size,no_epochs, saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    # LR Model training
    AUC_scores = []
    F1_scores = []
    training_time = []
    micro_F1_scores = []
    macro_F1_scores = []

    for i in range(5):
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3, stratify=train_ds[labelField])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_ds[textField])
        vocab_size = len(tokenizer.word_index) + 1

        X_train = tokenizer.texts_to_sequences(X_train)
        X_valid = tokenizer.texts_to_sequences(X_valid)
        X_test = tokenizer.texts_to_sequences(test_ds[textField])

        x_train = pad_sequences(X_train, maxlen=maxlen)
        x_valid = pad_sequences(X_valid, maxlen=maxlen)
        x_test =  pad_sequences(X_test, maxlen=maxlen)

        print(x_train.shape, "padded Training sequences")
        print(x_valid.shape, "padded validation sequences")
        print(x_test.shape, "padded testing sequences")

        x_train = np.asmatrix(x_train)
        x_valid = np.asmatrix(x_valid)

        training_generator = helpers._data_generator(
            x_train, y_train, maxlen, batch_size)
        validation_generator = helpers._data_generator(
            x_valid, y_valid, maxlen, batch_size)

        # Get number of training steps. This indicated the number of steps it takes
        # to cover all samples in one epoch.
        steps_per_epoch = x_train.shape[0] // batch_size
        if x_train.shape[0] % batch_size:
            steps_per_epoch += 1
        # Get number of validation steps.
        validation_steps = x_valid.shape[0] // batch_size
        if x_valid.shape[0] % batch_size:
            validation_steps += 1

        saver = ModelCheckpoint(saver_path +"/"+saver_name)
        # Logistic regression
        MLP_model = models.MLP(x_train.shape[1],vocab_size,embed_size,True,embedding_matrix,trainable)
        MLP_training_history = MLP_model.fit_generator(
            generator=training_generator,
            validation_data = validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[earlystopping,saver],
            epochs=no_epochs,
            verbose=0)

        predicted_labels = MLP_model.predict(x_test)
        print("iteration" + str(i))
        print("AUC score", roc_auc_score(test_ds[labelField], predicted_labels))
        print("F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels)))
        print("micro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        print("macro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField], predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        macro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        micro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration" + str(i))

    test_ds["MLP_prediction_"+embedding_type+"_trainable_"+str(trainable)] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    print("macro_f1_avg", np.mean(macro_F1_scores))
    print("micro_f1_avg", np.mean(micro_F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("macro_F1_mean: " + str(np.mean(macro_F1_scores)))
    f.write("\n")
    f.write("micro_F1_mean: " + str(np.mean(micro_F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")
def LSTM_model_training(train_ds, test_ds, textField, labelField,maxlen,embed_size
                      ,batch_size,no_epochs, saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    AUC_scores = []
    F1_scores = []
    training_time = []
    micro_F1_scores = []
    macro_F1_scores = []

    for i in range(5):
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3, stratify=train_ds[labelField])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_ds[textField])
        vocab_size = len(tokenizer.word_index) + 1

        X_train = tokenizer.texts_to_sequences(X_train)
        X_valid = tokenizer.texts_to_sequences(X_valid)
        X_test = tokenizer.texts_to_sequences(test_ds[textField])

        x_train = pad_sequences(X_train, maxlen=maxlen)
        x_valid = pad_sequences(X_valid, maxlen=maxlen)
        x_test =  pad_sequences(X_test, maxlen=maxlen)

        print(x_train.shape, "padded Training sequences")
        print(x_valid.shape, "padded validation sequences")
        print(x_test.shape, "padded testing sequences")

        x_train = np.asmatrix(x_train)
        x_valid = np.asmatrix(x_valid)

        training_generator = helpers._data_generator(
            x_train, y_train, maxlen, batch_size)
        validation_generator = helpers._data_generator(
            x_valid, y_valid, maxlen, batch_size)

        # Get number of training steps. This indicated the number of steps it takes
        # to cover all samples in one epoch.
        steps_per_epoch = x_train.shape[0] // batch_size
        if x_train.shape[0] % batch_size:
            steps_per_epoch += 1
        # Get number of validation steps.
        validation_steps = x_valid.shape[0] // batch_size
        if x_valid.shape[0] % batch_size:
            validation_steps += 1

        saver = ModelCheckpoint(saver_path +"/"+saver_name)
        LSTM_model = models.lstm_keras(x_train.shape[1],vocab_size,embed_size,False)
        LSTM_training_history = LSTM_model.fit_generator(
            generator=training_generator,
            validation_data = validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[earlystopping,saver],
            epochs=no_epochs,
            verbose=0)

        predicted_labels = LSTM_model.predict(x_test)
        print("iteration" + str(i))
        print("AUC score", roc_auc_score(test_ds[labelField], predicted_labels))
        print("F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels)))
        print("micro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        print("macro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField], predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        macro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        micro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration" + str(i))

    test_ds["LSTM_prediction"] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    print("macro_f1_avg", np.mean(macro_F1_scores))
    print("micro_f1_avg", np.mean(micro_F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("macro_F1_mean: " + str(np.mean(macro_F1_scores)))
    f.write("\n")
    f.write("micro_F1_mean: " + str(np.mean(micro_F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")
def LSTM_embeddings_model_training(train_ds, test_ds, textField, labelField,maxlen
                      ,embed_size, embedding_matrix,embedding_type, trainable
                      , batch_size,no_epochs, saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    # LR Model training
    AUC_scores = []
    F1_scores = []
    training_time = []
    micro_F1_scores = []
    macro_F1_scores = []
    for i in range(5):
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3, stratify=train_ds[labelField])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_ds[textField])
        vocab_size = len(tokenizer.word_index) + 1

        X_train = tokenizer.texts_to_sequences(X_train)
        X_valid = tokenizer.texts_to_sequences(X_valid)
        X_test = tokenizer.texts_to_sequences(test_ds[textField])

        x_train = pad_sequences(X_train, maxlen=maxlen)
        x_valid = pad_sequences(X_valid, maxlen=maxlen)
        x_test =  pad_sequences(X_test, maxlen=maxlen)

        print(x_train.shape, "padded Training sequences")
        print(x_valid.shape, "padded validation sequences")
        print(x_test.shape, "padded testing sequences")

        x_train = np.asmatrix(x_train)
        x_valid = np.asmatrix(x_valid)

        training_generator = helpers._data_generator(
            x_train, y_train, maxlen, batch_size)
        validation_generator = helpers._data_generator(
            x_valid, y_valid, maxlen, batch_size)

        # Get number of training steps. This indicated the number of steps it takes
        # to cover all samples in one epoch.
        steps_per_epoch = x_train.shape[0] // batch_size
        if x_train.shape[0] % batch_size:
            steps_per_epoch += 1
        # Get number of validation steps.
        validation_steps = x_valid.shape[0] // batch_size
        if x_valid.shape[0] % batch_size:
            validation_steps += 1

        saver = ModelCheckpoint(saver_path +"/"+saver_name)
        LSTM_model = models.lstm_keras(x_train.shape[1],vocab_size,embed_size,True,embedding_matrix,trainable)
        LSTM_training_history = LSTM_model.fit_generator(
            generator=training_generator,
            validation_data = validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[earlystopping,saver],
            epochs=no_epochs,
            verbose=0)

        predicted_labels = LSTM_model.predict(x_test)
        print("iteration" + str(i))
        print("AUC score", roc_auc_score(test_ds[labelField], predicted_labels))
        print("F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels)))
        print("micro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        print("macro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField], predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        macro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        micro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration" + str(i))

    test_ds["LSTM_prediction_"+embedding_type+"_trainable_"+str(trainable)] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    print("macro_f1_avg", np.mean(macro_F1_scores))
    print("micro_f1_avg", np.mean(micro_F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("macro_F1_mean: " + str(np.mean(macro_F1_scores)))
    f.write("\n")
    f.write("micro_F1_mean: " + str(np.mean(micro_F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")
def LSTM_BERT_embeddings_model_training(model,tokenizer,train_ds, test_ds, textField, labelField,maxlen
                      ,embed_size
                      , batch_size,no_epochs, saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    # LR Model training
    AUC_scores = []
    F1_scores = []
    training_time = []
    micro_F1_scores = []
    macro_F1_scores = []

    for i in range(5):
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3, stratify=train_ds[labelField])

        x_train = embeddings.get_bert_concat_emb_last_4_layers(model,tokenizer,X_train,maxlen,True)
        x_valid = embeddings.get_bert_concat_emb_last_4_layers(model,tokenizer,X_valid,maxlen,True)
        x_test =  embeddings.get_bert_concat_emb_last_4_layers(model,tokenizer,test_ds[textField],maxlen,True)

        print(x_train.shape, "padded Training sequences")
        print(x_valid.shape, "padded validation sequences")
        print(x_test.shape, "padded testing sequences")

        saver = ModelCheckpoint(saver_path +"/"+saver_name)
        LSTM_model = models.lstm_bert(embed_size)
        LSTM_model.fit(x_train, y_train,
                       epochs=no_epochs,
                       batch_size=batch_size,
                       validation_data=x_valid,
                       callbacks=[earlystopping,saver],
                       verbose=0)

        predicted_labels = LSTM_model.predict(x_test)
        print("iteration" + str(i))
        print("AUC score", roc_auc_score(test_ds[labelField], predicted_labels))
        print("F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels)))
        print("micro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        print("macro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField], predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        macro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        micro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration" + str(i))

    test_ds["lstm_bert"] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    print("macro_f1_avg", np.mean(macro_F1_scores))
    print("micro_f1_avg", np.mean(micro_F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("macro_F1_mean: " + str(np.mean(macro_F1_scores)))
    f.write("\n")
    f.write("micro_F1_mean: " + str(np.mean(micro_F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")
def BiLSTM_model_training(train_ds, test_ds, textField, labelField,maxlen,embed_size
                      ,batch_size,no_epochs, saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    AUC_scores = []
    F1_scores = []
    training_time = []
    micro_F1_scores = []
    macro_F1_scores = []
    for i in range(5):
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3, stratify=train_ds[labelField])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_ds[textField])
        vocab_size = len(tokenizer.word_index) + 1

        X_train = tokenizer.texts_to_sequences(X_train)
        X_valid = tokenizer.texts_to_sequences(X_valid)
        X_test = tokenizer.texts_to_sequences(test_ds[textField])

        x_train = pad_sequences(X_train, maxlen=maxlen)
        x_valid = pad_sequences(X_valid, maxlen=maxlen)
        x_test =  pad_sequences(X_test, maxlen=maxlen)

        print(x_train.shape, "padded Training sequences")
        print(x_valid.shape, "padded validation sequences")
        print(x_test.shape, "padded testing sequences")

        x_train = np.asmatrix(x_train)
        x_valid = np.asmatrix(x_valid)

        training_generator = helpers._data_generator(
            x_train, y_train, maxlen, batch_size)
        validation_generator = helpers._data_generator(
            x_valid, y_valid, maxlen, batch_size)

        # Get number of training steps. This indicated the number of steps it takes
        # to cover all samples in one epoch.
        steps_per_epoch = x_train.shape[0] // batch_size
        if x_train.shape[0] % batch_size:
            steps_per_epoch += 1
        # Get number of validation steps.
        validation_steps = x_valid.shape[0] // batch_size
        if x_valid.shape[0] % batch_size:
            validation_steps += 1

        saver = ModelCheckpoint(saver_path +"/"+saver_name)
        BiLSTM_model = models.blstm(x_train.shape[1],vocab_size,embed_size,False)
        BiLSTM_training_history = BiLSTM_model.fit_generator(
            generator=training_generator,
            validation_data = validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[earlystopping,saver],
            epochs=no_epochs,
            verbose=0)

        predicted_labels = BiLSTM_model.predict(x_test)
        print("iteration" + str(i))
        print("AUC score", roc_auc_score(test_ds[labelField], predicted_labels))
        print("F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels)))
        print("micro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        print("macro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField], predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        macro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        micro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration" + str(i))

    test_ds["BiLSTM_prediction"] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    print("macro_f1_avg", np.mean(macro_F1_scores))
    print("micro_f1_avg", np.mean(micro_F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("macro_F1_mean: " + str(np.mean(macro_F1_scores)))
    f.write("\n")
    f.write("micro_F1_mean: " + str(np.mean(micro_F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")
def BiLSTM_embeddings_model_training(train_ds, test_ds, textField, labelField,maxlen
                      ,embed_size, embedding_matrix,embedding_type, trainable
                      , batch_size,no_epochs, saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    # LR Model training
    AUC_scores = []
    F1_scores = []
    training_time = []
    micro_F1_scores = []
    macro_F1_scores = []

    for i in range(5):
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3, stratify=train_ds[labelField])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_ds[textField])
        vocab_size = len(tokenizer.word_index) + 1

        X_train = tokenizer.texts_to_sequences(X_train)
        X_valid = tokenizer.texts_to_sequences(X_valid)
        X_test = tokenizer.texts_to_sequences(test_ds[textField])

        x_train = pad_sequences(X_train, maxlen=maxlen)
        x_valid = pad_sequences(X_valid, maxlen=maxlen)
        x_test =  pad_sequences(X_test, maxlen=maxlen)

        print(x_train.shape, "padded Training sequences")
        print(x_valid.shape, "padded validation sequences")
        print(x_test.shape, "padded testing sequences")

        x_train = np.asmatrix(x_train)
        x_valid = np.asmatrix(x_valid)

        training_generator = helpers._data_generator(
            x_train, y_train, maxlen, batch_size)
        validation_generator = helpers._data_generator(
            x_valid, y_valid, maxlen, batch_size)

        # Get number of training steps. This indicated the number of steps it takes
        # to cover all samples in one epoch.
        steps_per_epoch = x_train.shape[0] // batch_size
        if x_train.shape[0] % batch_size:
            steps_per_epoch += 1
        # Get number of validation steps.
        validation_steps = x_valid.shape[0] // batch_size
        if x_valid.shape[0] % batch_size:
            validation_steps += 1

        saver = ModelCheckpoint(saver_path +"/"+saver_name)
        BiLSTM_model = models.blstm(x_train.shape[1],vocab_size,embed_size,True,embedding_matrix,trainable)
        BiLSTM_training_history = BiLSTM_model.fit_generator(
            generator=training_generator,
            validation_data = validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[earlystopping,saver],
            epochs=no_epochs,
            verbose=0)

        predicted_labels = BiLSTM_model.predict(x_test)
        print("iteration" + str(i))
        print("AUC score", roc_auc_score(test_ds[labelField], predicted_labels))
        print("F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels)))
        print("micro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        print("macro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField], predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        macro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        micro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration" + str(i))

    test_ds["Bilstm_prediction_"+embedding_type+"_trainable_"+str(trainable)] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    print("macro_f1_avg", np.mean(macro_F1_scores))
    print("micro_f1_avg", np.mean(micro_F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("macro_F1_mean: " + str(np.mean(macro_F1_scores)))
    f.write("\n")
    f.write("micro_F1_mean: " + str(np.mean(micro_F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")

def BiLSTM_BERT_embeddings_model_training(model,tokenizer,train_ds, test_ds, textField, labelField,maxlen
                      ,embed_size
                      , batch_size,no_epochs, saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    # LR Model training
    AUC_scores = []
    F1_scores = []
    training_time = []
    micro_F1_scores = []
    macro_F1_scores = []

    for i in range(5):
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3, stratify=train_ds[labelField])

        x_train = embeddings.get_bert_concat_emb_last_4_layers(model,tokenizer,X_train,maxlen,True)
        x_valid = embeddings.get_bert_concat_emb_last_4_layers(model,tokenizer,X_valid,maxlen,True)
        x_test =  embeddings.get_bert_concat_emb_last_4_layers(model,tokenizer,test_ds[textField],maxlen,True)

        print(x_train.shape, "padded Training sequences")
        print(x_valid.shape, "padded validation sequences")
        print(x_test.shape, "padded testing sequences")

        saver = ModelCheckpoint(saver_path +"/"+saver_name)
        BiLSTM_model = models.bilstm_bert(embed_size)
        BiLSTM_model.fit(x_train, y_train, epochs=no_epochs, batch_size=batch_size, validation_data=x_valid)

        predicted_labels = BiLSTM_model.predict(x_test)
        print("iteration" + str(i))
        print("AUC score", roc_auc_score(test_ds[labelField], predicted_labels))
        print("F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels)))
        print("micro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        print("macro F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField], predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        macro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="macro"))
        micro_F1_scores.append(f1_score(test_ds[labelField], np.rint(predicted_labels), average="micro"))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration" + str(i))

    test_ds["bilstm_bert"] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    print("macro_f1_avg", np.mean(macro_F1_scores))
    print("micro_f1_avg", np.mean(micro_F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("macro_F1_mean: " + str(np.mean(macro_F1_scores)))
    f.write("\n")
    f.write("micro_F1_mean: " + str(np.mean(micro_F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")
def BiLSTM_att_model_training(train_ds, test_ds, textField, labelField,maxlen,embed_size
                      ,batch_size,no_epochs, saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    AUC_scores = []
    F1_scores = []
    training_time = []

    for i in range(5):
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3, stratify=train_ds[labelField])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_ds[textField])
        vocab_size = len(tokenizer.word_index) + 1

        X_train = tokenizer.texts_to_sequences(X_train)
        X_valid = tokenizer.texts_to_sequences(X_valid)
        X_test = tokenizer.texts_to_sequences(test_ds[textField])

        x_train = pad_sequences(X_train, maxlen=maxlen)
        x_valid = pad_sequences(X_valid, maxlen=maxlen)
        x_test =  pad_sequences(X_test, maxlen=maxlen)

        print(x_train.shape, "padded Training sequences")
        print(x_valid.shape, "padded validation sequences")
        print(x_test.shape, "padded testing sequences")

        x_train = np.asmatrix(x_train)
        x_valid = np.asmatrix(x_valid)

        training_generator = helpers._data_generator(
            x_train, y_train, maxlen, batch_size)
        validation_generator = helpers._data_generator(
            x_valid, y_valid, maxlen, batch_size)

        # Get number of training steps. This indicated the number of steps it takes
        # to cover all samples in one epoch.
        steps_per_epoch = x_train.shape[0] // batch_size
        if x_train.shape[0] % batch_size:
            steps_per_epoch += 1
        # Get number of validation steps.
        validation_steps = x_valid.shape[0] // batch_size
        if x_valid.shape[0] % batch_size:
            validation_steps += 1

        saver = ModelCheckpoint(saver_path +"/"+saver_name)
        BiLSTM_att_model = models.blstm_atten(x_train.shape[1],vocab_size,embed_size,False)
        BiLSTM_att_training_history = BiLSTM_att_model.fit_generator(
            generator=training_generator,
            validation_data = validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[earlystopping,saver],
            epochs=no_epochs,
            verbose=0)

        predicted_labels = BiLSTM_att_model.predict(x_test)
        print("iteration" + str(i))
        print("AUC score", roc_auc_score(test_ds[labelField], predicted_labels))
        print("F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels)))
        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField], predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration" + str(i))

    test_ds["BiLSTM_att_prediction"] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")
def BiLSTM_att_embeddings_model_training(train_ds, test_ds, textField, labelField,maxlen
                      ,embed_size, embedding_matrix,embedding_type
                      , batch_size,no_epochs, saver_path,
                      saver_name, results_file, earlystopping = early_stopping):
    # LR Model training
    AUC_scores = []
    F1_scores = []
    training_time = []

    for i in range(5):
        start_time = time.time()
        # split the train dataset into train and validation
        X_train, X_valid, y_train, y_valid = train_test_split(train_ds[textField], train_ds[labelField], test_size=0.3,stratify=train_ds[labelField])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_ds[textField])
        vocab_size = len(tokenizer.word_index) + 1

        X_train = tokenizer.texts_to_sequences(X_train)
        X_valid = tokenizer.texts_to_sequences(X_valid)
        X_test = tokenizer.texts_to_sequences(test_ds[textField])

        x_train = pad_sequences(X_train, maxlen=maxlen)
        x_valid = pad_sequences(X_valid, maxlen=maxlen)
        x_test =  pad_sequences(X_test, maxlen=maxlen)

        print(x_train.shape, "padded Training sequences")
        print(x_valid.shape, "padded validation sequences")
        print(x_test.shape, "padded testing sequences")

        x_train = np.asmatrix(x_train)
        x_valid = np.asmatrix(x_valid)

        training_generator = helpers._data_generator(
            x_train, y_train, maxlen, batch_size)
        validation_generator = helpers._data_generator(
            x_valid, y_valid, maxlen, batch_size)

        # Get number of training steps. This indicated the number of steps it takes
        # to cover all samples in one epoch.
        steps_per_epoch = x_train.shape[0] // batch_size
        if x_train.shape[0] % batch_size:
            steps_per_epoch += 1
        # Get number of validation steps.
        validation_steps = x_valid.shape[0] // batch_size
        if x_valid.shape[0] % batch_size:
            validation_steps += 1

        saver = ModelCheckpoint(saver_path +"/"+saver_name)
        BiLSTM_att_model = models.blstm_atten(x_train.shape[1],vocab_size,embed_size,True,embedding_matrix,False)
        BiLLSTM_att_training_history = BiLSTM_att_model.fit_generator(
            generator=training_generator,
            validation_data = validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[earlystopping,saver],
            epochs=no_epochs,
            verbose=0)

        predicted_labels = BiLSTM_att_model.predict(x_test)
        print("iteration" + str(i))
        print("AUC score", roc_auc_score(test_ds[labelField], predicted_labels))
        print("F1 score", f1_score(test_ds[labelField], np.rint(predicted_labels)))
        exc_time = time.time() - start_time
        AUC_scores.append(roc_auc_score(test_ds[labelField], predicted_labels))
        F1_scores.append(f1_score(test_ds[labelField],np.rint(predicted_labels)))
        training_time.append(exc_time)
        keras.backend.clear_session()
        print("End iteration" + str(i))

    test_ds["BiLSTM_att_prediction_"+embedding_type] = predicted_labels
    print("AUC_avg", np.mean(AUC_scores))
    print("f1_avg", np.mean(F1_scores))
    f = open(results_file, "w")
    f.write(saver_name)
    f.write("\n")
    f.write("AUC_mean: " + str(np.mean(AUC_scores)))
    f.write("\n")
    f.write("F1_mean: " + str(np.mean(F1_scores)))
    f.write("\n")
    f.write("Excution Time: " + str(np.mean(training_time)))
    f.write("--------------------------------------------------------------------------------")
    f.write("\n")
    f.close()
    return test_ds
    print("Done!")