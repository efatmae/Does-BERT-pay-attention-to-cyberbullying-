import sys
import pandas as pd
import models_training,helpers, embeddings
def main(argv):
    print(argv)
    #parameter = argv[1]
    models = ["LR","MLP_KERAS_EMB"]
    train_df = pd.read_csv("../../../Data/Twitter_sexism/Twitter_sex_data_train.csv", index_col=False)
    test_df = pd.read_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index_col=False)
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    word_dictionary = helpers.get_word_dictionary(train_df, "Text_clean")
    batch_size = 32
    no_epochs = 100
    maxlen = 23
    embedding_size = 300
    for parameter in models:
        ######################################################################################################
        if parameter == "LR":
            print(parameter)
            #LR model settings
            max_features = 10000  # Only consider the top 20k words
            saver_path = "../../trained_models/LR/"
            saver_name = "Twitter_sexism.h5"
            lr_results_files = "../Results/LR/Twitter_sexism_results.txt"

            predicted_test_Set = models_training.LR_model_training(train_df,test_df,"Text_clean","oh_label",
                              max_features,batch_size,no_epochs,
                              saver_path,saver_name,lr_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv",index=False)
        ################################################################################################################
        ##MLP model settings with Glove embedding
        elif parameter == "MLP_KERAS_EMB":
            ##MLP model settings with Keras embeddings
            saver_path = "../../trained_models/MLP/"
            saver_name = "Twitter_sexism_keras_emb.h5"
            MLP_keras_emb_results_files = "../Results/MLP/Twitter_sexism_results_keras_emb.txt"

            predicted_test_Set = models_training.MLP_model_training(train_df, test_df,"Text_clean","oh_label",
                                   maxlen,embedding_size,batch_size,no_epochs,
                                    saver_path,
                                   saver_name, MLP_keras_emb_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv",index=False)
        #####################################################################################################################
        elif parameter == "MLP_GLOVE_EMB":
            saver_path = "../../trained_models/MLP/"
            saver_name = "Twitter_sexism_twitter_glove_emb.h5"
            MLP_twitter_glove_results_files = "../Results/MLP/Twitter_sexism_results_twitter_glove.txt"

            glove_Twitter_embeddings_matrix = embeddings.get_Glove_embeddings("../../../Data/Glove/glove.twitter.27B/glove.twitter.27B.200d.txt",
                                                                              word_dictionary,
                                                                              len(word_dictionary)+1,
                                                                              200)
            predicted_test_Set = models_training.MLP_embeddings_model_training(train_df, test_df,"Text_clean","oh_label",maxlen
                                  ,200, glove_Twitter_embeddings_matrix,"glove_twitter"
                                  , batch_size,no_epochs,saver_path,
                                  saver_name, MLP_twitter_glove_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv",index=False)
        ###################################################################################################################
        elif parameter == "MLP_GLOVE_CC_EMB":
            saver_path = "../../trained_models/MLP/"
            saver_name = "Twitter_sexism_glove_cc_emb.h5"
            MLP_cc_glove_results_files = "../Results/MLP/Twitter_sexism_results_cc_glove.txt"

            glove_cc_embeddings_matrix = embeddings.get_Glove_embeddings(
                "../../../Data/Glove/glove.42B.300d.txt",
                word_dictionary,
                len(word_dictionary) + 1,
                300)
            predicted_test_Set = models_training.MLP_embeddings_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                               maxlen
                                                                               ,300, glove_cc_embeddings_matrix,
                                                                               "glove_cc"
                                                                               , batch_size, no_epochs, saver_path,
                                                                               saver_name, MLP_cc_glove_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        ###################################################################################################################
        elif parameter == "MLP_SSWE_EMB":
            saver_path = "../../trained_models/MLP/"
            saver_name = "Twitter_sexism_sswe_u_emb.h5"
            MLP_sswe_results_files = "../Results/MLP/Twitter_sexism_results_sswe_u.txt"

            sswe_Twitter_embeddings_matrix = embeddings.get_sswe_embeddings("../../../Data/SSWE/embedding-results/sswe-u.bin",
                                                                              word_dictionary,
                                                                              len(word_dictionary)+1,
                                                                              50)
            predicted_test_Set = models_training.MLP_embeddings_model_training(train_df, test_df, "Text_clean","oh_label",maxlen
                                  ,50, sswe_Twitter_embeddings_matrix,"sswe_u"
                                  , batch_size,no_epochs,saver_path,
                                  saver_name, MLP_sswe_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv",index=False)
        #################################################################################################################
        elif parameter == "MLP_UD_EMB":
            saver_path = "../../trained_models/MLP/"
            saver_name = "Twitter_sexism_ud_emb.h5"
            MLP_ud_results_files = "../Results/MLP/Twitter_sexism_results_ud.txt"
            UD_embeddings_matrix = embeddings.get_UD_embeddings("../../../Data/ud_embeddings/ud_basic.vec",
                                                                              word_dictionary,
                                                                              len(word_dictionary)+1,
                                                                              300)
            predicted_test_Set = models_training.MLP_embeddings_model_training(train_df, test_df,"Text_clean","oh_label",maxlen
                                  ,300, UD_embeddings_matrix,"ud"
                                  , batch_size,no_epochs,saver_path,
                                  saver_name, MLP_ud_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv",index=False)
        ###################################################################################################################
        elif parameter == "MLP_GLOVE_WK_EMB":
            saver_path = "../../trained_models/MLP/"
            saver_name = "Twitter_sexism_glove_wk_emb.h5"
            MLP_glove_wk_results_files = "../Results/MLP/Twitter_sexism_results_glove_wk.txt"

            glove_wk_embeddings_matrix = embeddings.get_Glove_embeddings("../../../Data/Glove/glove.6B/glove.6B.200d.txt",
                                                                              word_dictionary,
                                                                              len(word_dictionary)+1,
                                                                              200)
            predicted_test_Set = models_training.MLP_embeddings_model_training(train_df, test_df, "Text_clean","oh_label",maxlen
                                  ,200, glove_wk_embeddings_matrix,"glove_wk"
                                  , batch_size,no_epochs,saver_path,
                                  saver_name, MLP_glove_wk_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv",index=False)
        #######################################################################################################################
        elif parameter == "MLP_W2V_EMB":
            saver_path = "../../trained_models/MLP/"
            saver_name = "Twitter_sexism_w2v_emb.h5"
            MLP_w2v_news_results_files = "../Results/MLP/Twitter_sexism_results_w2v_news.txt"

            w2v_new_embeddings_matrix = embeddings.get_google_news_embeddings(
                "../../../Data/Google_news/GoogleNews-vectors-negative300.bin",
                word_dictionary,
                len(word_dictionary) + 1,
                300)
            predicted_test_Set = models_training.MLP_embeddings_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                               maxlen
                                                                               ,300, w2v_new_embeddings_matrix,
                                                                               "w2v_news"
                                                                               , batch_size, no_epochs, saver_path,
                                                                               saver_name, MLP_w2v_news_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #######################################################################################################################
        #####################################################################################################################
        ### LSTM model settings ###############
        elif parameter == "LSTM_KERAS_EMB":
            saver_path = "../../trained_models/LSTM/"
            saver_name = "Twitter_sexism_keras_emb.h5"
            LSTM_keras_emb_results_files = "../Results/LSTM/Twitter_sexism_results_keras_emb.txt"

            predicted_test_Set = models_training.LSTM_model_training(train_df, test_df,"Text_clean","oh_label",
                                   maxlen,embedding_size,batch_size,no_epochs,
                                    saver_path,
                                   saver_name, LSTM_keras_emb_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #####################################################################################################################
        elif parameter == "LSTM_GLOVE_EMB":
            saver_path = "../../trained_models/LSTM/"
            saver_name = "Twitter_sexism_twitter_glove_emb.h5"
            LSTM_twitter_glove_results_files = "../Results/LSTM/Twitter_sexism_results_twitter_glove.txt"

            glove_Twitter_embeddings_matrix = embeddings.get_Glove_embeddings(
                "../../../Data/Glove/glove.twitter.27B/glove.twitter.27B.200d.txt",
                word_dictionary,
                len(word_dictionary) + 1,
                200)
            predicted_test_Set = models_training.LSTM_embeddings_model_training(train_df, test_df, "Text_clean",
                                                                               "oh_label", maxlen
                                                                               , 200, glove_Twitter_embeddings_matrix,
                                                                               "glove_twitter"
                                                                               , batch_size, no_epochs, saver_path,
                                                                               saver_name,
                                                                               LSTM_twitter_glove_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #######################################################################################################################
        elif parameter == "LSTM_GLOVE_CC_EMB":
            saver_path = "../../trained_models/LSTM/"
            saver_name = "Twitter_sexism_glove_cc_emb.h5"
            LSTM_cc_glove_results_files = "../Results/LSTM/Twitter_sexism_results_cc_glove.txt"

            glove_cc_embeddings_matrix = embeddings.get_Glove_embeddings(
                "../../../Data/Glove/glove.42B.300d.txt",
                word_dictionary,
                len(word_dictionary) + 1,
                300)
            predicted_test_Set = models_training.LSTM_embeddings_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                               maxlen
                                                                               ,300, glove_cc_embeddings_matrix,
                                                                               "glove_cc"
                                                                               , batch_size, no_epochs, saver_path,
                                                                               saver_name, LSTM_cc_glove_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        ###################################################################################################################
        elif parameter == "LSTM_SSWE_EMB":
            saver_path = "../../trained_models/LSTM/"
            saver_name = "Twitter_sexism_sswe_u_emb.h5"
            LSTM_sswe_results_files = "../Results/LSTM/Twitter_sexism_results_sswe_u.txt"

            sswe_Twitter_embeddings_matrix = embeddings.get_sswe_embeddings("../../../Data/SSWE/embedding-results/sswe-u.bin",
                                                                              word_dictionary,
                                                                              len(word_dictionary)+1,
                                                                              50)
            predicted_test_Set = models_training.LSTM_embeddings_model_training(train_df, test_df, "Text_clean","oh_label",maxlen
                                  ,50, sswe_Twitter_embeddings_matrix,"sswe_u"
                                  , batch_size,no_epochs,saver_path,
                                  saver_name, LSTM_sswe_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv",index=False)
        #################################################################################################################
        elif parameter == "LSTM_UD_EMB":
            saver_path = "../../trained_models/LSTM/"
            saver_name = "Twitter_sexism_ud_emb.h5"
            LSTM_ud_results_files = "../Results/LSTM/Twitter_sexism_results_ud.txt"
            UD_embeddings_matrix = embeddings.get_UD_embeddings("../../../Data/ud_embeddings/ud_basic.vec",
                                                                              word_dictionary,
                                                                              len(word_dictionary)+1,
                                                                              300)
            predicted_test_Set = models_training.LSTM_embeddings_model_training(train_df, test_df,"Text_clean","oh_label",maxlen
                                  ,300, UD_embeddings_matrix,"ud"
                                  , batch_size,no_epochs,saver_path,
                                  saver_name, LSTM_ud_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv",index=False)
        ###################################################################################################################
        elif parameter == "LSTM_GLOVE_WK_EMB":
            saver_path = "../../trained_models/LSTM/"
            saver_name = "Twitter_sexism_glove_wk_emb.h5"
            LSTM_glove_wk_results_files = "../Results/LSTM/Twitter_sexism_results_glove_wk.txt"

            glove_wk_embeddings_matrix = embeddings.get_Glove_embeddings("../../../Data/Glove/glove.6B/glove.6B.200d.txt",
                                                                              word_dictionary,
                                                                              len(word_dictionary)+1,
                                                                              200)
            predicted_test_Set = models_training.LSTM_embeddings_model_training(train_df, test_df, "Text_clean","oh_label",maxlen
                                  ,200, glove_wk_embeddings_matrix,"glove_wk"
                                  , batch_size,no_epochs,saver_path,
                                  saver_name, LSTM_glove_wk_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv",index=False)
        #######################################################################################################################
        elif parameter == "LSTM_W2V_EMB":
            saver_path = "../../trained_models/LSTM/"
            saver_name = "Twitter_sexism_w2v_emb.h5"
            LSTM_w2v_news_results_files = "../Results/LSTM/Twitter_sexism_results_w2v_news.txt"

            w2v_new_embeddings_matrix = embeddings.get_google_news_embeddings(
                "../../../Data/Google_news/GoogleNews-vectors-negative300.bin",
                word_dictionary,
                len(word_dictionary) + 1,
                300)
            predicted_test_Set = models_training.LSTM_embeddings_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                               maxlen
                                                                               ,300, w2v_new_embeddings_matrix,
                                                                               "w2v_news"
                                                                               , batch_size, no_epochs, saver_path,
                                                                               saver_name, LSTM_w2v_news_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #######################################################################################################################
        #####################################################################################################################
        ### BILSTM model settings ###############
        elif parameter == "BILSTM_KERAS_EMB":
            saver_path = "../../trained_models/BiLSTM/"
            saver_name = "Twitter_sexism_keras_emb.h5"
            BiLSTM_keras_emb_results_files = "../Results/BiLSTM/Twitter_sexism_results_keras_emb.txt"

            predicted_test_Set = models_training.BiLSTM_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                     maxlen, embedding_size, batch_size, no_epochs,
                                                                     saver_path,
                                                                     saver_name, BiLSTM_keras_emb_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #####################################################################################################################
        elif parameter == "BILSTM_GLOVE_EMB":
            saver_path = "../../trained_models/BiLSTM/"
            saver_name = "Twitter_sexism_twitter_glove_emb.h5"
            BiLSTM_twitter_glove_results_files = "../Results/BiLSTM/Twitter_sexism_results_twitter_glove.txt"

            glove_Twitter_embeddings_matrix = embeddings.get_Glove_embeddings(
                "../../../Data/Glove/glove.twitter.27B/glove.twitter.27B.200d.txt",
                word_dictionary,
                len(word_dictionary) + 1,
                200)
            predicted_test_Set = models_training.BiLSTM_embeddings_model_training(train_df, test_df, "Text_clean",
                                                                                "oh_label", maxlen
                                                                                , 200, glove_Twitter_embeddings_matrix,
                                                                                "glove_twitter"
                                                                                , batch_size, no_epochs, saver_path,
                                                                                saver_name,
                                                                                BiLSTM_twitter_glove_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #######################################################################################################################
        elif parameter == "BILSTM_GLOVE_CC_EMB":
            saver_path = "../../trained_models/BiLSTM/"
            saver_name = "Twitter_sexism_glove_cc_emb.h5"
            BiLSTM_cc_glove_results_files = "../Results/BiLSTM/Twitter_sexism_results_cc_glove.txt"

            glove_cc_embeddings_matrix = embeddings.get_Glove_embeddings(
                "../../../Data/Glove/glove.42B.300d.txt",
                word_dictionary,
                len(word_dictionary) + 1,
                300)
            predicted_test_Set = models_training.BiLSTM_embeddings_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                                maxlen
                                                                                , 300, glove_cc_embeddings_matrix,
                                                                                "glove_cc"
                                                                                , batch_size, no_epochs, saver_path,
                                                                                saver_name, BiLSTM_cc_glove_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        ###################################################################################################################
        elif parameter == "BILSTM_SSWE_EMB":
            saver_path = "../../trained_models/BiLSTM/"
            saver_name = "Twitter_sexism_sswe_u_emb.h5"
            BiLSTM_sswe_results_files = "../Results/BiLSTM/Twitter_sexism_results_sswe_u.txt"

            sswe_Twitter_embeddings_matrix = embeddings.get_sswe_embeddings(
                "../../../Data/SSWE/embedding-results/sswe-u.bin",
                word_dictionary,
                len(word_dictionary) + 1,
                50)
            predicted_test_Set = models_training.BiLSTM_embeddings_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                                maxlen
                                                                                , 50, sswe_Twitter_embeddings_matrix,
                                                                                "sswe_u"
                                                                                , batch_size, no_epochs, saver_path,
                                                                                saver_name, BiLSTM_sswe_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #################################################################################################################
        elif parameter == "BILSTM_UD_EMB":
            saver_path = "../../trained_models/BiLSTM/"
            saver_name = "Twitter_sexism_ud_emb.h5"
            BiLSTM_ud_results_files = "../Results/BiLSTM/Twitter_sexism_results_ud.txt"
            UD_embeddings_matrix = embeddings.get_UD_embeddings("../../../Data/ud_embeddings/ud_basic.vec",
                                                                word_dictionary,
                                                                len(word_dictionary) + 1,
                                                                300)
            predicted_test_Set = models_training.BiLSTM_embeddings_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                                maxlen
                                                                                , 300, UD_embeddings_matrix, "ud"
                                                                                , batch_size, no_epochs, saver_path,
                                                                                saver_name, BiLSTM_ud_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        ###################################################################################################################
        elif parameter == "BILSTM_GLOVE_WK_EMB":
            saver_path = "../../trained_models/BiLSTM/"
            saver_name = "Twitter_sexism_glove_wk_emb.h5"
            BiLSTM_glove_wk_results_files = "../Results/BiLSTM/Twitter_sexism_results_glove_wk.txt"

            glove_wk_embeddings_matrix = embeddings.get_Glove_embeddings("../../../Data/Glove/glove.6B/glove.6B.200d.txt",
                                                                         word_dictionary,
                                                                         len(word_dictionary) + 1,
                                                                         200)
            predicted_test_Set = models_training.BiLSTM_embeddings_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                                maxlen
                                                                                , 200, glove_wk_embeddings_matrix,
                                                                                "glove_wk"
                                                                                , batch_size, no_epochs, saver_path,
                                                                                saver_name, BiLSTM_glove_wk_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #######################################################################################################################
        elif parameter == "BILSTM_W2V_EMB":
            saver_path = "../../trained_models/BiLSTM/"
            saver_name = "Twitter_sexism_w2v_emb.h5"
            BiLSTM_w2v_news_results_files = "../Results/BiLSTM/Twitter_sexism_results_w2v_news.txt"

            w2v_new_embeddings_matrix = embeddings.get_google_news_embeddings(
                "../../../Data/Google_news/GoogleNews-vectors-negative300.bin",
                word_dictionary,
                len(word_dictionary) + 1,
                300)
            predicted_test_Set = models_training.BiLSTM_embeddings_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                                maxlen
                                                                                , 300, w2v_new_embeddings_matrix,
                                                                                "w2v_news"
                                                                                , batch_size, no_epochs, saver_path,
                                                                                saver_name, BiLSTM_w2v_news_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #######################################################################################################################
        #####################################################################################################################
        ### BILSTM_attention model settings ###############
        elif parameter == "BILSTM_att_KERAS_EMB":
            saver_path = "../../trained_models/BiLSTM_atten/"
            saver_name = "Twitter_sexism_keras_emb.h5"
            BiLSTM_atten_keras_emb_results_files = "../Results/BiLSTM_atten/Twitter_sexism_results_keras_emb.txt"

            predicted_test_Set = models_training.BiLSTM_att_model_training(train_df, test_df, "Text_clean", "oh_label",
                                                                       maxlen, embedding_size, batch_size, no_epochs,
                                                                       saver_path,
                                                                       saver_name, BiLSTM_atten_keras_emb_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #######################################################################################################################
        elif parameter == "BILSTM_att_GLOVE_CC_EMB":
            saver_path = "../../trained_models/BiLSTM_atten/"
            saver_name = "Twitter_sexism_glove_cc_emb.h5"
            BiLSTM_att_cc_glove_results_files = "../Results/BiLSTM_atten/Twitter_sexism_results_cc_glove.txt"

            glove_cc_embeddings_matrix = embeddings.get_Glove_embeddings(
                "../../../Data/Glove/glove.42B.300d.txt",
                word_dictionary,
                len(word_dictionary) + 1,
                300)
            predicted_test_Set = models_training.BiLSTM_att_embeddings_model_training(train_df, test_df, "Text_clean",
                                                                                  "oh_label",
                                                                                  maxlen
                                                                                  , 300, glove_cc_embeddings_matrix,
                                                                                  "glove_cc"
                                                                                  , batch_size, no_epochs, saver_path,
                                                                                  saver_name, BiLSTM_att_cc_glove_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        ###################################################################################################################
        elif parameter == "BILSTM_att_SSWE_EMB":
            saver_path = "../../trained_models/BiLSTM_atten/"
            saver_name = "Twitter_sexism_sswe_u_emb.h5"
            BiLSTM_att_sswe_results_files = "../Results/BiLSTM_atten/Twitter_sexism_results_sswe_u.txt"

            sswe_Twitter_embeddings_matrix = embeddings.get_sswe_embeddings(
                "../../../Data/SSWE/embedding-results/sswe-u.bin",
                word_dictionary,
                len(word_dictionary) + 1,
                50)
            predicted_test_Set = models_training.BiLSTM_att_embeddings_model_training(train_df, test_df, "Text_clean",
                                                                                  "oh_label",
                                                                                  maxlen
                                                                                  , 50, sswe_Twitter_embeddings_matrix,
                                                                                  "sswe_u"
                                                                                  , batch_size, no_epochs, saver_path,
                                                                                  saver_name, BiLSTM_att_sswe_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #################################################################################################################
        elif parameter == "BILSTM_att_UD_EMB":
            saver_path = "../../trained_models/BiLSTM_atten/"
            saver_name = "Twitter_sexism_ud_emb.h5"
            BiLSTM_att_ud_results_files = "../Results/BiLSTM_atten/Twitter_sexism_results_ud.txt"
            UD_embeddings_matrix = embeddings.get_UD_embeddings("../../../Data/ud_embeddings/ud_basic.vec",
                                                                word_dictionary,
                                                                len(word_dictionary) + 1,
                                                                300)
            predicted_test_Set = models_training.BiLSTM_att_embeddings_model_training(train_df, test_df, "Text_clean",
                                                                                  "oh_label",
                                                                                  maxlen
                                                                                  , 300, UD_embeddings_matrix, "ud"
                                                                                  , batch_size, no_epochs, saver_path,
                                                                                  saver_name, BiLSTM_att_ud_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        ###################################################################################################################
        elif parameter == "BILSTM_att_GLOVE_WK_EMB":
            saver_path = "../../trained_models/BiLSTM_atten/"
            saver_name = "Twitter_sexism_glove_wk_emb.h5"
            BiLSTM_att_glove_wk_results_files = "../Results/BiLSTM_atten/Twitter_sexism_results_glove_wk.txt"

            glove_wk_embeddings_matrix = embeddings.get_Glove_embeddings("../../../Data/Glove/glove.6B/glove.6B.200d.txt",
                                                                         word_dictionary,
                                                                         len(word_dictionary) + 1,
                                                                         200)
            predicted_test_Set = models_training.BiLSTM_att_embeddings_model_training(train_df, test_df, "Text_clean",
                                                                                  "oh_label",
                                                                                  maxlen
                                                                                  , 200, glove_wk_embeddings_matrix,
                                                                                  "glove_wk"
                                                                                  , batch_size, no_epochs, saver_path,
                                                                                  saver_name, BiLSTM_att_glove_wk_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        #######################################################################################################################
        elif parameter == "BILSTM_att_W2V_EMB":
            saver_path = "../../trained_models/BiLSTM_atten/"
            saver_name = "Twitter_sexism_w2v_emb.h5"
            BiLSTM_att_w2v_news_results_files = "../Results/BiLSTM_atten/Twitter_sexism_results_w2v_news.txt"

            w2v_new_embeddings_matrix = embeddings.get_google_news_embeddings(
                "../../../Data/Google_news/GoogleNews-vectors-negative300.bin",
                word_dictionary,
                len(word_dictionary) + 1,
                300)
            predicted_test_Set = models_training.BiLSTM_att_embeddings_model_training(train_df, test_df, "Text_clean",
                                                                                  "oh_label",
                                                                                  maxlen
                                                                                  , 300, w2v_new_embeddings_matrix,
                                                                                  "w2v_news"
                                                                                  , batch_size, no_epochs, saver_path,
                                                                                  saver_name, BiLSTM_att_w2v_news_results_files)
            predicted_test_Set.to_csv("../../../Data/Twitter_sexism/Twitter_sex_data_test.csv", index=False)
        else:
            print("paeametr not found")
if __name__ == "__main__":
   main(sys.argv) #