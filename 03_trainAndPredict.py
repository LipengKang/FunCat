import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import datetime
import os
import getopt
import sys
from keras import layers
from keras import models
from keras.engine.input_layer import Input
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


### Processing file 
X_shape = -1
cat_ind = -1
funcat_version = 0.3

def print_help():
    print("Funcat version:", funcat_version, "\n")
    print("Usgae:")
    print("\t-c: the index of the one-hot columns")
    print("\t-i: the directory of the input data")
    print("\t\t--input")
    print("\t-t: the training model, chosen from nn, sl, rf")
    print("\t\t--train")
    print("\t-m: the existing model location")
    print("\t\t--model")
    print("\t-h: print help")
    print("\t\t--help")
    print("\t-v: print version")
    print("\t\t--version\n")
    print("Example:")
    print("python3 funcat.py -i ./data/ -t sl -c 31 ")    

def returnXY(filename1):
    data = pd.read_table(filename1,header=None,na_values=['.']).iloc[:,3:]
    # print(data.shape)
    data.dropna(inplace = True)

    if cat_ind > 0:
        data = pd.concat([data.reset_index(drop=True), pd.DataFrame(to_categorical(data.iloc[:,cat_ind - 4],num_classes=16,dtype='int64'),)], axis=1)
        data.columns = range(4,data.shape[1] + 4)
        data.drop(data.columns[cat_ind-4], axis = 1, inplace=True)

    Y = data.iloc[:,0]
    X = data.iloc[:,1:(data.shape[1])]
    return X ,Y

def return_single_XY(filename1):
    data = pd.read_table(filename1,header=None,na_values=['.']).iloc[:,3:]
    # print(data.shape)
    data.dropna(inplace = True)

    # print(data.iloc[:,cat_ind - 4])
    # print("========")
    # if cat_ind > 0:
    #     data = pd.concat([data.reset_index(drop=True), pd.DataFrame(to_categorical(data.iloc[:,cat_ind - 4],num_classes=16,dtype='int64'),)], axis=1)
    #     data.columns = range(3,data.shape[1] + 3)
    #     # data.drop(data.columns[cat_ind-4], axis = 1, inplace=True)

    # print(data.iloc[:,cat_ind - 4])
    Y = data.iloc[:,0]
    X = data.iloc[:,4:10]
    return X ,Y

def return_double_XY(filename1):
    data = pd.read_table(filename1,header=None,na_values=['.']).iloc[:,3:]
    # print(data.shape)
    data.dropna(inplace = True)

    # print(data.iloc[:,cat_ind - 4])
    # print("========")
    # if cat_ind > 0:
    #     data = pd.concat([data.reset_index(drop=True), pd.DataFrame(to_categorical(data.iloc[:,cat_ind - 4],num_classes=16,dtype='int64'),)], axis=1)
    #     data.columns = range(3,data.shape[1] + 3)
    #     # data.drop(data.columns[cat_ind-4], axis = 1, inplace=True)

    # print(data.iloc[:,cat_ind - 4])
    Y1 = data.iloc[:,0]
    Y2 = data.iloc[:,1]
    X1 = data.iloc[:,4:21]
    Y = pd.concat([Y1,Y2])
    X = pd.concat([X1,X1])

    return X ,Y

def return_trible_XY(filename1):
    data = pd.read_table(filename1,header=None,na_values=['.']).iloc[:,3:]
    # print(data.shape)
    data.dropna(inplace = True)

    # print(data.iloc[:,cat_ind - 4])
    # print("========")
    if cat_ind > 0:
        data = pd.concat([data.reset_index(drop=True), pd.DataFrame(to_categorical(data.iloc[:,cat_ind - 4],num_classes=16,dtype='int64'),)], axis=1)
        data.columns = range(3,data.shape[1] + 3)
        # data.drop(data.columns[cat_ind-4], axis = 1, inplace=True)

    # print(data.iloc[:,cat_ind - 4])
    Y1 = data.iloc[:,0]
    Y2 = data.iloc[:,1]
    Y3 = data.iloc[:,2]
    X1 = data.iloc[:,4:(data.shape[1])]

    Y = pd.concat([Y1,Y2,Y3])
    X = pd.concat([X1,X1,X1])
    return X ,Y


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(X_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])
    return model


def build_model_random_forest():
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=60, # 20 trees
                            max_depth=3,
                            warm_start = True)
    return model


def build_model_simple_linear():
    model = models.Sequential()
    model.add(layers.Dense(1, activation='relu', input_shape=(X_shape,)))

    model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])
    return model

### Process command line option
opts,args = getopt.getopt(sys.argv[1:],'m:i:t:vhc:',['model=','input=','train=','version','help'])

model_class = " "
save_model_dic = "./"
input_dic = " "

for opt_name,opt_value in opts:
    if opt_name in ('-h','--help'):
        print_help()
        exit()
    if opt_name in ('-v','--version'):
        print("Funcat version:", funcat_version, "\n\n")
        exit()
    if opt_name in ('-t','--train'):
        model_class = opt_value
    if opt_name in ('-m','--model'):
        save_model_dic = opt_value
    if opt_name in ('-i', '--input'):
        input_dic = opt_value
    if opt_name in ('-c'):
        cat_ind = int(opt_value)

if model_class == " ":
    print("[W] Option -m requires argument")
    exit()

if input_dic == " ":
    print("[W] Option -i requires argument")
    exit()

print("[I] Using model:", model_class)

model = models.Sequential()

# if save_model_dic == "./":
#     print("[I] Model save directory:", save_model_dic)
#     model_save_name = " "
    
#     if model_class == "sl":
#         model_save_name = "./sl.model"
#         model = build_model_simple_linear()
#     elif model_class == "nn":
#         model_save_name = "./nn.model"
#         model = build_model()
#     elif model_class == "rf":
#         model_save_name = "./rf.model"
#         model = build_model_random_forest()
#     else:
#         print("[W] Please enter the right model class!")
#         exit()
# else:
#     print("[I] Using exist model:", save_model_dic)
#     model = models.load_model(save_model_dic)
    
print("[I] Using input directory:" , input_dic)


# model_nn = build_model()
# model_sl = build_model_simple_linear()

file_list = os.listdir(input_dic)
count = 0 

for file in file_list:
    if "split" in file:
        filename1 = input_dic + file
        
        X1, Y1 = returnXY(filename1)

        if count == 0:
            X_shape = X1.shape[1]
            if save_model_dic == "./":
                print("[I] Model save directory:", save_model_dic)
                model_save_name = " "
    
                if model_class == "sl":
                    model_save_name = "./sl.model"
                    model = build_model_simple_linear()
                elif model_class == "nn":
                    model_save_name = "./nn.model"
                    model = build_model()
                elif model_class == "rf":
                    model_save_name = "./rf.model"
                    model = build_model_random_forest()
                else:
                    print("[W] Please enter the right model class!")
                    exit()
            else:
                print("[I] Using exist model:", save_model_dic)
                if model_class == 'rf':
                    model = joblib.load(save_model_dic)
                else:
                    model = models.load_model(save_model_dic)

        count += 1
        time_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        print("[I] Training the file:", file)
        print("[I]", time_start , file, model_class, "training starts.")

        if model_class == "rf":
            if X1.shape[0] > 0:
                model.fit(X1, Y1)
        else:
            if X1.shape[0] > 0:
                model.fit(X1, Y1, epochs=2, batch_size=16, verbose=1)

        # time_middle = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print("[I]", time_middle , file, "sl training starts.")
        # model_sl.fit(X1, Y1, epochs=2, batch_size=16, verbose=1)

        time_end = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("[I]", time_end , file, "training ends.\n")
    

if model_class == "rf":
    import joblib
    joblib.dump(model, model_save_name)
else:
    model.save(model_save_name)

def returnXY_old(filename1, filename2):
    data = pd.read_table(filename1,header=None,na_values=['.']).iloc[:,3:]
    Y = pd.read_table(filename2,header=None,na_values=['.'])

    Y = Y.loc[:,3].replace('\.+', np.nan, regex=True)
    Y = np.asarray(Y).astype(np.float32)
    data = pd.concat([Y.iloc[:,3:],data.iloc[:,3:]], axis=1)

    subs = data == "."
    data[subs] = np.nan

    hs_level = data[29]
    print(hs_level.loc[0:5,])
    hs_level_oh = to_categorical(data[29],num_classes=16)
    print(hs_level_oh.loc[0:5,])
    print(hs_level_oh.shape)
    data = pd.concat([data.reset_index(drop=True), pd.DataFrame(to_categorical(data[30],num_classes=16),)], axis=1)
    new_data = pd.concat([data.reset_index(drop=True), pd.DataFrame(hs_level_oh,)], axis=1)
    print(new_data.iloc[:,29:46])


    X = new_data.iloc[:,3:]
    nn = pd.concat([pd.DataFrame(Y), X], axis=1)
    data.dropna(inplace = True)
    Y = data.iloc[:,0]
    Y = np.asarray(Y).astype(np.float32)
    X = data.iloc[:,1:(data.shape[1])]
    return X ,Y

def eva(output_file):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.metrics import roc_curve
    file_list = os.listdir(input_dic)


    fw = open(output_file, "w")

    fw.write("file\tMAE\tMSE\tRMSE\n")
    for file in file_list:
        if "split" in file:
            filename1 = input_dic + file
            # print("[I] Training the file:", file)
            X1, Y1 = returnXY(filename1)

            fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y1, model.predict(X1))

            fw.write(file)
            fw.write("\t")
            fw.write(str(mean_absolute_error(Y1,model.predict(X1))))
            fw.write("\t")
            fw.write(str(mean_squared_error(Y1,model.predict(X1))))
            fw.write("\t")
            fw.write(str(np.sqrt(mean_squared_error(Y1,model.predict(X1)))))
            fw.write("\n")
        # print(file+"\t"+mean_absolute_error(Y1,model.predict(X1))+"\t" + mean_squared_error(Y1,model.predict(X1)), np.sqrt(mean_squared_error(Y1,model.predict(X1))))
    fw.close()




n_estimators = [5,20,50] # number of trees in the random forest
max_features = ['auto'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(100, 120, num = 2)] # maximum number of levels allowed in each decision tree
min_samples_split = [6] # minimum sample number to split a node
min_samples_leaf = [1] # minimum sample number that can be stored in a leaf node
bootstrap = [True] # method used to sample data points

random_grid = {'n_estimators': n_estimators,

    'max_features': max_features,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf,

    'bootstrap': bootstrap}



def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2,filename):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.savefig(filename, format="pdf", bbox_inches="tight")

# ind = 0
# if(ind > 100):
#     plot_grid_search(rf_random.cv_results_, n_estimators, max_depth, 'N Estimators', 'Max Depth', "./")


def plot_tree():

    ### Draw the random forest and the decision trees
    from sklearn import tree
    import pydotplus
    from IPython.display import Image
    model_rf = joblib.load("/Users/song/Library/CloudStorage/OneDrive-mails.ucas.edu.cn/task/funcat/task/05snp_only/rf.model")
    Estimators = model_rf.estimators_
    for index, model in enumerate(Estimators):
        filename = 'iris_' + str(index) + '.pdf'
        dot_data = tree.export_graphviz(model , out_file=None,
                            #  feature_names=iris.feature_names,
                            #  class_names=iris.target_names,
                            filled=True, rounded=True,
                            special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        # 使用ipython的终端jupyter notebook显示。
        Image(graph.create_png())
        graph.write_pdf(filename)


def get_auc(t,p):
    fpr_c1, tpr_c1, thresholds_c1 = roc_curve(t, p)
    auc_c1 = auc(fpr_c1, tpr_c1)
    return auc_c1, fpr_c1, tpr_c1


def plot_curve():
    true_filename = "/Users/song/Library/CloudStorage/OneDrive-mails.ucas.edu.cn/task/funcat/task/05snp_only/snpeff/cis-eQTL-randomSNP_1.sum.eqtl"
    Xt,Yt = returnXY(true_filename)
    sum_bed = pd.read_table(true_filename,header=None,na_values=['-'])
    # print(data.shape)
    sum_bed.dropna(inplace = True)
    e1 = sum_bed.iloc[:,2]
    c1 = sum_bed.iloc[:,3]
    c2 = sum_bed.iloc[:,4]
    c3 = sum_bed.iloc[:,5]
    c4 = sum_bed.iloc[:,6]
    c_avr = (c1+c2+c3)/3

    import joblib
    model_rf = joblib.load("/Users/song/Library/CloudStorage/OneDrive-mails.ucas.edu.cn/task/funcat/task/05snp_only/rf.model")
    # model_rf = models.load_model("/Users/song/Library/CloudStorage/OneDrive-mails.ucas.edu.cn/task/funcat/task/05snp_only/narrow_nn.model")

    Yp = model_rf.predict(Xt)
    # Yp = reg.predict(Xt)


    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    auc_x1, fpr_x1, tpr_x1 = get_auc(e1,sum_bed.iloc[:,8])
    auc_x13, fpr_x13, tpr_x13 = get_auc(e1,sum_bed.iloc[:,13])
    auc_x32, fpr_x32, tpr_x32 = get_auc(e1,sum_bed.iloc[:,32])
    auc_x34, fpr_x34, tpr_x34 = get_auc(e1,sum_bed.iloc[:,34])
    auc_p, fpr_p, tpr_p = get_auc(e1, Yp)
    # auc_pm, fpr_pm, tpr_pm = get_auc(e1, Ypm)
    fpr_c1, tpr_c1, thresholds_c1 = roc_curve(e1, c1)
    fpr_c2, tpr_c2, thresholds_c2 = roc_curve(e1, c2)
    fpr_c3, tpr_c3, thresholds_c3 = roc_curve(e1, c3)
    fpr_c4, tpr_c4, thresholds_c4 = roc_curve(e1, c4)
    fpr_cavr, tpr_cavr, thresholds_cavr = roc_curve(e1, c_avr)

    auc_c1 = auc(fpr_c1, tpr_c1)
    auc_c2 = auc(fpr_c2, tpr_c2)
    auc_c3 = auc(fpr_c3, tpr_c3)
    auc_c4 = auc(fpr_c4, tpr_c4)
    auc_cavr = auc(fpr_cavr, tpr_cavr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_c1, tpr_c1, label='MAF (area = {:.3f})'.format(auc_c1))
    plt.plot(fpr_c2, tpr_c2, label='Genome (area = {:.3f})'.format(auc_c2))
    plt.plot(fpr_c3, tpr_c3, label='Conservation (area = {:.3f})'.format(auc_c3))
    plt.plot(fpr_c4, tpr_c4, label='Epigenetics (area = {:.3f})'.format(auc_c4))
    # plt.plot(fpr_cavr, tpr_cavr, label='cavr (area = {:.3f})'.format(auc_cavr))
    plt.plot(fpr_p, tpr_p, label='P2-4 (area = {:.3f})'.format(auc_p))
    # plt.plot(fpr_pm, tpr_pm, label='P1-4 (area = {:.3f})'.format(auc_pm))
    # plt.plot(fpr_x1, tpr_x1, label='CDS_HC (area = {:.3f})'.format(auc_x1))
    # plt.plot(fpr_x13, tpr_x13, label='TE (area = {:.3f})'.format(auc_x13))
    # plt.plot(fpr_x32, tpr_x32, label='RNAPII (area = {:.3f})'.format(auc_x32))
    plt.plot(fpr_x34, tpr_x34, label='GRO (area = {:.3f})'.format(auc_x34))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


def show_data():
    auc_c1_arr = []
    auc_c2_arr = []
    auc_c3_arr = []
    auc_c4_arr = []
    auc_cavr_arr = []
    f = open("/Users/song/Desktop/density.txt",'w')
    for i in range(1,101):
        filename = "/Users/song/Library/CloudStorage/OneDrive-mails.ucas.edu.cn/task/funcat/task/04m123/random/" + "cis-eQTL-randomSNP_" +  str(i)  + ".sum.eqtl"
        sum_bed = pd.read_table(filename,header=None,na_values=['-'])
        # print(data.shape)
        sum_bed.dropna(inplace = True)
        e1 = sum_bed.iloc[:,2]
        c1 = sum_bed.iloc[:,3]
        c2 = sum_bed.iloc[:,4]
        c3 = sum_bed.iloc[:,5]
        c4 = sum_bed.iloc[:,6]
        c_avr = (c1+c2+c3)/3
        from sklearn.metrics import roc_curve
        from sklearn.metrics import auc

        fpr_c1, tpr_c1, thresholds_c1 = roc_curve(e1, c1)
        fpr_c2, tpr_c2, thresholds_c2 = roc_curve(e1, c2)
        fpr_c3, tpr_c3, thresholds_c3 = roc_curve(e1, c3)
        fpr_c4, tpr_c4, thresholds_c4 = roc_curve(e1, c4)
        fpr_cavr, tpr_cavr, thresholds_cavr = roc_curve(e1, c_avr)

        auc_c1 = auc(fpr_c1, tpr_c1)
        auc_c2 = auc(fpr_c2, tpr_c2)
        auc_c3 = auc(fpr_c3, tpr_c3)
        auc_c4 = auc(fpr_c4, tpr_c4)
        auc_cavr = auc(fpr_cavr, tpr_cavr)

        auc_c1_arr.append(auc_c1)
        auc_c2_arr.append(auc_c2)
        auc_c3_arr.append(auc_c3)
        auc_c4_arr.append(auc_c4)
        auc_cavr_arr.append(auc_cavr)

        for i in range(0,len(auc_c1_arr)):
            f.write("sl")
            f.write("\t")
            f.write(auc_c1_arr[i])
            f.write("\t")
            f.write(auc_c2_arr[i])
            f.write("\t")
            f.write(auc_c3_arr[i])
            f.write("\t")
            f.write(auc_c4_arr[i])
            f.write("\n")
        f.close()