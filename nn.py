import tensorflow as tf
import numpy as np
import pandas as pd


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) - 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def add_network(network_structure, input_data, middle_function, output_function):
    output_value = []
    output_value.append(input_data)
    for layer in range(len(network_structure)-1):
        if layer == len(network_structure)-2:
            output = add_layer(output_value[layer], network_structure[layer], network_structure[layer+1], output_function)
        else:
            output = add_layer(output_value[layer], network_structure[layer], network_structure[layer+1], middle_function)
        output_value.append(output)
    
    return output_value

####################################讀取資料#######################################
    
df = pd.read_csv('data_nn.csv')
TEST_RATIO = 0.25
#TRA_INDEX = int((1-TEST_RATIO) * df.shape[0])#訓練資料的資料量
TRA_INDEX = 5

#df.sort_values('Time', inplace = True)


total_x = df.iloc[0:df.shape[0], 0:-1].values
_total_y_ = df.iloc[0:df.shape[0], -1].values
total_y = _total_y_[:, np.newaxis]

train_x = df.iloc[0:TRA_INDEX, 0:-1].values
_train_y_ = df.iloc[0:TRA_INDEX, -1].values
train_y = _train_y_[:, np.newaxis]

#print(train_x,'\n')
#print(train_y,'\n')

test_x = df.iloc[TRA_INDEX:, 0:-1].values
_test_y_ = df.iloc[TRA_INDEX:, -1].values
test_y = _test_y_[:, np.newaxis]

'''
重要參數如下:
    訓練用:
        train_x = 檔案中的輸入資料(第2排到~倒數第3排)
        train_y = 檔案中的類別資料(倒數第1排)
    測試用:
        test_x = 檔案中的輸入資料(第2排到~倒數第3排)
        test_y = 檔案中的類別資料(倒數第1排)

'''
####################################讀取完資料######################################


#####################################網路架設#######################################

input_number = train_x.shape[1] #資料input的數量
output_number = train_y.shape[1]#資料output的數量
nn_structure = [input_number, 6, 4, output_number]

input_placeholder = tf.placeholder("float", [None, input_number])
output_placeholder = tf.placeholder("float", [None, output_number])

nn_return = add_network(nn_structure, input_placeholder, tf.nn.tanh, tf.nn.sigmoid)

nn_layer_number = len(nn_structure)
nn_output_layer = nn_return[nn_layer_number-1]
#nn_softmax_output = tf.nn.softmax(nn_output_layer)


'''
重要參數如下:
    placeholder:
        input_placeholder = 輸入nn的placeholder
        output_placeholder = nn輸出層的placeholder
    
    nn:
        nn_return = 全網路輸出值
        nn_output_layer = 輸出層網路輸出值

'''
####################################網路架設結束######################################
   
######################################訓練開始########################################
learning_rate = 0.01
training_epochs = 10000
batch_size = 3
display_step = 1000

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output_placeholder,logits=nn_output_layer))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

#init = tf.global_variables_initializer()
#sess.run(init)


with tf.Session() as sess:
    
    #sess.run(init)
    
    saver_read_data = tf.train.Saver()
    saver_read_data.restore(sess, "my_net_3/save_net.ckpt")
    
    print(sess.run(cross_entropy,{input_placeholder: test_x, output_placeholder: test_y}))
    
    #print(sess.run(nn_return,{input_placeholder: train_x, output_placeholder: train_y}))
    #print(sess.run(nn_output_layer,{input_placeholder: train_x, output_placeholder: train_y}))
    #print(train_y)
    for i in range(training_epochs):
        batch_idx = np.random.choice(train_x.shape[0], batch_size)
        batch_input_data = train_x[batch_idx]
        batch_output_data = train_y[batch_idx]
        
        sess.run(optimizer, {input_placeholder: batch_input_data, output_placeholder: batch_output_data})
        #print(sess.run(nn_output_layer,{input_placeholder: batch_input_data, output_placeholder: batch_output_data}))
    
    #print(sess.run(nn_output_layer,{input_placeholder: train_x, output_placeholder: train_y}))
    print(sess.run(cross_entropy,{input_placeholder: test_x, output_placeholder: test_y}))

  
######################################訓練結束########################################
    
    saver_write_data = tf.train.Saver()
    save_path = saver_write_data.save(sess, "my_net_3/save_net.ckpt")
    print("Save to path: ", save_path)      
       
tf.reset_default_graph()









