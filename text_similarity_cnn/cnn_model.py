# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 128  # 词向量维度
    seq_length = 20  # 序列长度
    num_classes = 2  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 100  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 四个待输入的数据
        self.input_xl = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_xl')
        self.input_xr = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_xr')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 1. word_embedding 层
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputsl = tf.nn.embedding_lookup(embedding, self.input_xl)
            embedding_inputsr = tf.nn.embedding_lookup(embedding, self.input_xr)


	#2. cnn层   其中convq1和convq2的size是[batch_size, seq_length, num_filters] 用于求相似度
        #            fq1 ,fq2的size是[batch_size, hidden_dim] 是句子的最终表达，用于最后的拼接
        with tf.name_scope("convq1"):
            convq1 = tf.layers.conv1d(embedding_inputsl, self.config.num_filters,
                    self.config.kernel_size,padding='same', name='convq1')
        with tf.name_scope("fq1"):
            mp_q1 = tf.reduce_max(convq1, reduction_indices=[1], name='mp_q1')
            fq1 = tf.contrib.layers.dropout(mp_q1, self.keep_prob)

        #fq2
        with tf.name_scope("convq2"):
            convq2 = tf.layers.conv1d(embedding_inputsr, self.config.num_filters,
                                         self.config.kernel_size, padding='same', name='convq2')
        with tf.name_scope("fq2"):
            mp_q2 = tf.reduce_max(convq2, reduction_indices=[1], name='mp_q2')
            fq2 = tf.contrib.layers.dropout(mp_q2, self.keep_prob)


	
	#3. 求convq1 和convq2的相似度矩阵,将这两个句子中的每对词拼接后，后经过一个relu的全连接网络，之后降维成一个二维矩阵
           #矩阵的size是[batch_size, seq_length, seq_length]
        with tf.name_scope("similarity_matrix"):
	    #将两个句子的每个词两两拼接
            simq1 = tf.expand_dims(convq1,2)
            simq1 = tf.tile(simq1, [1, 1, self.config.seq_length, 1])
            simq2 = tf.expand_dims(convq2,1) 
            simq2 = tf.tile(simq2,[1,self.config.seq_length,1,1])
            simq1_q2 = tf.concat([simq1, simq2],3)
            print("the shape of simq1_q2 is ",simq1_q2.shape)

	    #对每个拼接经过一个relu(wx+b)
            weights_size = 2 * self.config.num_filters
            weights = tf.get_variable("weights", shape=(1,1,weights_size), dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            weights = tf.contrib.layers.dropout(weights, self.keep_prob)
            weights = tf.tile(weights, [self.config.seq_length,self.config.seq_length, 1])
            unsummed_dots = tf.multiply(simq1_q2, weights)
            b = tf.Variable(tf.constant(0.1, shape=[weights_size]), name="b")
            unsummed_dots = tf.nn.relu(tf.nn.bias_add(unsummed_dots, b), name="relu")
            similarity = tf.reduce_sum(unsummed_dots, axis=3)
            



	#4. 求 q2对q1的attention: q2toq1 attention signifies which query words are most relevant to each DB-query word.
        with tf.name_scope("attention_qustion2_to_question1"):
            attentions = tf.nn.softmax(similarity)
            attentions_tiled = tf.expand_dims(attentions, 3)
            attentions_tiled = tf.tile(attentions_tiled, [1, 1, 1,self.config.num_filters])

            question2_tiled = tf.expand_dims(convq2, 1)
            question2_tiled = tf.tile(question2_tiled, [1, self.config.seq_length, 1, 1])
 
            A_q2_q1 = tf.multiply(attentions_tiled, question2_tiled)
            A_q2_q1 = tf.reduce_sum(A_q2_q1, axis=2)
			

	#5. 求 q2对q1的attention: q2toq1 attention signifies which query words are most relevant to each DB-query word.
        with tf.name_scope("attention_question1_to_question2"):
            attentions = tf.nn.softmax(similarity)
            attention_tiled = tf.expand_dims(attentions, 3) 
            attention_tiled = tf.tile(attention_tiled, [1, 1, 1, self.config.num_filters])
					
            question1_tiled = tf.expand_dims(convq1, 2)
            question1_tiled = tf.tile(question1_tiled, [1, 1, self.config.seq_length, 1])

            A_q1_q2 = tf.multiply(attentions_tiled, question1_tiled)
            A_q1_q2 = tf.reduce_sum(A_q1_q2, axis=1)


         #6. 求最终的attention
        with tf.name_scope("final_attention"):
            attention_concat = tf.concat([A_q1_q2, A_q2_q1], axis=2)
            weights_size = 2 * self.config.num_filters
            weights = tf.get_variable("A_W", shape=(1,weights_size), dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            weights = tf.contrib.layers.dropout(weights, self.keep_prob)
            weights = tf.tile(weights, [self.config.seq_length, 1])
            final_attention = tf.multiply(weights, attention_concat)
            b = tf.Variable(tf.constant(0.1, shape=[weights_size]), name="b")
            final_attention = tf.nn.relu(tf.nn.bias_add(final_attention, b), name="relu1")
        
        #7. join层
        with tf.name_scope("join"):
            join_layer = tf.concat([convq1,final_attention, convq2], axis=2)
         
        #8 . 输出层
        with tf.name_scope("output"):
            weights_size = 4*self.config.num_filters
            weights_final = tf.get_variable("weights_final", shape=(1,weights_size), dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            weights_final = tf.contrib.layers.dropout(weights_final, self.keep_prob)
            weights_final = tf.tile(weights_final, [self.config.seq_length, 1])
            sum_output = tf.multiply(weights_final,join_layer)
            bf = tf.Variable(tf.constant(0.1, shape=[weights_size]), name="bf")
            sum_output = tf.nn.bias_add(sum_output, bf)
            predict_beg = tf.reduce_sum(sum_output, axis=2)

            #分类器
            self.logits = tf.layers.dense(predict_beg, self.config.num_classes, name='logits')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# coding: utf-8

