from base import *
from dense import *
import sys

num = 3
# baseline seq2seq model
class seq_attn(base_enc_dec):
    @init_final
    def __init__(self, labels, alabel, elabel, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode, beam_size=5):
        #self.iftest = 1
        #self.iftest = 0
        with tf.variable_scope('decode'):
            self.init_W = tf.get_variable('Init_W', initializer=tf.random_normal([h_size+12, h_size]))
            self.init_b = tf.get_variable('Init_b', initializer=tf.zeros([h_size]))
            self.attn_W = Dense("Attn_W",1,2*h_size,nonlinearity=tf.tanh,name='Attn_W')
            #self.attn_V = tf.get_variable('Attn_V', initializer=tf.random_normal([h_size]))
            #self.attn_W2 = Dense("Attn_W2", 1, h_size,nonlinearity=tf.tanh, name='Attn_W2')
        base_enc_dec.__init__(self, labels, alabel, elabel, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode,
                              beam_size)

    """
    prev_h[0]: word-level last state
    prev_h[1]: decoder last state
    baseline seq2seq model
    """

    def run_encode(self, prev_h, input_labels):
        mask = self.gen_mask(input_labels, EOU)
        embedding = self.embed_labels(input_labels)
        h = self.word_level_rnn(prev_h[0], embedding)
  #      prev_h[1] = prev_h[1] * mask + tf.tanh(tf.matmul(h, self.init_W) + self.init_b) * (
  #      1 - mask)  # learn initial state from context
  #      d = self.decode_level_rnn(prev_h[1], embedding)
        return [h, prev_h[1] + 1 - mask]

    def run_decode(self, prev_h, input_labels):
        h, labels, alabel, elabel, num_seq = input_labels
        mask = self.gen_mask(labels, EOU)
        embedding = self.embed_labels(labels)
        alabel=tf.cast(alabel,tf.float32)
        elabel=tf.cast(elabel,tf.float32)
        alabel=tf.reshape(alabel,[self.batch_size,1])
        elabel=tf.reshape(elabel,[self.batch_size,1])
        #prev_h[2]=tf.cast(prev_h[2],tf.float32)
        #prev_h[3]=tf.cast(prev_h[3],tf.float32)
        alabel=alabel*(1-mask)+prev_h[1]*mask
        elabel=elabel*(1-mask)+prev_h[2]*mask
        alabel=tf.reshape(alabel,[self.batch_size])
        elabel=tf.reshape(elabel,[self.batch_size])
        alabel=tf.cast(alabel,tf.int64)
        elabel=tf.cast(elabel,tf.int64)
        
        astate = tf.one_hot(alabel,5)#batch_size*5
        estate = tf.one_hot(elabel,7)#batch_size*5
        alabel=tf.reshape(alabel,[self.batch_size,1])
        elabel=tf.reshape(elabel,[self.batch_size,1])
        h = tf.concat(1,[h,astate,estate])
        prev_h = prev_h[0] * mask + tf.tanh(tf.matmul(h, self.init_W) + self.init_b) * (
        1 - mask)
        # get needed h
        attn= self.attention(self.h, prev_h, num_seq)
        return [self.decode_level_rnn(prev_h, tf.concat(1,[embedding,attn, astate,estate])), tf.cast(alabel,tf.float32), tf.cast(elabel,tf.float32)]

        
    # attention_states: attn_length*batch_size*hidden_size
    # d: hidden state of last step for decoding
    def attention(self, attention_states, d, num_seq):
        attn_length = tf.shape(attention_states)[0]
        #hidden = tf.reshape(attention_states,[attn_length, self.batch_size, 1, self.h_size])
        # v^T * tanh(w1*h + w2*d)
        #hidden_features = tf.nn.conv2d(hidden, self.attn_W1, [1, 1, 1, 1], "SAME") #attn_length*batch_size*1*hidden_size
        #hidden_features = self.attn_W1(attention_states, True)#attn_lengh*batch_size*1
        y = tf.tile(tf.expand_dims(d,0),[attn_length,1,1])
        features=tf.concat(2,[attention_states,y])
        s = tf.reshape((self.attn_W(features, True)),[attn_length,self.batch_size])
        #s = tf.reduce_sum(self.attn_V * tf.tanh(hidden_features + y), [2, 3])# attn_length*batch_size
        #s=tf.squeeze(hidden_features)+y
        mask = tf.reshape((tf.cast(tf.less(self.num_seq, num_seq), tf.float32)),[attn_length,self.batch_size])
        s1 = s*mask-(1-mask)*sys.maxint
        #a = tf.exp(s)#*mask
        #a = tf.div(a, tf.reshape(tf.reduce_sum(a,[0]),[1,self.batch_size]))
        s1 = tf.transpose(s1,[1,0])
        a = tf.nn.softmax(s1)
        a=tf.transpose(a,[1,0])
        # d = sigma(ai*hi)
        if self.mode == 2:
            print "aaa"
            s=tf.transpose(s,[1,0])
            a = tf.nn.softmax(s) 
            a=tf.transpose(a,[1,0])
        #new_d = tf.reduce_sum(tf.reshape(a, [attn_length, self.batch_size, 1, 1]) * hidden, [0, 2])
        #new_d = tf.reduce_sum(hidden, [0, 2])
        new_d=tf.reshape(a,[attn_length,self.batch_size,1])*attention_states
        new_d = tf.reduce_sum(new_d,[0])
        return new_d # batch_size*h_size

    def scan_step(self):
        num_seq = tf.zeros([self.batch_size, 1])
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_decode = tf.zeros([self.batch_size, self.h_size])
        self.h, self.post_num_seq = tf.scan(self.run_encode, self.labels, initializer=[init_encode, num_seq])
        self.num_seq = tf.concat(0,[tf.zeros([1,self.batch_size, 1]),self.post_num_seq[:-1]])
        h_d,_,_ = tf.scan(self.run_decode, [self.h, self.labels, self.alabel, self.elabel,self.post_num_seq], initializer=[init_decode,tf.zeros([self.batch_size,1]),tf.zeros([self.batch_size,1])]) 
        return [self.h, h_d]


    def decode_bs(self, h_d):
        last_d = h_d[1][-1]
        astate = tf.tile(tf.one_hot(self.alabel[-1],5), [self.beam_size, 1])
        estate = tf.tile(tf.one_hot(self.elabel[-1],7),[self.beam_size, 1])
        k = 0
        prev = tf.reshape(last_d, [1, self.h_size])
        prev_d = tf.tile(prev, [self.beam_size, 1])
        self.h = tf.tile(self.h, [1, self.beam_size, 1])
        m = tf.tile(tf.constant([[100.0]]), [self.beam_size, 1]) 
        while k < 30:
            if k == 0:
                prev_d = prev    
            inp = self.beam_search(prev_d, k)
            prev_d = tf.reshape(tf.gather(prev_d, self.beam_path[-1]), [self.beam_size, self.h_size])
            k += 1
            with tf.variable_scope('decode') as dec:
                dec.reuse_variables()
               # attn_length = tf.shape(self.h)[0]
               # hidden = tf.reshape(self.h,[attn_length, self.beam_size, 1, self.h_size])
                self.batch_size = self.beam_size
                attn = self.attention(self.h, prev_d, m)
                self.batch_size = 1
                _, d_new = self.decodernet(tf.concat(1,[inp,attn,astate,estate]), prev_d)
                prev_d = d_new
        decoded =  tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        #decoded =  tf.reshape(self.beam_symbols, [self.beam_size, -1])
        return decoded 
 
 
