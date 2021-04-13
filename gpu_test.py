# %%
import keras
import tensorflow as tf

# %%
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1}) 
# %%
sess = tf.Session(config=config) 
# %%
keras.backend.set_session(sess)