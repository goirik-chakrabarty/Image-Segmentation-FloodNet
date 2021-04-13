# %%
import keras
import tensorflow as tf


# %%
print(keras.__version__)
print(tf.__version__)

# %% 
print("Num CPU Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
# %%
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1})
# %%
sess = tf.Session(config=config) 
# %%
keras.backend.set_session(sess)
# %%
config=tf.ConfigProto(log_device_placement=True)
# %%
