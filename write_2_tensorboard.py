import os
import shutil
import numpy as np
import tensorflow as tf

log_dir = "logs"

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
writer = tf.summary.create_file_writer(log_dir)

data = np.loadtxt('analysis_file.txt')
for i in range(1000):
    with writer.as_default():
        tf.summary.scalar("MT-DDPG-reward", data[:, 2][i], step=i)
    writer.flush()
