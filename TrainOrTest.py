import tensorflow as tf
import numpy as np
import tflearn
import argparse
import pprint as pp
import sys

from replay_buffer import ReplayBuffer
from AandC import *
from noise import *
from gym_torcs import TorcsEnv

mu = np.array([0.0, 0.5, 0.01])
# theta = np.array([0.15, 0.15, 0.15])
theta = np.array([0.0, 0.0, 0.0])
# sigma = np.array([0.2, 0.2, 0.2])
sigma = np.array([0.1, 0.1, 0.1])

# irestart = 0 重新开始; = 1 从给定步数开始
irestart = 0
restart_step = 55

if (irestart == 0):
    restart_step = 0

# ---------------------------------------------------------------------

include_track = True


def trainDDPG(sess, args, actor, critic):
    saver = tf.train.Saver()
    # 创建一个虚拟环境
    env = TorcsEnv(throttle=True)
    # 如果重新开始，则创建新的Session
    # 如果继续训练则加载模型
    if (irestart == 0):
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, "ckpt/model")
    # 初始化目标网络（Target Network）权重
    actor.update_target_network()
    critic.update_target_network()
    # 初始化回放缓存（replay memory）
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
    episode_count = args['episode_count']
    max_steps = args['max_steps']

    epsilon = 1.0

    for i in range(restart_step, episode_count):
        ob = env.reset()
        # 状态 states
        if include_track is False:
            s = np.hstack(
                (ob.angle, ob.trackPos, ob.speedX, ob.speedY))
            # (ob.angle, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.rpm))
            print(ob.angle, ' ', ob.trackPos, ' ', ob.speedX, ' ', ob.speedY)
        else:
            s = np.hstack(
                (ob.angle, ob.trackPos, ob.speedX, ob.speedY, ob.rpm, ob.track))

        ep_reward = 0

        ep_ave_max_q = 0

        msteps = max_steps
        if (i < 100):
            msteps = 60
        elif (i >= 100 and i < 200):
            msteps = 100 + (i - 100) * 9
        else:
            msteps = 1000 + (i - 200) * 5

        msteps = min(msteps, max_steps)

        for j in range(msteps):
            # 动作噪声（action noise）
            if (args['noise'] == 'OU'):  # Ornstein-Uhlenbeck噪声
                a = actor.predict(np.reshape(s, (1, actor.s_dim)), noise=False)
                a[0, :] += OU(x=a[0, :], mu=mu, sigma=sigma, theta=theta) * max(epsilon, 0.0)
            elif (args['noise'] == 'nonoise'):
                a = actor.predict(np.reshape(s, (1, actor.s_dim)), noise=False)

            else:
                print("noise error ", args['noise'])
                sys.exit()

            # 前10轮油门加满， 初始化Agent的网络
            if (i < 10):
                a[0][0] = 0.0
                a[0][1] = 1.0
                a[0][2] = 0.0

            # 输出训练信息
            print("episode: ", i, "step: ", j, "action: ", a)
            # 执行动作，获得来自环境的反馈
            ob, r, terminal, info = env.step(a[0])
            # 执行动作后的环境状态s2
            if include_track is False:
                s2 = np.hstack(
                    (ob.angle, ob.trackPos, ob.speedX, ob.speedY))
                # (ob.angle, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.rpm))
                print(ob.angle, ' ', ob.trackPos, ' ', ob.speedX, ' ', ob.speedY)
            else:
                s2 = np.hstack(
                    (ob.angle, ob.trackPos, ob.speedX, ob.speedY, ob.rpm, ob.track))

            # 将状态 s、s2 加入到回放缓存中
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))
            # 将经验加入到回放缓存中，直到经验样本数量达到minibatch指定的大小
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(args['minibatch_size']))
                # 计算目标网络的q-value
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # 给定目标网络更新critic
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                                                                             i, (ep_ave_max_q / float(j))))

                with open("analysis_file.txt", "a") as myfile:
                    myfile.write(
                        str(i) + " " + str(j) + " " + str(ep_reward) + " " + str(ep_ave_max_q / float(j)) + "\n")
                break

        if (np.mod(i, 10) == 0 and i > 1):
            saver.save(sess, "ckpt/model")
            print("saved model after ", i, " episodes ")


# 测试模型的性能
def testDDPG(sess, args, actor, critic):
    env = TorcsEnv(throttle=True)

    episode_count = args['episode_count']
    max_steps = args['max_steps']

    for i in range(restart_step, episode_count):
        ob = env.reset()

        # s = np.hstack(
        #    (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
        s = np.hstack(
            (ob.angle, ob.trackPos, ob.speedX, ob.speedY, ob.rpm, ob.track))

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(max_steps):

            a = actor.predict(np.reshape(s, (1, actor.s_dim)))
            a[0, :] += OU(x=a[0, :], mu=mu, sigma=sigma, theta=theta)

            # 输出训练信息
            print("episode: ", i, "step: ", j, "action: ", a)

            ob, r, terminal, info = env.step(a[0])
            # s2 = np.hstack(
            #    (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            s2 = np.hstack(
                (ob.angle, ob.trackPos, ob.speedX, ob.speedY, ob.rpm, ob.track))

            s = s2
            ep_reward += r

            if terminal:
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                                                                             i, 0))
                with open("performance.txt", "a") as myfile:
                    myfile.write(
                        str(i) + " " + str(j) + " " + str(ep_reward) + " " + str(0) + "\n")
                break
        with open("performance.txt", "a") as myfile:
            myfile.write(
                str(i) + " " + str(j) + " " + str(ep_reward) + " " + str(0) + "\n")
