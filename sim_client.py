import socket
import sys
import getopt
import os
import time

import zmq

PI = 3.14159265359

data_size = 2 ** 17


def clip(v, lo, hi):
    if v < lo:
        return lo
    elif v > hi:
        return hi
    else:
        return v


def bargraph(x, mn, mx, w, c='X'):
    '''Draws a simple asciiart bar graph. Very handy for
    visualizing what's going on with the data.
    x= Value from sensor, mn= minimum plottable value,
    mx= maximum plottable value, w= width of plot in chars,
    c= the character to plot with.'''
    if not w: return ''  # No width!
    if x < mn: x = mn  # Clip to bounds.
    if x > mx: x = mx  # Clip to bounds.
    tx = mx - mn  # Total real units possible to show on graph.
    if tx <= 0: return 'backwards'  # Stupid bounds.
    upw = tx / float(w)  # X Units per output char width.
    if upw <= 0: return 'what?'  # Don't let this happen.
    negpu, pospu, negnonpu, posnonpu = 0, 0, 0, 0
    if mn < 0:  # Then there is a negative part to graph.
        if x < 0:  # And the plot is on the negative side.
            negpu = -x + min(0, mx)
            negnonpu = -mn + x
        else:  # Plot is on pos. Neg side is empty.
            negnonpu = -mn + min(0, mx)  # But still show some empty neg.
    if mx > 0:  # There is a positive part to the graph
        if x > 0:  # And the plot is on the positive side.
            pospu = x - max(0, mn)
            posnonpu = mx - x
        else:  # Plot is on neg. Pos side is empty.
            posnonpu = mx - max(0, mn)  # But still show some empty pos.
    nnc = int(negnonpu / upw) * '-'
    npc = int(negpu / upw) * c
    ppc = int(pospu / upw) * c
    pnc = int(posnonpu / upw) * '_'
    return '[%s]' % (nnc + npc + ppc + pnc)


# ZeroMQ连接相关
TIMEOUT = 10000


class Client():
    def __init__(self, H=None, p=None, e=None, d=None):
        # If you don't like the option defaults,  change them here.
        self.host = 'localhost'
        self.server_port = 12345
        self.client_port = 12346
        self.maxEpisodes = 1  # "Maximum number of learning episodes to perform"
        self.debug = False
        self.maxSteps = 100000  # 50steps/second
        if H: self.host = H
        if p: self.port = p
        if e: self.maxEpisodes = e
        if d: self.debug = d
        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    # 创建连接
    # 使用ZeroMQ 创建与虚拟环境的链接
    # 返回ok
    def setup_connection(self):
        # == Set Up ZeroMQ Socket ==
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:" + str(self.server_port))
        message = "gym:init"
        self.socket.send_string(message)
        time.sleep(0.5)
        self.socket.send_string(message)
    def close_connection(self):
        self.socket.close()

    def reset_env(self):
        # == Set Up ZeroMQ Socket ==
        message = "gym:init"
        self.socket.send_string(message)
        time.sleep(0.5)
        self.socket.send_string(message)

    def get_servers_input(self):
        '''Server's input is stored in a ServerState object'''
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:" + str(self.client_port))

        while True:
            socket.send_string("request")
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            evt = dict(poller.poll(TIMEOUT))
            if evt:
                if evt.get(socket) == zmq.POLLIN:
                    response = socket.recv(zmq.NOBLOCK)
                    # print(response)
                    self.S.parse_server_str(response)
                    # print(self.S.__repr__())
                    break
            time.sleep(0.5)
            socket.close()

    def respond_to_server(self):
        # steer, accel, brake
        message = "action:"+str(self.R.d['steer'])+','+str(self.R.d['accel'])+','+str(self.R.d['brake'])
        self.socket.send_string(message)
        time.sleep(0.2)
        self.socket.send_string(message)
        '''
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1], str(emsg[0])))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())
        # Or use this for plain output:
        # if self.debug: print self.R
        '''

    def shutdown(self):
        # TODO：尚未完成内容
        if not self.so: return
        print(("Race terminated or %d steps elapsed. Shutting down %d."
               % (self.maxSteps, self.port)))
        self.so.close()
        self.so = None
        # sys.exit() # No need for this really.


import json


class ServerState():
    '''What the server is reporting right now.'''

    def __init__(self):
        self.servstr = str()
        self.d = dict()

    def parse_server_str(self, server_string):
        '''Parse the server string.'''
        data = json.loads(server_string)
        self.d = data

    def __repr__(self):
        # Comment the next line for raw output:
        return self.fancyout()
        # -------------------------------------
        out = str()
        for k in sorted(self.d):
            strout = str(self.d[k])
            if type(self.d[k]) is list:
                strlist = [str(i) for i in self.d[k]]
                strout = ', '.join(strlist)
            out += "%s: %s\n" % (k, strout)
        return out

    def fancyout(self):
        '''Specialty output for useful ServerState monitoring.'''
        out = str()
        sensors = [  # Select the ones you want in the order you want them.
            'speedZ',
            'speedY',
            'speedX',
            'rpm',
            'track',
            'trackPos',
            'angle',
            'speed',
            'damage'
        ]

        # for k in sorted(self.d): # Use this to get all sensors.
        for k in sensors:
            if type(self.d.get(k)) is list:  # Handle list type data.
                if k == 'track':  # Nice display for track sensors.
                    strout = str()
                    raw_tsens = ['%.1f' % x for x in self.d['track']]
                    strout += ' '.join(raw_tsens[:9]) + '_' + raw_tsens[9] + '_' + ' '.join(raw_tsens[10:])
                else:
                    strlist = [str(i) for i in self.d[k]]
                    strout = ', '.join(strlist)
            else:  # Not a list type of value.
                if k == 'damage':
                    strout = '%6.0f %s' % (self.d[k], bargraph(self.d[k], 0, 10000, 50, '~'))
                elif k == 'speedX':
                    cx = 'X'
                    if self.d[k] < 0: cx = 'R'
                    strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k], -30, 300, 50, cx))
                elif k == 'speedY':  # This gets reversed for display to make sense.
                    strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k] * -1, -25, 25, 50, 'Y'))
                elif k == 'speedZ':
                    strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k], -13, 13, 50, 'Z'))
                elif k == 'z':
                    strout = '%6.3f %s' % (self.d[k], bargraph(self.d[k], .3, .5, 50, 'z'))
                elif k == 'trackPos':  # This gets reversed for display to make sense.
                    cx = '<'
                    if self.d[k] < 0: cx = '>'
                    strout = '%6.3f %s' % (self.d[k], bargraph(self.d[k] * -1, -1, 1, 50, cx))
                elif k == 'rpm':
                    strout = bargraph(self.d[k], 0, 10000, 50)
                elif k == 'angle':
                    asyms = [
                        "  !  ", ".|'  ", "./'  ", "_.-  ", ".--  ", "..-  ",
                        "---  ", ".__  ", "-._  ", "'-.  ", "'\.  ", "'|.  ",
                        "  |  ", "  .|'", "  ./'", "  .-'", "  _.-", "  __.",
                        "  ---", "  --.", "  -._", "  -..", "  '\.", "  '|."]
                    rad = self.d[k]
                    deg = int(rad * 180 / PI)
                    symno = int(.5 + (rad + PI) / (PI / 12))
                    symno = symno % (len(asyms) - 1)
                    strout = '%5.2f %3d (%s)' % (rad, deg, asyms[symno])
                else:
                    strout = str(self.d[k])
            out += "%s: %s\n" % (k, strout)
        return out


class DriverAction():
    '''What the driver is intending to do (i.e. send to the server).
    Composes something like this for the server:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)'''

    def __init__(self):
        self.actionstr = str()
        # "d" is for data dictionary.
        self.d = {'accel': 0.2,
                  'brake': 0,
                  'handbrake': 0,
                  'steer': 0,
                  'meta': 0
                  }

    def clip_to_limits(self):
        """There pretty much is never a reason to send the server
        something like (steer 9483.323). This comes up all the time
        and it's probably just more sensible to always clip it than to
        worry about when to. The "clip" command is still a snakeoil
        utility function, but it should be used only for non standard
        things or non obvious limits (limit the steering to the left,
        for example). For normal limits, simply don't worry about it."""
        self.d['steer'] = clip(self.d['steer'], -1, 1)
        self.d['brake'] = clip(self.d['brake'], 0, 1)
        self.d['accel'] = clip(self.d['accel'], 0, 1)
        self.d['clutch'] = clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0
        if self.d['meta'] not in [0, 1]:
            self.d['meta'] = 0
        if type(self.d['focus']) is not list or min(self.d['focus']) < -180 or max(self.d['focus']) > 180:
            self.d['focus'] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not type(v) is list:
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out
        return out + '\n'

    def fancyout(self):
        '''Specialty output for useful monitoring of bot's effectors.'''
        out = str()
        od = self.d.copy()
        for k in sorted(od):
            if k == 'clutch' or k == 'brake' or k == 'accel':
                strout = ''
                strout = '%6.3f %s' % (od[k], bargraph(od[k], 0, 1, 50, k[0].upper()))
            elif k == 'steer':  # Reverse the graph to make sense.
                strout = '%6.3f %s' % (od[k], bargraph(od[k] * -1, -1, 1, 50, 'S'))
            else:
                strout = str(od[k])
            out += "%s: %s\n" % (k, strout)
        return out


# == Misc Utility Functions
def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]


def drive_example(c):
    '''This is only an example. It will get around the track but the
    correct thing to do is write your own `drive()` function.'''
    S, R = c.S.d, c.R.d
    target_speed = 100

    # Steer To Corner
    R['steer'] = S['angle'] * 10 / PI
    # Steer To Center
    R['steer'] -= S['trackPos'] * .10

    # Throttle Control
    if S['speedX'] < target_speed - (R['steer'] * 50):
        R['accel'] += .01
    else:
        R['accel'] -= .01
    if S['speedX'] < 10:
        R['accel'] += 1 / (S['speedX'] + .1)

    # Traction Control System
    if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) -
            (S['wheelSpinVel'][0] + S['wheelSpinVel'][1]) > 5):
        R['accel'] -= .2

    # Automatic Transmission
    R['gear'] = 1
    if S['speedX'] > 50:
        R['gear'] = 2
    if S['speedX'] > 80:
        R['gear'] = 3
    if S['speedX'] > 110:
        R['gear'] = 4
    if S['speedX'] > 140:
        R['gear'] = 5
    if S['speedX'] > 170:
        R['gear'] = 6
    return


# ================ MAIN ================
if __name__ == "__main__":
    C = Client(p=3101)
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive_example(C)
        C.respond_to_server()
    C.shutdown()
