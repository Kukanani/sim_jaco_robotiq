#!/usr/bin/env python
#
# Copyright (c) 2018 Adam Allevato, The University of Texas at Austin
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the SIM Lab nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Example of PyBullet Kinova Jaco sim and environment. This script moves the
# EEF in a circle and prints out the joint torques to a file, and repeats this
# 10 times with different object mass held in the EEF.
#
# Requirements
#   Python 3.5+ (I think that should be the min version)
#   Dependencies:
#     pybullet
#     matplotlib
#     numpy
# Usage:
#   ./example.py - Default usage (headless)
#   ./example.py debug - Run with PyBullet GUI window visible.

import os
import sys
import errno
import pybullet as p
import time
import math
from math import pi
import matplotlib.pyplot as plt
import numpy as np

# Set debug = True to enable GUI rendering in PyBullet
DEBUG_DEFAULT = False

class KinovaSim(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def __init__(self, debug=DEBUG_DEFAULT):
        self.debug = debug

    def run(self, theta):
        # load arguments
        block_mass, = theta

        # connect and configure sim
        if self.debug:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setRealTimeSimulation(False)
        p.setGravity(0, 0, -10)

        # load arm
        model_path = "jaco_robotiq_object.urdf"
        arm_id = p.loadURDF(model_path, [0, 0, 0], useFixedBase=True)

        # variables specific to this arm
        eef_index = 8
        num_joints = 18
        rp = [-pi / 2, pi * 5 / 4, 0., pi / 2, 0., pi * 5 / 4, -pi / 2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ll = [-pi] * num_joints
        ul = [pi] * num_joints
        jd = [0.000001] * num_joints
        jr = np.array(ll) - np.array(ul)
        ik_solver = p.IK_DLS

        # configure arm
        for i in range(num_joints):
            p.resetJointState(arm_id, i, rp[i])
            p.enableJointForceTorqueSensor(arm_id, i)

        # extract joint names
        joint_id = {}
        for i in range(p.getNumJoints(arm_id)):
            jointInfo = p.getJointInfo(arm_id, i)
            joint_id[jointInfo[1].decode('UTF-8')] = jointInfo[0]

        # adjust block mass
        weight_joint_id = joint_id["weight_joint"]
        p.changeDynamics(arm_id, weight_joint_id, mass=block_mass)

        # set up variables needed during sim
        len_seconds = 5
        # number of steps to wait before recording torques
        burn_in = 150
        start_time = time.time()
        controlled_joints = list(range(eef_index-1))
        num_controlled_joints = eef_index-1
        orn = p.getQuaternionFromEuler([math.pi / 2, math.pi / 2, math.pi / 2])
        targetVelocities = [0] * num_controlled_joints
        forces = [500] * num_controlled_joints
        positionGains = [0.03] * num_controlled_joints
        velocityGains = [1] * num_controlled_joints

        # time counter
        t = 0.
        # store torques as we go
        torques = []

        for steps in range(240 * len_seconds + burn_in):
            p.stepSimulation()
            t += 0.005
            if self.debug:
                time.sleep(1 / 240)

            # move the arm
            pos = [0.2 * math.cos(t), -0.3, 0.3 + 0.2 * math.sin(t)]
            jointPoses = p.calculateInverseKinematics(arm_id, eef_index, pos, orn,
                                                      lowerLimits=ll,
                                                      upperLimits=ul,
                                                      jointRanges=jr,
                                                      restPoses=rp,
                                                      jointDamping=jd,
                                                      solver=ik_solver)[:num_controlled_joints]
            p.setJointMotorControlArray(bodyIndex=arm_id,
                                        jointIndices=controlled_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=jointPoses,
                                        targetVelocities=targetVelocities,
                                        forces=forces,
                                        positionGains=positionGains,
                                        velocityGains=velocityGains)

            # get the torque applied at each joint as reported by the force/torque sensors
            if steps > burn_in:
                js = p.getJointStates(arm_id, controlled_joints)
                torques.append([j[3] for j in js])

        # plotting
        # fig, ax = plt.subplots()
        # ax.plot(torques)
        # ax.set_ylim([-20, 20])
        # plt.savefig("mass_{}.png".format(block_mass))

        p.disconnect()
        if np.any(np.isnan(torques)):
            print("torques contains {} NaN values!".format(np.sum(np.isnan(torques))))
        return np.array(torques).transpose()


def makedirs(directory):
    """
    Improved version of makedirs that will make a directory if it doesn't exist.
    :param directory: directory path to create
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        # don't get mad if the folder already exists.
        if e.errno != errno.EEXIST:
            raise



def main():
    debug = DEBUG_DEFAULT
    if len(sys.argv) > 1:
        if sys.argv[1] == "debug":
            debug = True
        else:
            print("Argument not recognized. You can pass the argument 'debug'")
            exit(0)

    dataset_types = ["test"]
    dataset_sizes = [10]
    for dataset_size, dataset_type in zip(dataset_sizes, dataset_types):
        print("Running")
        overall_start_time = time.time()
        for i in range(dataset_size):
            start_time = time.time()
            datapoint_dir = "example_output/{}/{}".format(dataset_type, i)
            makedirs(datapoint_dir)

            mass1 = [np.random.random()]
            with KinovaSim(debug) as ks:
                torques1 = ks.run(theta=mass1)

            # save out the data
            np.save(os.path.join(datapoint_dir, "mass1.npy"), mass1, allow_pickle=False)
            np.save(os.path.join(datapoint_dir, "torques1.npy"), torques1, allow_pickle=False)

            now = time.time()
            print("completed run {:03d}. Elapsed time: {:.2f}s Total: {:.2f}s".format(i+1, now - start_time, now - overall_start_time))


if __name__ == "__main__":
    main()
