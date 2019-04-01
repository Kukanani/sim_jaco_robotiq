import os
import errno
import pybullet as p
import time
import math
from math import pi
import matplotlib.pyplot as plt
import numpy as np

DEBUG = False
# Set DEBUG = True to enable GUI rendering in PyBullet
# DEBUG = True


class KinovaSim(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def run(self, theta):
        # load arguments
        block_mass, = theta

        # connect and configure sim
        if DEBUG:
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
            if DEBUG:
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
    dataset_types = ["test"]
    dataset_sizes = [300]
    for dataset_size, dataset_type in zip(dataset_sizes, dataset_types):
        print("saving dataset type " + dataset_type)
        overall_start_time = time.time()
        for i in range(dataset_size):
            start_time = time.time()
            datapoint_dir = "example_output/{}/{}".format(dataset_type, i)
            makedirs(datapoint_dir)
            mass1 = [np.random.random()]
            with KinovaSim() as ks:
                torques1 = ks.run(theta=mass1)

            mass2 = [np.random.random()]
            if dataset_type == "test":
                mass2[0] += 1
            with KinovaSim() as ks:
                torques2 = ks.run(theta=mass2)

            # save out the data
            np.save(os.path.join(datapoint_dir, "mass1.npy"), mass1, allow_pickle=False)
            np.save(os.path.join(datapoint_dir, "mass2.npy"), mass2, allow_pickle=False)
            np.save(os.path.join(datapoint_dir, "torques1.npy"), torques1, allow_pickle=False)
            np.save(os.path.join(datapoint_dir, "torques2.npy"), torques2, allow_pickle=False)

            now = time.time()
            print("completed run {}. Elapsed time: {} Total: {}".format(i, now - start_time, now - overall_start_time))


if __name__ == "__main__":
    main()
