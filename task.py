import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        

    def get_reward(self,task_name):
        """Uses current pose of sim to return reward."""
        reward=0
        if(task_name=='Sample_Task'):
            reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        elif(task_name=='Takeoff'):
            
            #print('target_pos',self.target_pos)
            #print('sim_pose',self.sim.pose)
            reward =reward+(1.-.1*(abs(self.sim.pose[2] - self.target_pos[2])))
            if(abs(np.max(self.sim.v[:3]))<=1 or abs(np.max(self.sim.v[:3]))>10):
                reward=reward-0.5  #panalty the reward if the vertical velocity diminishes
            else:
                reward=reward+0.1  # bonus if vertical velocity doesn't dimishes

            
                
                
            #print("reward", reward)
            
        elif(task_name=='Hover'):
            #reward/Panalty for positional changes
            x_i=self.sim.pose[2]
            x_f=self.target_pos[2]
            x_c=(x_i+x_f)/2
            if(x_i<=0):
                reward=reward+(-0.2)
            elif(x_i>0 and x_i<x_c):
                reward=reward+(1-(x_c-x_i)*0.02)
            elif(x_i>x_c and x_i<x_f):
                reward=reward-(1-(x_f-x_i)*0.02)
            

            
            
        return reward

    def step(self, rotor_speeds,task_name):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(task_name) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
