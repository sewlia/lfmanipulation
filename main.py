import mpld3
import csv
import numpy as np
from numpy import linalg as la
from numpy.linalg import inv
import pydot
from IPython.display import SVG, display
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    AbstractValue,
    AngleAxis,
    DiagramBuilder,
    Diagram,
    Integrator,
    JacobianWrtVariable,
    JointSliders,
    LeafSystem,
    MeshcatVisualizer,
    MultibodyPlant,
    MultibodyPositionToGeometryPose,
    namedview,
    ConnectPlanarSceneGraphVisualizer,
    Parser,
    PiecewisePolynomial,
    PiecewisePose,
    ExternallyAppliedSpatialForce,
    Quaternion,
    Rgba,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SceneGraph,
    Simulator,
    StartMeshcat,
    TrajectorySource,
    ConstantVectorSource,
    BasicVector
)
from pydrake.visualization import ModelVisualizer, VisualizationConfig, ApplyVisualizationConfig
from pydrake.multibody.math import SpatialForce, SpatialVelocity
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from manipulation import running_as_notebook, ConfigureParser
from manipulation.station import MakeHardwareStation, load_scenario
from manipulation.scenarios import AddMultibodyTriad, SetColor
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.systems.primitives import LogVectorOutput
from pydrake.solvers import Solve
from enum import Enum

class ControllerMode(Enum):
    PRE_GRASP_POSITION = 1
    GRIPPER_OPEN = 2
    APPROACH = 3
    GRIPPER_CLOSED = 4
    LEADER_PRE_GRASP = 5
    LEADER_GRIPPER_OPEN = 6
    LEADER_APPROACH = 7
    LEADER_GRIPPER_CLOSED = 8
    LEADER_FOLLOWER = 9
    
class PositionInitialController(LeafSystem):

    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._youbot1_index = plant.GetModelInstanceByName("youbot1")
        self._youbot2_index = plant.GetModelInstanceByName("youbot2")
        self._youbot3_index = plant.GetModelInstanceByName("youbot3")

        self._slab_index = plant.GetModelInstanceByName("slab")

        self._G1 = plant.GetFrameByName("palm_link", self._youbot1_index)
        self._G2 = plant.GetFrameByName("palm_link", self._youbot2_index)
        self._G3 = plant.GetFrameByName("palm_link", self._youbot3_index)


        self._Base1 = plant.GetFrameByName("base_footprint", youbot1_index)
        self._Base2 = plant.GetFrameByName("base_footprint", youbot2_index)
        self._Base3 = plant.GetFrameByName("base_footprint", youbot3_index)

        self._slab_frame = plant.GetFrameByName("base_link", self._slab_index)

        self.log_file_path_u1 = '/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/youbot1_u.csv'
        self.log_file_path_u2 = '/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/youbot2_u.csv'
        self.log_file_path_u3 = '/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/youbot3_u.csv'
        self.log_file_path_gamma_hat1 = '/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/gamma_hat1.csv'
        self.log_file_path_gamma_hat2 = '/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/gamma_hat2.csv'
        self.log_file_path_u1_lf  = '/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/u1_lf.csv'
        self.log_file_path_u2_lf  = '/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/u2_lf.csv'
        self.log_file_path_u3_lf  = '/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/u3_lf.csv'
        
        self.log_file_path_e_x = '/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/e_x.csv'
        # Ensure the log file is cleared or created at initialization

        self._W = plant.world_frame()

        # The fingers are not actuated for end-effector to reach a position
        self._joint_indices1 = [
            plant.GetJointByName(j, self._youbot1_index).position_start()
            for j in ( "virtual_joint_x", "virtual_joint_y", "virtual_joint_theta", "arm_joint_1"
                      , "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"
                      )
        ]
        self._joint_indices2 = [
            plant.GetJointByName(j, self._youbot2_index).position_start()
            for j in ( "virtual_joint_x", "virtual_joint_y", "virtual_joint_theta", "arm_joint_1"
                      , "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"
                      )
        ]
        self._joint_indices3 = [
            plant.GetJointByName(j, self._youbot3_index).position_start()
            for j in ( "virtual_joint_x", "virtual_joint_y", "virtual_joint_theta", "arm_joint_1"
                      , "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"
                      )
        ]

        # Joint indices are numbered for the entire multi-body plant, hence the shift of indices
        self._joint_indices22 = [x - 14 for x in self._joint_indices2]
        self._joint_indices33 = [x - 28 for x in self._joint_indices3]
        
        # Sets the phase of the system. All robots move to a different tasks simultaneously if triggered.
        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(ControllerMode.PRE_GRASP_POSITION)
        )

        self.DeclareVectorInputPort("youbot1_state",28)
        self.DeclareVectorInputPort("youbot2_state",28)
        self.DeclareVectorInputPort("youbot3_state",28)
        self.DeclareVectorInputPort("slab_state",13) #[w, qx, qy, qz, x, y, z] and the velocities
        self.DeclareVectorInputPort("x_o_d", 6)
        self.DeclareVectorInputPort("gamma_o_hat", 6)
    
        self.DeclareVectorOutputPort("youbot1_actuation",10, self.CalculateTorqueYoubot1)
        self.DeclareVectorOutputPort("youbot2_actuation",10, self.CalculateTorqueYoubot2)
        self.DeclareVectorOutputPort("youbot3_actuation",10, self.CalculateTorqueYoubot3)
        self.DeclareVectorOutputPort("dot_x_o_d",6, self.Calculate_Dot_X_O_D)
        self.DeclareVectorOutputPort("dot_gamma_o_hat", 6, self.Calculate_Gamma_O_Hat)


    def Calculate_Gamma_O_Hat(self, context, output):
        slab_velocity = self.get_input_port(3).Eval(context)[7:]
        gains = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0.1]])
        Gamma = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,-9.81]])
        dot_gamma_o_hat = -  gains @ Gamma @slab_velocity.T
                    

        output.SetFromVector(dot_gamma_o_hat)


    def Calculate_Dot_X_O_D(self, context, output):
        x_o_d = self.get_input_port(4).Eval(context)
        X_W_Slab = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._slab_frame)
        position_slab = X_W_Slab.translation()
        euler_slab = RollPitchYaw(X_W_Slab.rotation()).vector()
        x_o = np.hstack([euler_slab, position_slab])
        tau = 0.5 # Higher the value slower the convergence
        dot_x_o_d = (x_o-x_o_d)/tau
        #print(x_o_d)
        output.SetFromVector(dot_x_o_d)

    def CalculateTorqueYoubot1(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        print(context.get_time())
        # Pose of youbot 1, i.e. joint angles and velocities
        position = self.get_input_port(0).Eval(context)[0:14]
        velocity = self.get_input_port(0).Eval(context)[14:]
        
        # Pose of slab, [w, qx, qy, qz, x, y, z] for position 
        slab_position = self.get_input_port(3).Eval(context)[0:7]
        slab_velocity = self.get_input_port(3).Eval(context)[7:]
        
        x_o_d = self.get_input_port(4).Eval(context)

        gamma_o_hat = self.get_input_port(5).Eval(context)


        # The context here is empty and we need to populate it from the multibody plant i.e. from outside this LeafSystem
        self._plant.SetPositions(self._plant_context, self._youbot1_index, position)
        self._plant.SetVelocities(self._plant_context, self._youbot1_index, velocity)
        self._plant.SetPositions(self._plant_context, self._slab_index, slab_position)
        self._plant.SetVelocities(self._plant_context, self._slab_index, slab_velocity)
        
        #tau_g = plant.CalcGravityGeneralizedForces(self._plant_context)

        # Relative transform of frame E1 wrt the world frame
        X_W_E1 = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G1)
        X_W_Slab = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._slab_frame)
        X_B1_E1 = self._plant.CalcRelativeTransform(self._plant_context, self._Base1, self._G1)

        # Retrieving position and orientation of youbot 1 and slab
        position_youbot1 = X_W_E1.translation()
        euler_youbot1 = RollPitchYaw(X_W_E1.rotation()).vector()

        position_slab = X_W_Slab.translation()
        euler_slab = RollPitchYaw(X_W_Slab.rotation()).vector()
        
        # Needed to specify the grasping point. The final grasping point is epsilon close to this.
        point_position = position_slab + [-0.25, 0, 0]

        # Relative transform of the end effector in the object's frame
        X_O_E1 = self._plant.CalcRelativeTransform(self._plant_context, self._slab_frame, self._G1)
        
        # Retrieveing position of end effector in base frame
        position_end_effector_in_base = X_B1_E1.translation()
        euler_end_effector_in_base = RollPitchYaw(X_B1_E1.rotation()).vector()
       


        J_o1 = np.array([[1,0,0,0,0,0],
                         [0,1,0,0,0,0],
                         [0,0,1,0,0,0],
                         [0,X_O_E1.translation()[2],-X_O_E1.translation()[1],1,0,0],
                         [ -X_O_E1.translation()[2],0,X_O_E1.translation()[0],0,1,0],
                         [X_O_E1.translation()[1],-X_O_E1.translation()[0],0,0,0,1]])
        
        J_or = np.array([[1,np.sin(euler_slab[0])*np.tan(euler_slab[1]),np.cos(euler_slab[0])*np.tan(euler_slab[1]),0,0,0],
                         [0,np.cos(euler_slab[0]),-np.sin(euler_slab[1]),0,0,0],
                         [0,np.sin(euler_slab[0])/np.cos(euler_slab[1]),np.cos(euler_slab[0])/np.cos(euler_slab[1]),0,0,0],
                         [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        #print(J_o1)
        
        J_G1 = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G1,
            [0, 0, 0],
            self._W,
            self._W,
        )
        # Only selecting the desired columns
        J_G1 = J_G1[:,self._joint_indices1]
        
        #print(np.shape(J_G1))
        #print(position_end_effector_in_base[1])
        error_null = np.array([position_end_effector_in_base[0]-0.3,position_end_effector_in_base[1],0,0,0,0,0,0 ])
        #print('error', error_null)
        #print('inverse', np.shape(np.linalg.pinv(J_G1)))
        null_jacobian = np.eye(8)-np.linalg.pinv(J_G1)@J_G1
        #print('null j', null_jacobian)
        
        torque_null =  null_jacobian @ error_null.T
        #print(torque_null)
        position_full = np.hstack((euler_youbot1, position_youbot1))
        #exit()
        #print(error_null)

        # Changing phases
        # Commanding torques to the robot
        Gamma = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,-9.81]])
            
        if mode == ControllerMode.PRE_GRASP_POSITION:
            kp = 5
            kv = 4
            u = -kp*np.eye(6)@(position_full-[np.pi/2, 0, np.pi/2, -0.875, 0, 0.4]).T - kv*np.eye(6)@J_G1@velocity[self._joint_indices1].T 
            #print(context.get_time())
            #print(u)

            torque = (J_G1).T@u.T 
            torque = np.hstack([torque, 0, 0])
            #
            #torque = np.hstack([torque_null,0,0])
            if la.norm(u)<=0.05:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.GRIPPER_OPEN)
        
        if mode == ControllerMode.GRIPPER_OPEN:
            gain = 5 # May not be needed since the effort is limited to 1
            torque = - gain * (position[12:14] - [0.03, 0.03])
            torque = np.hstack([0,0,0,0,0,0,0,0, torque])
            if la.norm(torque)<=0.00001:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.APPROACH)
        
        if mode == ControllerMode.APPROACH:
            kp = 0.3
            kv = 5 # High velocity damping to approach the object slowly
            u = -kp * np.eye(6)@(position_full-np.hstack([np.pi/2, 0, np.pi/2, point_position])).T - kv*np.eye(6)@J_G1@velocity[self._joint_indices1].T
            torque = (J_G1).T@u.T #+  torque_null
            torque = np.hstack([torque, 0, 0])
            if (la.norm(position_full-np.hstack([np.pi/2, 0, np.pi/2, point_position])))<=0.1:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.GRIPPER_CLOSED)
        
        if mode == ControllerMode.GRIPPER_CLOSED:

            gain = 100
            torque = - gain*(position[12:14] - [0, 0])
            torque = np.hstack([ 0,0,0,0,0,0,0,0, torque])
            
            if abs(torque[8])+abs(torque[9])<=1.02:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.LEADER_PRE_GRASP)

        if mode == ControllerMode.LEADER_PRE_GRASP or mode == ControllerMode.LEADER_GRIPPER_OPEN or mode == ControllerMode.LEADER_APPROACH or mode == ControllerMode.LEADER_APPROACH or mode ==ControllerMode.LEADER_GRIPPER_CLOSED :
            gain = 100
            torque2 = - gain*(position[12:14] - [0, 0])
            u = -1 * np.eye(6)@(position_full-np.hstack([np.pi/2, 0, np.pi/2, point_position])).T - 1*np.eye(6)@J_G1@velocity[self._joint_indices1].T
            torque1 = (J_G1).T@u.T #+ torque_null
            torque = np.hstack([torque1,torque2])
        
        if mode ==ControllerMode.LEADER_FOLLOWER:
            kp = 1
            x_o = np.hstack([euler_slab, position_slab])
            kv = 1
            delta_x_e = x_o_d - x_o
            u = inv(J_o1) @ (kp * inv(J_or)  @ delta_x_e.T - kv * slab_velocity.T + Gamma@gamma_o_hat.T)
            #with open(self.log_file_path_u1, 'a', newline='') as file:
            #    writer = csv.writer(file)
            #    writer.writerow([context.get_time(), u[0], u[1], u[2], u[3], u[4], u[5]])
            torque = (J_G1.T)@u.T+torque_null
            torque = np.hstack([torque, -1, -1])
            with open(self.log_file_path_gamma_hat1, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([context.get_time(), gamma_o_hat[5]])
        output.SetFromVector(torque)

    def CalculateTorqueYoubot2(self, context, output):

        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        position = self.get_input_port(1).Eval(context)[0:14]
        velocity = self.get_input_port(1).Eval(context)[14:]
        slab_position = self.get_input_port(3).Eval(context)[0:7]
        slab_velocity = self.get_input_port(3).Eval(context)[7:]

        x_o_d = self.get_input_port(4).Eval(context)
        gamma_o_hat = self.get_input_port(5).Eval(context)


        self._plant.SetPositions(self._plant_context, self._youbot2_index, position)
        self._plant.SetVelocities(self._plant_context, self._youbot2_index, velocity)
        self._plant.SetPositions(self._plant_context, self._slab_index, slab_position)
        self._plant.SetVelocities(self._plant_context, self._slab_index, slab_velocity)

        
        X_W_E2 = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G2)
        X_W_Slab = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._slab_frame)
        X_B2_E2 = self._plant.CalcRelativeTransform(self._plant_context, self._Base2, self._G2)

        position_youbot2 = X_W_E2.translation()
        euler_youbot2 = RollPitchYaw(X_W_E2.rotation()).vector()
        
        position_slab = X_W_Slab.translation()
        euler_slab = RollPitchYaw(X_W_Slab.rotation()).vector()

        point_position = position_slab + [0.25, 0, 0]
        
        position_end_effector_in_base = X_B2_E2.translation()
        euler_end_effector_in_base = RollPitchYaw(X_B2_E2.rotation()).vector()

        # Relative transform of the end effector in the object's frame
        X_O_E2 = self._plant.CalcRelativeTransform(self._plant_context, self._slab_frame, self._G2)
        

        J_o2 = np.array([[1,0,0,0,0,0],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,0],
                    [0,X_O_E2.translation()[2],-X_O_E2.translation()[1],1,0,0],
                    [ -X_O_E2.translation()[2],0,X_O_E2.translation()[0],0,1,0],
                    [X_O_E2.translation()[1],-X_O_E2.translation()[0],0,0,0,1]])
        
        J_or = np.array([[1,np.sin(euler_slab[0])*np.tan(euler_slab[1]),np.cos(euler_slab[0])*np.tan(euler_slab[1]),0,0,0],
                         [0,np.cos(euler_slab[0]),-np.sin(euler_slab[1]),0,0,0],
                         [0,np.sin(euler_slab[0])/np.cos(euler_slab[1]),np.cos(euler_slab[0])/np.cos(euler_slab[1]),0,0,0],
                         [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        J_G2 = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G2,
            [0, 0, 0],
            self._W,
            self._W,
        )

        position_full = np.hstack((euler_youbot2, position_youbot2))
        J_G2 = J_G2[:,self._joint_indices2]
        error_null = np.array([position_end_effector_in_base[0]-0.3,position_end_effector_in_base[1],0,0,0,0,0,0 ])
        #print(position_end_effector_in_base)
        #print('error', error_null)
        #print('inverse', np.shape(np.linalg.pinv(J_G1)))
        null_jacobian = np.eye(8)-np.linalg.pinv(J_G2)@J_G2
        #null_jacobian = np.eye(8)-(J_G2).T@(np.linalg.pinv(J_G2)).T
        #print('null j', null_jacobian)
        torque_null = - null_jacobian @ error_null.T
        
        Gamma = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,-9.81]])

        if mode == ControllerMode.PRE_GRASP_POSITION:
            kp = 3
            kv = 2
            u = -kp*np.eye(6)@(position_full-[np.pi/2, 0, -np.pi/2, 0.875, 0, 0.4]).T - kv*np.eye(6)@J_G2@velocity[self._joint_indices22].T
            torque = (J_G2).T@u.T #+ torque_null
            torque = np.hstack([torque, 0, 0])
            if la.norm(u)<=0.05:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.GRIPPER_OPEN)

        if mode == ControllerMode.GRIPPER_OPEN:
            torque = -  (position[12:14] - [0.03, 0.03])
            torque = np.hstack([0,0,0,0,0,0,0,0, torque])
            if la.norm(torque)<=0.00001:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.APPROACH)

        if mode == ControllerMode.APPROACH:
            kp = 0.3
            ko=0.3
            kv = 5
            u = -np.array([[ko,0,0,0,0,0],[0,ko,0,0,0,0],[0,0,ko,0,0,0],[0,0,0,kp,0,0],[0,0,0,0,kp,0],[0,0,0,0,0,kp]])@(position_full-np.hstack([np.pi/2, 0, -np.pi/2, point_position])).T - kv*np.eye(6)@J_G2@velocity[self._joint_indices22].T
            torque = (J_G2).T@u.T#+ torque_null
            torque = np.hstack([torque, 0, 0])
            if (la.norm(position_full-np.hstack([np.pi/2, 0, -np.pi/2, point_position])))<=0.1:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.GRIPPER_CLOSED)

        if mode == ControllerMode.GRIPPER_CLOSED:

            gain = 100
            torque = -  gain*(position[12:14] - [0, 0])
            
            torque = np.hstack([0,0,0,0,0,0,0,0, torque])
            
            if abs(torque[8])+abs(torque[9])<=1.02:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.LEADER_PRE_GRASP)
            
        if mode == ControllerMode.LEADER_PRE_GRASP or mode == ControllerMode.LEADER_GRIPPER_OPEN or mode == ControllerMode.LEADER_APPROACH or mode == ControllerMode.LEADER_APPROACH or mode ==ControllerMode.LEADER_GRIPPER_CLOSED:
            gain = 100
            u = -1 * np.eye(6)@(position_full-np.hstack([np.pi/2, 0, -np.pi/2, point_position])).T - 1*np.eye(6)@J_G2@velocity[self._joint_indices22].T
            torque1 = (J_G2).T@u.T#+ torque_null
            torque2 = - gain*(position[12:14] - [0, 0])
            #print(torque2)
            torque = np.hstack([torque1,torque2])

        if mode ==ControllerMode.LEADER_FOLLOWER:
            kp = 1
            kv = 1
            x_o = np.hstack([euler_slab, position_slab])
            delta_x_e = x_o_d - x_o
            u =  inv(J_o2) @ (kp * inv(J_or)  @ delta_x_e.T - kv * slab_velocity.T + Gamma@gamma_o_hat.T)
            with open(self.log_file_path_u2_lf, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([context.get_time(), u[0], u[1], u[2], u[3], u[4], u[5]])
            torque = (J_G2.T)@u.T+torque_null
            torque = np.hstack([torque, -1, -1])
            print('gamma_o_hat', gamma_o_hat)
            
            with open(self.log_file_path_gamma_hat2, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([context.get_time(), gamma_o_hat[5]])
        output.SetFromVector(torque)

    def CalculateTorqueYoubot3(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        # Pose of youbot 1, i.e. joint angles and velocities
        position = self.get_input_port(2).Eval(context)[0:14]
        velocity = self.get_input_port(2).Eval(context)[14:]
        
        # Pose of slab, [w, qx, qy, qz, x, y, z] for position 
        slab_position = self.get_input_port(3).Eval(context)[0:7]
        slab_velocity = self.get_input_port(3).Eval(context)[7:]

        # The context here is empty and we need to populate it from the multibody plant i.e. from outside this LeafSystem
        self._plant.SetPositions(self._plant_context, self._youbot3_index, position)
        self._plant.SetVelocities(self._plant_context, self._youbot3_index, velocity)
        self._plant.SetPositions(self._plant_context, self._slab_index, slab_position)
        self._plant.SetVelocities(self._plant_context, self._slab_index, slab_velocity)

        # Relative transform of frame E1 wrt the world frame
        X_W_E3 = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G3)
        X_W_Slab = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._slab_frame)
        X_B3_E3 = self._plant.CalcRelativeTransform(self._plant_context, self._Base3, self._G3)

        # Retrieving position and orientation of youbot 3 and slab
        position_youbot3 = X_W_E3.translation()
        euler_youbot3 = RollPitchYaw(X_W_E3.rotation()).vector()

        position_slab = X_W_Slab.translation()
        euler_slab = RollPitchYaw(X_W_Slab.rotation()).vector()
        x_o = np.hstack([euler_slab, position_slab])
        v_o = slab_velocity

        point_position = position_slab + [0, 0.25, 0]

        position_end_effector_in_base = X_B3_E3.translation()
        euler_end_effector_in_base = RollPitchYaw(X_B3_E3.rotation()).vector()
        
        X_O_E3 = self._plant.CalcRelativeTransform(self._plant_context, self._slab_frame, self._G3)

        J_o3 = np.array([[1,0,0,0,0,0],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,0],
                    [0,X_O_E3.translation()[2],-X_O_E3.translation()[1],1,0,0],
                    [ -X_O_E3.translation()[2],0,X_O_E3.translation()[0],0,1,0],
                    [X_O_E3.translation()[1],-X_O_E3.translation()[0],0,0,0,1]])
        
        J_or = np.array([[1,np.sin(euler_slab[0])*np.tan(euler_slab[1]),np.cos(euler_slab[0])*np.tan(euler_slab[1]),0,0,0],
                         [0,np.cos(euler_slab[0]),-np.sin(euler_slab[1]),0,0,0],
                         [0,np.sin(euler_slab[0])/np.cos(euler_slab[1]),np.cos(euler_slab[0])/np.cos(euler_slab[1]),0,0,0],
                         [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        
        # Jacobian for differential kinematics
        J_G3 = self._plant.CalcJacobianSpatialVelocity(
        self._plant_context,
        JacobianWrtVariable.kQDot,
        self._G3,
        [0, 0, 0],
        self._W,
        self._W,
        )
        J_G3 = J_G3[:,self._joint_indices3]

        # Pose of end-effector in the world frame
        position_full = np.hstack((euler_youbot3, position_youbot3))
        #print(context.get_time())
        torque = [0,0,0,0,0,0,0,0,0,0]


        if mode == ControllerMode.LEADER_PRE_GRASP:
            # Only selecting the desired columns
            kp = 3
            kv = 3
            u = -kp*np.eye(6)@(position_full-np.hstack([np.pi/2, 0, 0, 0, 1, 0.4])).T - kv*np.eye(6)@J_G3@velocity[self._joint_indices33].T
            torque = (J_G3).T@u.T #+ torque_null
            torque = np.hstack([torque, 0, 0])
            #print(torque)
            #exit()
            if la.norm(u)<=0.07:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.LEADER_GRIPPER_OPEN)

        if mode == ControllerMode.LEADER_GRIPPER_OPEN:
            torque = -  (position[12:14] - [0.03, 0.03])
            torque = np.hstack([0,0,0,0,0,0,0,0, torque])
            if la.norm(torque)<=0.00001:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.LEADER_APPROACH)

        if mode == ControllerMode.LEADER_APPROACH:
            kp = 1
            ko=0.3
            kv = 5
            u = -kp*(position_full-np.hstack([np.pi/2, 0, 0, point_position])).T - kv*np.eye(6)@J_G3@velocity[self._joint_indices33].T
            torque = (J_G3).T@u.T#+ torque_null
            torque = np.hstack([torque, 0, 0])
            
            if (la.norm(position_full-np.hstack([np.pi/2, 0, 0, point_position])))<=0.05:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.LEADER_GRIPPER_CLOSED)

        if mode == ControllerMode.LEADER_GRIPPER_CLOSED:
            gain = 10
            torque = -  gain*(position[12:14] - [0, 0])
            torque = np.hstack([0,0,0,0,0,0,0,0, torque])
            if abs(torque[8])+abs(torque[9])<0.15:
                context.get_mutable_abstract_state(int(self._mode_index)).set_value(ControllerMode.LEADER_FOLLOWER)

        if mode == ControllerMode.LEADER_FOLLOWER:
            #print(context.get_time())

            kp = 0.3
            ko=0.3
            kv = 5
            c1 = 1
            c2 = 1
            # Position Funnels and Controller
            if context.get_time()<=300:
                gamma_phi = 1000000
                gamma_theta = 1000000
                gamma_psi = 1000000
                gamma_x = (15-3)*np.exp(-0.3*(context.get_time()-250))+3
                #print('gamma_x', gamma_x)
                gamma_y = (15-3)*np.exp(-0.3*(context.get_time()-250))+3
                gamma_z = (15-3)*np.exp(-0.3*(context.get_time()-250))+3

                # Velocity Funnels and Controller
                gamma_omega_x =  1000000
                gamma_omega_y =  1000000
                gamma_omega_z =  1000000
                gamma_v_x =  (15-3)*np.exp(-0.3*(context.get_time()-250))+3
                gamma_v_y =  (15-3)*np.exp(-0.3*(context.get_time()-250))+3
                gamma_v_z =  (15-3)*np.exp(-0.3*(context.get_time()-250))+3

                gamma= np.diag([gamma_phi,gamma_theta, gamma_psi, gamma_x, gamma_y, gamma_z])
                #print('gamma', gamma)
                error = x_o - [0, 0, 0, 1, 1, 0.4]
            elif context.get_time()>=300 and context.get_time()<=350:
                gamma_phi = 1000000
                gamma_theta = 1000000
                gamma_psi = 1000000
                gamma_x = (15-3)*np.exp(-0.3*(context.get_time()-300))+3
                #print('gamma_x', gamma_x)
                gamma_y = (15-3)*np.exp(-0.3*(context.get_time()-300))+3
                gamma_z = (15-3)*np.exp(-0.3*(context.get_time()-300))+3

                # Velocity Funnels and Controller
                gamma_omega_x =  1000000
                gamma_omega_y =  1000000
                gamma_omega_z =  1000000
                gamma_v_x =  (15-3)*np.exp(-0.3*(context.get_time()-300))+3
                gamma_v_y =  (15-3)*np.exp(-0.3*(context.get_time()-300))+3
                gamma_v_z =  (15-3)*np.exp(-0.3*(context.get_time()-300))+3

                gamma= np.diag([gamma_phi,gamma_theta, gamma_psi, gamma_x, gamma_y, gamma_z])
                #print('gamma', gamma)
                error = x_o - [0, 0, 0, 0, 1.414, 0.4]
            elif context.get_time()>=350 and context.get_time()<=400:
                gamma_phi = 1000000
                gamma_theta = 1000000
                gamma_psi = 1000000
                gamma_x = (15-3)*np.exp(-0.3*(context.get_time()-350))+3
                #print('gamma_x', gamma_x)
                gamma_y = (15-3)*np.exp(-0.3*(context.get_time()-350))+3
                gamma_z = (15-3)*np.exp(-0.3*(context.get_time()-350))+3

                # Velocity Funnels and Controller
                gamma_omega_x =  1000000
                gamma_omega_y =  1000000
                gamma_omega_z =  1000000
                gamma_v_x =  (15-3)*np.exp(-0.3*(context.get_time()-350))+3
                gamma_v_y =  (15-3)*np.exp(-0.3*(context.get_time()-350))+3
                gamma_v_z =  (15-3)*np.exp(-0.3*(context.get_time()-350))+3

                gamma= np.diag([gamma_phi,gamma_theta, gamma_psi, gamma_x, gamma_y, gamma_z])
                #print('gamma', gamma)
                error = x_o - [0, 0, 0, -1, 1, 0.4]
            elif context.get_time()>=400 and context.get_time()<=450:
                gamma_phi = 1000000
                gamma_theta = 1000000
                gamma_psi = 1000000
                gamma_x = (15-3)*np.exp(-0.3*(context.get_time()-400))+3
                #print('gamma_x', gamma_x)
                gamma_y = (15-3)*np.exp(-0.3*(context.get_time()-400))+3
                gamma_z = (15-3)*np.exp(-0.3*(context.get_time()-400))+3

                # Velocity Funnels and Controller
                gamma_omega_x =  1000000
                gamma_omega_y =  1000000
                gamma_omega_z =  1000000
                gamma_v_x =  (15-3)*np.exp(-0.3*(context.get_time()-400))+3
                gamma_v_y =  (15-3)*np.exp(-0.3*(context.get_time()-400))+3
                gamma_v_z =  (15-3)*np.exp(-0.3*(context.get_time()-400))+3

                gamma= np.diag([gamma_phi,gamma_theta, gamma_psi, gamma_x, gamma_y, gamma_z])
                #print('gamma', gamma)
                error = x_o - [0, 0, 0, -1.414, 0, 0.4]
            else :
                gamma_phi = 1000000
                gamma_theta = 1000000
                gamma_psi = 1000000
                gamma_x = (15-3)*np.exp(-0.3*(context.get_time()-450))+3
                #print('gamma_x', gamma_x)
                gamma_y = (15-3)*np.exp(-0.3*(context.get_time()-450))+3
                gamma_z = (15-3)*np.exp(-0.3*(context.get_time()-450))+3

                # Velocity Funnels and Controller
                gamma_omega_x =  1000000
                gamma_omega_y =  1000000
                gamma_omega_z =  1000000
                gamma_v_x =  (15-3)*np.exp(-0.3*(context.get_time()-450))+3
                gamma_v_y =  (15-3)*np.exp(-0.3*(context.get_time()-450))+3
                gamma_v_z =  (15-3)*np.exp(-0.3*(context.get_time()-450))+3

                gamma= np.diag([gamma_phi,gamma_theta, gamma_psi, gamma_x, gamma_y, gamma_z])
                #print('gamma', gamma)
                error = x_o - [0, 0, 0, 0, 0, 0.4]
            #with open(self.log_file_path_e_x, 'a', newline='') as file:
            #    writer = csv.writer(file)
            #    writer.writerow([context.get_time(), error[0], error[1], error[2], error[3], error[4], error[5]])
            #print('x_o', x_o)
            #print('error', error)
            xi_x = inv(gamma) @ error.T
            #print('xi_x', xi_x)
            varepsilon_x = np.array([np.log((1+xi_x[0])/(1-xi_x[0])), 
                                        np.log((1+xi_x[1])/(1-xi_x[1])), 
                                        np.log((1+xi_x[2])/(1-xi_x[2])), 
                                        np.log((1+xi_x[3])/(1-xi_x[3])), 
                                        np.log((1+xi_x[4])/(1-xi_x[4])), 
                                        np.log((1+xi_x[5])/(1-xi_x[5]))])
            #print('vare_x', varepsilon_x)
            r_x = np.diag([2/(1-xi_x[0]**2), 2/(1-xi_x[1]**2), 2/(1-xi_x[2]**2), 2/(1-xi_x[3]**2), 2/(1-xi_x[4]**2), 2/(1-xi_x[5]**2)])
            #print('rx', r_x)
            v_r = - c1 * J_or @ inv(gamma) @ r_x @ varepsilon_x.T
            #print('v_r', v_r)

            gamma_v= np.diag([gamma_omega_x,gamma_omega_y, gamma_omega_z, gamma_v_x, gamma_v_y, gamma_v_z])
            #print('v_o', v_o)
            #print('v_r', v_r)
            error_v = v_o - v_r 
            #print('error_v', error_v)
            xi_v = inv(gamma_v) @ error_v.T
            varepsilon_v = np.array([np.log((1+xi_v[0])/(1-xi_x[0])), 
                                        np.log((1+xi_v[1])/(1-xi_v[1])), 
                                        np.log((1+xi_v[2])/(1-xi_v[2])), 
                                        np.log((1+xi_v[3])/(1-xi_v[3])), 
                                        np.log((1+xi_v[4])/(1-xi_v[4])), 
                                        np.log((1+xi_v[5])/(1-xi_v[5]))])
            r_v = np.diag([2/(1-xi_v[0]**2), 2/(1-xi_v[1]**2), 2/(1-xi_v[2]**2), 2/(1-xi_v[3]**2), 2/(1-xi_v[4]**2), 2/(1-xi_v[5]**2)])
            u_1 = - c2 * inv(J_o3) @ inv(gamma_v) @ r_v @ varepsilon_v.T
            #print('u_1', u_1)
            torque = (J_G3).T@u_1.T
            #with open(self.log_file_path_xi_x, 'a', newline='') as file:
            #    writer = csv.writer(file)
            #    writer.writerow([context.get_time(), u_1[0], u_1[1], u_1[2], u_1[3], u_1[4], u_1[5]])
            #with open(self.log_file_path_u3, 'a', newline='') as file:
            #    writer = csv.writer(file)
            #    writer.writerow([context.get_time(), u_1[0], u_1[1], u_1[2], u_1[3], u_1[4], u_1[5]])
            #u = -kp*(position_full-np.hstack([np.pi/2, 0, 0, 1, 1, 0.2])).T - kv*np.eye(6)@J_G3@velocity[self._joint_indices33].T
            #torque = (J_G3).T@u.T
            torque = np.hstack([torque, -0.7, -0.7])
                
        output.SetFromVector(torque)

class FloatingForceForSlab(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._W = plant.world_frame()
        self._slab_index = plant.GetModelInstanceByName("slab")
        self._slab_indices = plant.GetBodyIndices( self._slab_index)
        
        self._slab_frame = plant.GetFrameByName("base_link", self._slab_index)
        #self._mode_index = self.DeclareAbstractState(AbstractValue.Make(ControllerMode.PRE_GRASP_POSITION))
        #slab_position = self.get_input_port(3).Eval(context)[0:7]
        self.DeclareAbstractOutputPort("forces_out",self._AllocateForces, self.CalcForces)
    
    def _AllocateForces(self):
        # This function allocates the initial abstract value for the output.
        # Adjust this if your actual output structure is different.
        return AbstractValue.Make([ExternallyAppliedSpatialForce()])
    
    def CalcForces(self, context, output):
        #mode = context.get_abstract_state(int(self._mode_index)).get_value()
        #print(mode)
        # This function calculates the output based on the current context,
        # updating the provided `output` AbstractValue.
        external_force = ExternallyAppliedSpatialForce()
        #external_force.body_index = self._slab_index
        # Assuming you are setting force values here
        if context.get_time()>251.5:
            force_vector = SpatialForce(np.array([0,0,0]),np.array([0, 0, 0]))
            print(force_vector)
        else:
            force_vector = SpatialForce(np.array([0,0,0]),np.array([0, 0, 9.81 * 0.05]))
        external_force.F_Bq_W = force_vector
        
        # Assuming you want to update the body index and application point as well
        external_force.body_index = self._slab_frame.body().index()
        # Set the application point if necessary. For example, at the body's origin:
        external_force.p_BoBq_B = [0, 0, 0]  # Adjust as needed

        # Update the output abstract value
        output.set_value([external_force])


meshcat = StartMeshcat()

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0002)
parser = Parser(plant, scene_graph)
url1 = "/Users/mayanksewlia/drake_env/lib/python3.11/site-packages/pydrake/share/drake/manipulation/models/youbot/urdf/youbot1_copy.urdf"
url2 = "/Users/mayanksewlia/drake_env/lib/python3.11/site-packages/pydrake/share/drake/manipulation/models/youbot/urdf/youbot2_copy.urdf"
url3 = "/Users/mayanksewlia/drake_env/lib/python3.11/site-packages/pydrake/share/drake/manipulation/models/youbot/urdf/youbot3_copy.urdf"
url4 = "/Users/mayanksewlia/drake_env/lib/python3.11/site-packages/pydrake/share/drake/manipulation/models/slab_object/slab.sdf"

youbot1 = parser.AddModels(url1)
youbot2 = parser.AddModels(url2)
youbot3 = parser.AddModels(url3)
slab =    parser.AddModels(url4)

# Turn off gravity
g = plant.mutable_gravity_field()
g.set_gravity_vector([0,0,-9.81])

youbot1_index = plant.GetModelInstanceByName("youbot1")
youbot2_index = plant.GetModelInstanceByName("youbot2")
youbot3_index = plant.GetModelInstanceByName("youbot3")

slab_index = plant.GetModelInstanceByName("slab")

G1 = plant.GetFrameByName("palm_link", youbot1_index)
G2 = plant.GetFrameByName("palm_link", youbot2_index)
G3 = plant.GetFrameByName("palm_link", youbot3_index)

plant.set_gravity_enabled(youbot1_index, is_enabled= False)
plant.set_gravity_enabled(youbot2_index, is_enabled= False)
plant.set_gravity_enabled(youbot3_index, is_enabled= False)
plant.set_gravity_enabled(slab_index, is_enabled= True)

plant.Finalize()
controller = builder.AddSystem(PositionInitialController(plant))

builder.Connect(
    controller.get_output_port(0), plant.GetInputPort("youbot1_actuation")
)
builder.Connect(
    controller.get_output_port(1), plant.GetInputPort("youbot2_actuation")
)
builder.Connect(
    controller.get_output_port(2), plant.GetInputPort("youbot3_actuation")
)
builder.Connect(
    plant.GetOutputPort("youbot1_state"), controller.get_input_port(0)
)
builder.Connect(
    plant.GetOutputPort("youbot2_state"), controller.get_input_port(1)
)
builder.Connect(
    plant.GetOutputPort("youbot3_state"), controller.get_input_port(2)
)
builder.Connect(
    plant.GetOutputPort("slab_state"), controller.get_input_port(3)
)


integrator1 = builder.AddSystem(Integrator(6))
integrator1.set_name("integrator1")

builder.Connect(
    controller.get_output_port(3), integrator1.get_input_port()
)
builder.Connect(
    integrator1.get_output_port(), controller.get_input_port(4)
)
integrator2 = builder.AddSystem(Integrator(6))
integrator2.set_name("integrator2")

builder.Connect(
    controller.get_output_port(4), integrator2.get_input_port()
)
builder.Connect(
    integrator2.get_output_port(), controller.get_input_port(5)
)
gravity_compensator = builder.AddSystem(FloatingForceForSlab(plant))
#builder.Connect(
#    plant.GetOutputPort("slab_state"), gravity_compensator.get_input_port()
#)
builder.Connect(
    gravity_compensator.get_output_port(), plant.get_applied_spatial_force_input_port()
)



#youbot1_index = plant.GetModelInstanceByName("youbot1")
#youbot2_index = plant.GetModelInstanceByName("youbot2")
#youbot3_index = plant.GetModelInstanceByName("youbot3")

#G1 = plant.GetFrameByName("palm_link", youbot1_index)
#G2 = plant.GetFrameByName("palm_link", youbot2_index)
#G3 = plant.GetFrameByName("palm_link", youbot3_index)
#Base2 = plant.GetFrameByName("base_footprint", youbot2_index)
#AddMultibodyTriad((G1), scene_graph)
#AddMultibodyTriad((G2), scene_graph)
#AddMultibodyTriad((G3), scene_graph)

#AddMultibodyTriad((Base2), scene_graph)
#AddMultibodyTriad((G2), scene_graph)
#AddMultibodyTriad((G3), scene_graph)

visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
#visualization_config = VisualizationConfig()
#visualization_config.publish_contacts = True
#visualization_config.publish_proximity = True
#ApplyVisualizationConfig(visualization_config, builder, meshcat=meshcat)

#js_command = """
#const color = 0xB833FF; // Replace RRGGBB with your desired hex color code
#viewer.background = new THREE.Color(color);
#"""

#logger1 = LogVectorOutput(controller.get_output_port(0), builder)
#logger2 = LogVectorOutput(controller.get_output_port(1), builder)
#logger3 = LogVectorOutput(controller.get_output_port(2), builder)

#logger_1 = LogVectorOutput(controller.get_output_port(0), builder)
#logger_2 = LogVectorOutput(controller.get_output_port(1), builder)
#logger_3 = LogVectorOutput(controller.get_output_port(2), builder)
#logger_4 = LogVectorOutput(controller.get_output_port(2), builder)
logger_5 = LogVectorOutput(integrator1.get_output_port(), builder)
logger_6 = LogVectorOutput(plant.GetOutputPort("slab_state"), builder)
#logger_7 = LogVectorOutput(controller.get_output_port(4), builder) # u1


# Create the diagram.
diagram = builder.Build()
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)

# Initial Conditions
plant.SetPositions(
    plant_context,
    plant.GetModelInstanceByName("youbot1"),
    np.array([-1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
)
plant.SetPositions(
    plant_context,
    plant.GetModelInstanceByName("youbot2"),
    np.array([1,0,np.pi,0,0,0,0,0,0,0,0,0,0,0]),
)
plant.SetPositions(
    plant_context,
    plant.GetModelInstanceByName("youbot3"),
    np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
)
plant.SetPositions(
    plant_context,
    plant.GetModelInstanceByName("slab"),
    np.array([1, 0, 0, 0, 0, 0, 0.4]),
)
#print(plant.get_output_port(17).Eval(plant_context))

visualizer.StartRecording()
simulator.set_target_realtime_rate(5)
simulator.AdvanceTo(600)
visualizer.StopRecording()
visualizer.PublishRecording()

#youbot1_actuation = logger_1.FindLog(simulator.get_context()).data()
#time_stamps = logger_1.FindLog(simulator.get_context()).sample_times()

#youbot1_control = logger_7.FindLog(simulator.get_context()).data()
#time_stamps = logger_7.FindLog(simulator.get_context()).sample_times()

SlabState = logger_6.FindLog(simulator.get_context()).data()
time_stamps1 = logger_6.FindLog(simulator.get_context()).sample_times()

XOD = logger_5.FindLog(simulator.get_context()).data()
time_stamps2 = logger_5.FindLog(simulator.get_context()).sample_times()

#print(youbot1_control)
# Transpose the data if necessary depending on how you wish to organize it
SlabState1 = np.vstack((time_stamps1, SlabState)).T
XOD2 = np.vstack((time_stamps2, XOD)).T
#print(data_to_save)output.txt
#with open('/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/SlabState.csv', 'w', newline='') as csvfile:
#    csvwriter = csv.writer(csvfile)
#    for row in SlabState1:
#        csvwriter.writerow(row)

#with open('/Users/mayanksewlia/My Drive/LF_Manipulation/Manipulation_Log/XOD.csv', 'w', newline='') as csvfile:
#    csvwriter = csv.writer(csvfile)
#    for row in XOD2:
#        csvwriter.writerow(row)


#print(plant.get_output_port(17).get_name())
