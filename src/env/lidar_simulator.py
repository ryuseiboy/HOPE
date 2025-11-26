import sys
sys.path.append("../")
sys.path.append(".")
import math
import numpy as np
from shapely.geometry import LineString, Point
from shapely.affinity import affine_transform

from env.vehicle import State,VehicleBox

ORIGIN = Point((0,0))

class LidarSimlator():
    def __init__(self, 
        lidar_range:float = 10.0,
        lidar_num:int = 120
    ) -> None:
        '''
        Args:
            lidar_range(float): the max distance that the obstacle can be dietected.
            lidar_num(int): the beam num of the lidar simulation.
        '''
        self.lidar_range = lidar_range
        self.lidar_num = lidar_num
        self.lidar_lines_cache = {}
        self.vehicle_boundary_cache = {}
        self._ensure_lidar_setup(lidar_num)

    def _ensure_lidar_setup(self, beam_num:int):
        if beam_num in self.lidar_lines_cache:
            return
        lidar_lines = []
        for a in range(beam_num):
            lidar_lines.append(LineString(((0,0), (math.cos(a*math.pi/beam_num*2)*self.lidar_range,\
                 math.sin(a*math.pi/beam_num*2)*self.lidar_range))))
        self.lidar_lines_cache[beam_num] = lidar_lines
        self.vehicle_boundary_cache[beam_num] = self._calc_vehicle_boundary(lidar_lines)

    def get_observation(self, ego_state:State, obstacles:list, beam_num:int=None):
        '''
        Get the lidar observation from the vehicle's view.

        Args:
            ego_state: the state of ego car.
            obstacles: the list of obstacles in map
            beam_num: lidar beam num to use. Default None uses self.lidar_num.

        Return:
            lidar_obs(np.array): the lidar data in sequence of angle, with the length of lidar_num.
        '''
        if beam_num is None:
            beam_num = self.lidar_num

        self._ensure_lidar_setup(beam_num)

        ego_pos = (ego_state.loc.x, ego_state.loc.y, ego_state.heading)
        rotated_obstacles = self._rotate_and_filter_obstacles(ego_pos, obstacles)
        lidar_obs = self._fast_calc_lidar_obs(rotated_obstacles, beam_num)
        lidar_obs = np.array(lidar_obs - self.vehicle_boundary_cache[beam_num])
        if beam_num != self.lidar_num:
            lidar_obs = self._upsample_lidar_obs(lidar_obs, beam_num, self.lidar_num)
        return lidar_obs
    
    def _calc_vehicle_boundary(self, lidar_lines:list):
        lidar_base = []
        for l in lidar_lines:
            distance = l.intersection(VehicleBox).distance(ORIGIN)
            lidar_base.append(distance)
        return np.array(lidar_base)

    def _rotate_and_filter_obstacles(self, ego_pos:tuple, obstacles:list):
        '''
        Rotate the obstacles around the vehicle and remove the obstalces which is out of lidar range.
        '''
        x, y, theta = ego_pos
        a = math.cos(theta)
        b = math.sin(theta)
        x_off = -x*a - y*b
        y_off = x*b - y*a
        affine_mat = [a, b, -b, a, x_off, y_off]

        rotated_obstacles = []
        for obs in obstacles:
            rotated_obs = affine_transform(obs, affine_mat)
            if rotated_obs.distance(ORIGIN) < self.lidar_range:
                rotated_obstacles.append(rotated_obs)
        
        return rotated_obstacles
    
    def _fast_calc_lidar_obs(self, obstacles:list, beam_num:int):
        '''
        Obtain the lidar observation making use of numpy builtin matrix acceleration.

        Parameter:
            obstacles ( list(LinearRing) ): the obstacles around the vehicle which have been transformed to the ego referrence.

        Return:
            lidar_obs (np.ndarray): in shape (LIDAR_NUM,)
        '''

        # Line 1: the lidar ray, ax + by + c = 0
        theta = np.array([a*math.pi/beam_num*2 for a in range(beam_num)]) # (beam_num,)
        a = np.sin(theta).reshape(-1,1) # (beam_num, 1)
        b = -np.cos(theta).reshape(-1,1)
        c = 0

        # convert obstacles(LinerRing) to edges ((x1,y1), (x2,y2))
        x1s, x2s, y1s, y2s = [], [], [], []
        for obst in obstacles:
            obst_coords = np.array(obst.coords) # (n+1,2)
            x1s.extend(list(obst_coords[:-1, 0]))
            x2s.extend(list(obst_coords[1:, 0]))
            y1s.extend(list(obst_coords[:-1, 1]))
            y2s.extend(list(obst_coords[1:, 1]))
        if len(x1s) == 0: # no obstacle around
            return np.ones((beam_num,))*self.lidar_range
        x1s, x2s, y1s, y2s  = np.array(x1s).reshape(1,-1), np.array(x2s).reshape(1,-1),\
            np.array(y1s).reshape(1,-1), np.array(y2s).reshape(1,-1), 
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1,-1) # (1,E)
        e = (x1s - x2s).reshape(1,-1)
        f = (y1s*x2s - x1s*y2s).reshape(1,-1)

        # calculate the intersections
        det = a*e - b*d # (beam_num, E)
        parallel_line_pos = (det==0) # (beam_num, E)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (beam_num, E)
        raw_y = (c*d - a*f)/det

        # select the true intersections, set the false positive interesections to inf
        tmp_inf = 100
        tmp_zero = 1e-8
        # the false positive intersections on line L1(not on ray L1)
        # here we assume the orientation of lidar[0] is 0 rad in the ego coordinate.
        raw_x[:beam_num//4][raw_x[:beam_num//4]<-tmp_zero] = tmp_inf
        raw_x[beam_num//4*3:][raw_x[beam_num//4*3:]<-tmp_zero] = tmp_inf
        raw_x[beam_num//4:beam_num//4*3][raw_x[beam_num//4:beam_num//4*3]>tmp_zero] = tmp_inf
        raw_y[:beam_num//2][raw_y[:beam_num//2]<-tmp_zero] = tmp_inf
        raw_y[beam_num//2:][raw_y[beam_num//2:]>tmp_zero] = tmp_inf
        # the false positive intersections on line L2(not on edge L2)
        raw_x[raw_x>np.maximum(x1s, x2s)] = tmp_inf
        raw_x[raw_x<np.minimum(x1s, x2s)] = tmp_inf
        raw_y[raw_y>np.maximum(y1s, y2s)] = tmp_inf
        raw_y[raw_y<np.minimum(y1s, y2s)] = tmp_inf
        # the (L1, L2) which are parallel
        raw_x[parallel_line_pos] = tmp_inf

        lidar_obs = np.min(np.sqrt(raw_x**2 + raw_y**2), axis=1) # (beam_num,)
        lidar_obs = np.clip(lidar_obs, 0, self.lidar_range)
        return lidar_obs

    def _upsample_lidar_obs(self, lidar_obs:np.ndarray, source_num:int, target_num:int):
        """
        Upsample lidar observations from `source_num` beams to `target_num` beams by linear interpolation over angle.
        """
        if source_num == target_num:
            return lidar_obs
        src_angles = np.linspace(0, 2*np.pi, source_num, endpoint=False)
        tgt_angles = np.linspace(0, 2*np.pi, target_num, endpoint=False)
        # add wrap-around point to keep circular continuity
        src_angles_ext = np.concatenate([src_angles, [2*np.pi]])
        src_values_ext = np.concatenate([lidar_obs, lidar_obs[:1]])
        return np.interp(tgt_angles, src_angles_ext, src_values_ext)

