import pandas as pd
import numpy as np 

from PIL import Image
import os
import cv2
import pickle
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits.mplot3d import Axes3D

import torch
from scipy.stats import multivariate_normal
from mpl_toolkits.axes_grid1 import make_axes_locatable

import time as time_func
import random
import math
import torch.utils.data as utils_data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb

import sys

import utils

import read_csv as rc

import param as p
import plot_func as pf

class BEVPlotter:
    """
    This class is for plotting results.
    """
    def __init__(self,
        data_dir,
        prediction_dir,
        map_dir,
        file_id,
        cart = p.CART):
        self.file_id = file_id
        self.cart = cart
        # Map
        with open(map_dir, 'rb') as map_file:
            map_data = pickle.load(map_file)
        
        if cart:
            lane_markings = map_data['lane_nodes']
            lane_y_min1 = min([min(lane['l'][:,1]) for lane in lane_markings])
            lane_y_min2 = min([min(lane['r'][:,1]) for lane in lane_markings])
            lane_y_max1 = max([max(lane['l'][:,1]) for lane in lane_markings])
            lane_y_max2 = max([max(lane['r'][:,1]) for lane in lane_markings])
            lane_y_min = min(lane_y_min1, lane_y_min2)
            #min(, 0)
            
            lane_y_max = max(lane_y_max1, lane_y_max2)

            lane_x_max1 = max([max(lane['l'][:,0]) for lane in lane_markings])
            lane_x_max2 = max([max(lane['l'][:,0]) for lane in lane_markings])
            lane_x_min1 = min([min(lane['l'][:,0]) for lane in lane_markings])
            lane_x_min2 = min([min(lane['l'][:,0]) for lane in lane_markings])
            lane_x_min = min(lane_x_min1, lane_x_min2)
            
            lane_x_max = max(lane_x_max1, lane_x_max2)
            
            self.image_width = int((lane_x_max-lane_x_min)*p.WIDTH_X)
            self.image_height = int(-1*(lane_y_max-lane_y_min)*p.HEIGHT_X)
            WIDTH_B = int(lane_x_min*p.WIDTH_X)
            HEIGHT_B = int(lane_y_max*p.HEIGHT_X)
            self.BIAS = np.ones((1,2), dtype= int)
            
            self.BIAS[0,0] *= WIDTH_B
            self.BIAS[0,1] *= HEIGHT_B
            
        else:
            self.image_height = int(map_data['image_height']*p.HEIGHT_X)
            self.image_width = int(map_data['image_width']*p.WIDTH_X)
            lane_markings = map_data['lane_nodes_frenet']
            self.BIAS = np.ones((1,2), dtype= int)*0
        print(min([min(-1*lane['l'][:,1]) for lane in lane_markings]))        
        self.lanes = [lane_markings[0]['l']]
        for lane_marking in lane_markings:
            self.lanes.append(lane_marking['r'])
            #self.lanes.append(lane_marking['l'])
        print(self.image_height, self.image_width)
        self.background_image = np.ones((self.image_height, self.image_width,3),\
                         dtype=np.uint8)*p.COLOR_CODES['BACKGROUND']
        self.background_image = self.background_image.astype(np.uint8)
        
        self.y_lower_lim = math.inf
        self.y_upper_lim = -1*math.inf
        for lane in self.lanes:
            self.y_lower_lim = min(self.y_lower_lim, min(lane[:,1]))
            self.y_upper_lim = max(self.y_upper_lim, max(lane[:,1]))
            
            lane[:,0] = lane[:,0]*p.WIDTH_X 
            lane[:,1] = lane[:,1]*p.HEIGHT_X + p.HEIGHT_B
            if self.cart == False:
                lane[:,1] = self.image_height-lane[:,1]
            
            
            lane = lane.astype(int)
            if self.cart ==False:
                lane[:,1] = np.mean(lane[:,1])
                lane = lane[[0,-1]]
            
            self.background_image = pf.draw_line(self.background_image,
                                                lane - self.BIAS, 
                                                p.COLOR_CODES['LANE_MARKING'])
        print(self.y_lower_lim, self.y_upper_lim)
        # Traj Data
        
        data_df = pd.read_csv(data_dir)
       
        if cart:
            data_df = data_df[['id','frame','x', 'y', 'laneId', 'height', 'width']]
        else:
            data_df = data_df[['id','frame','s', 'd', 'laneId', 'height', 'width']]
        
        data_df = data_df.sort_values(by=['id', 'frame'])
        self.data = data_df.to_numpy()
        #pdb.set_trace()
        # Traj Pred
        
        prediction_df = pd.read_csv(prediction_dir)
        prediction_df = prediction_df.sort_values(by=['file', 'id', 'frame'])
        prediction = prediction_df.to_numpy()
        prediction = prediction[prediction[:,0]==file_id]
        
        
        self.prediction_ids = prediction[:,:3]
        self.prediction_ids = np.concatenate(\
            (np.arange(0,len(self.prediction_ids)).reshape(-1,1),\
              self.prediction_ids), axis = 1)
        # itr, file, id, frame
        #print(self.prediction_ids)
        self.mode_probs = np.zeros((prediction.shape[0],3))
        if 'mode_prob' in list(prediction_df):
            self.multimodal = True
            self.pred_trajs = np.zeros((prediction.shape[0],\
                                         3, p.TGT_SEQ_LEN, 2))
        
        else:
            self.multimodal = False
            self.pred_trajs = np.zeros((prediction.shape[0],\
                                         1, p.TGT_SEQ_LEN, 2))
        self.gt_trajs = np.zeros((prediction.shape[0],\
                                  1, p.TGT_SEQ_LEN, 2)) 
        if self.multimodal:
            for i in range(prediction.shape[0]):
                self.mode_probs[i] = \
                    np.array(list(map(float,prediction[i,3].split(';')))) 
                if cart:
                    self.pred_trajs[i,0,:,0] = \
                        np.array(list(map(float,prediction[i,10].split(';')))) 
                    self.pred_trajs[i,0,:,1] = \
                        np.array(list(map(float,prediction[i,11].split(';')))) 
                    self.pred_trajs[i,1,:,0] = \
                        np.array(list(map(float,prediction[i,12].split(';'))))
                    self.pred_trajs[i,1,:,1] = \
                        np.array(list(map(float,prediction[i,13].split(';'))))
                    self.pred_trajs[i,2,:,0] = \
                        np.array(list(map(float,prediction[i,14].split(';'))))
                    self.pred_trajs[i,2,:,1] = \
                        np.array(list(map(float,prediction[i,15].split(';'))))
                else:
                    self.pred_trajs[i,0,:,0] = \
                        np.array(list(map(float,prediction[i,4].split(';')))) 
                    self.pred_trajs[i,0,:,1] = \
                        np.array(list(map(float,prediction[i,5].split(';')))) 
                    self.pred_trajs[i,1,:,0] = \
                        np.array(list(map(float,prediction[i,6].split(';'))))
                    self.pred_trajs[i,1,:,1] = \
                        np.array(list(map(float,prediction[i,7].split(';'))))
                    self.pred_trajs[i,2,:,0] = \
                        np.array(list(map(float,prediction[i,8].split(';'))))
                    self.pred_trajs[i,2,:,1] = \
                        np.array(list(map(float,prediction[i,9].split(';'))))
        else:
            for i in range(prediction.shape[0]):
                if cart:
                    self.pred_trajs[i,0,:,0] = \
                        np.array(list(map(float,prediction[i,5].split(';')))) 
                    self.pred_trajs[i,0,:,1] = \
                        np.array(list(map(float,prediction[i,6].split(';')))) 
                else:
                    self.pred_trajs[i,0,:,0] = \
                        np.array(list(map(float,prediction[i,3].split(';')))) 
                    self.pred_trajs[i,0,:,1] = \
                        np.array(list(map(float,prediction[i,4].split(';')))) 
        '''
        if self.cart == False:
            for i in range(prediction.shape[0]):
                self.gt_trajs[i,0,:,0] = \
                            np.array(list(map(float,prediction[i,-2].split(';')))) 
                self.gt_trajs[i,0,:,1] = \
                    np.array(list(map(float,prediction[i,-1].split(';'))))
        '''        
        
    def plot_predictions(self):
        tv_ids = np.unique(self.prediction_ids[:,2]) 
        n_plot = 0
        rmse = 0
        n_sample = 0
        n_collision = np.zeros((p.TGT_SEQ_LEN))
        collision_time = np.zeros((p.TGT_SEQ_LEN))
        p_n_collision = np.zeros((p.TGT_SEQ_LEN))
        p_collision_time = np.zeros((p.TGT_SEQ_LEN))
        n_offroad = np.zeros((p.TGT_SEQ_LEN))
        offroad_time = np.zeros((p.TGT_SEQ_LEN))
        p_n_offroad = np.zeros((p.TGT_SEQ_LEN))
        p_offroad_time = np.zeros((p.TGT_SEQ_LEN))
        n_prediction = 0
        for tv_itr, tv_id in enumerate(tv_ids):
            print('TV:{}'.format(tv_id))
            if p.DEBUG_MODE and tv_itr>50:
                break

            tv_data = self.data[self.data[:,0]==tv_id]
            frames = self.prediction_ids[self.prediction_ids[:,2]==tv_id,3]
            indexes = self.prediction_ids[self.prediction_ids[:,2]==tv_id,0]
            for itr,frame in enumerate(frames):
                n_prediction += 1
                ind = indexes[itr]
                assert(self.prediction_ids[ind, 3] == frame)
                frame_data = self.data[self.data[:,1]== frame]
                tv_hist = tv_data[tv_data[:,1]<=frame]
                tv_gt_future = tv_data[tv_data[:,1]>=frame]
                tv_future = self.pred_trajs[ind]
                tv_gt_from_model = self.gt_trajs[ind]
                tv_mode_prob = self.mode_probs[ind]
                
                if p.CHECK_COL:
                    n_col, col_time, p_n_col, p_col_time, n_off, off_time, p_n_off, p_off_time\
                          = self.check_violation(tv_id, frame, tv_hist, tv_gt_future, 
                        tv_future, tv_gt_from_model,tv_mode_prob)
                    n_collision += n_col
                    collision_time += col_time
                    n_offroad += n_off
                    offroad_time += off_time
                    p_n_collision += p_n_col
                    p_collision_time += p_col_time
                    p_n_offroad += p_n_off
                    p_offroad_time += p_off_time
                
                
                if p.CALC_MET:#TODO: complete code for multimodal
                    mse, fut_len = self.calc_mse(tv_id,frame_data, tv_hist, tv_gt_future,\
                                    tv_future, tv_gt_from_model, tv_mode_prob)
                    rmse +=mse
                    mse = 0
                    fut_len = 1
                    n_sample += fut_len
                if p.VIS_PRED and n_plot<p.N_PLOT:
                    if p.ONE_PER_TRACK and itr>0:
                        continue
                    n_plot +=1
                    self.plot_frame(tv_id,frame_data, tv_hist, tv_gt_future,\
                                 tv_future, tv_gt_from_model, tv_mode_prob, metric = np.sqrt(mse/fut_len)) 
               
        if p.CALC_MET:    
            rmse = np.sqrt(rmse/n_sample)
        else:
            rmse = 0
        print('RMSE: {}, n_samples:{}'.format(rmse, n_sample)) 
        print('N Collisions: {}'.format(n_collision)) 
        print('N Colliding Timesteps: {}'.format(collision_time))      
        print('Prob N Collisions: {}'.format(p_n_collision)) 
        print('N Colliding Timesteps: {}'.format(p_collision_time))
        
        print('N Offroad: {}'.format(n_offroad)) 
        print('N Offroad Timesteps: {}'.format(offroad_time))      
        print('Prob N Offroad: {}'.format(p_n_offroad)) 
        print('N Offroad Timesteps: {}'.format(p_offroad_time))
        print('Total Samples: {}'.format(n_prediction))
        print('CollisionRate:{}, OffRoadRate:{}'.format(\
            sum(n_collision)/(p.N_MODE*n_prediction),\
                  sum(n_offroad)/(p.N_MODE*n_prediction)))
        print('PCollisionRate:{}, POffRoadRate:{}'.format(\
            sum(p_n_collision)/(p.N_MODE*n_prediction),\
                  sum(p_n_offroad)/(p.N_MODE*n_prediction)))

    def check_violation(self, tv_id, frame, tv_hist, tv_gt_future, 
                     tv_future, tv_gt_future_from_model,tv_mode_prob):
        '''
        n_colision: Number of collision (first time-step within a colliding scene)

        '''
        n_collision = np.zeros((p.TGT_SEQ_LEN))
        collision_time = np.zeros((p.TGT_SEQ_LEN))
        p_n_collision = np.zeros((p.TGT_SEQ_LEN))
        p_collision_time = np.zeros((p.TGT_SEQ_LEN))
        n_offroad = np.zeros((p.TGT_SEQ_LEN))
        offroad_time = np.zeros((p.TGT_SEQ_LEN))
        p_n_offroad = np.zeros((p.TGT_SEQ_LEN))
        p_offroad_time = np.zeros((p.TGT_SEQ_LEN))
        tv_hist = np.copy(tv_hist)
        tv_gt_future = np.copy(tv_gt_future)
        tv_future = np.copy(tv_future)
        traj_hist = tv_hist[:,2:4]
        

        traj_fut_gt = tv_gt_future[:,2:4]
        traj_fut_gt = traj_fut_gt[0:-1:p.FPS_DIV]
        traj_fut_gt_len = traj_fut_gt.shape[0]
        fut_len = min(p.TGT_SEQ_LEN, traj_fut_gt_len)
        traj_fut_gt = traj_fut_gt[:fut_len]
        
        # Plot future modes
        n_mode = tv_future.shape[0]
        for i in range(n_mode):
            tv_future[i,:,0] += traj_hist[-1,0]
            tv_future[i,:,1] += traj_hist[-1,1]
        tv_future = tv_future[:,:fut_len]
        mode_sort = np.argsort(-1*tv_mode_prob, axis = -1) 
        assert(n_mode>=p.N_MODE)
        n_mode = p.N_MODE
        tv_future_sorted = tv_future[mode_sort[:n_mode]]

        for i in range(n_mode):
            first_col_seq = -1
            collision_flag = False
            first_off_seq = -1
            offroad_flag = False

            mode_prob = tv_mode_prob[i]
            for seq in range(fut_len):
                frame_data = self.data[self.data[:,1]== frame+seq*p.FPS]
                for veh_itr, veh_id in enumerate(frame_data[:,0]):
                    if veh_id != tv_id:
                        if pf.check_collision(tv_future_sorted[i,seq], 
                                              [frame_data[veh_itr, 2], 
                                               frame_data[veh_itr, 3]], 
                                               p.LONG_COL,
                                               p.LAT_COL):
                            collision_time[seq] +=1
                            p_collision_time[seq] += mode_prob
                            collision_flag = True
                            if first_col_seq<0:
                                first_col_seq = seq
                        if pf.check_offroad(tv_future_sorted[i,seq,1], self.y_lower_lim, self.y_upper_lim):
                            offroad_time[seq] +=1
                            p_offroad_time[seq] += mode_prob
                            offroad_flag = True
                            if first_off_seq<0:
                                first_off_seq = seq
            n_collision[first_col_seq] += collision_flag
            p_n_collision[first_col_seq] += collision_flag*mode_prob

            n_offroad[first_off_seq] += offroad_flag
            p_n_offroad[first_off_seq] += offroad_flag*mode_prob

            
        return n_collision, collision_time, p_n_collision, p_collision_time,\
            n_offroad, offroad_time, p_n_offroad, p_offroad_time                     
    
    def calc_mse(self, tv_id, frame_data, tv_hist, tv_gt_future, 
                     tv_future, tv_gt_future_from_model,tv_mode_prob):
        
        tv_hist = np.copy(tv_hist)
        tv_gt_future = np.copy(tv_gt_future)
        tv_future = np.copy(tv_future)
        traj_hist = tv_hist[:,2:4]
        traj_hist = np.flip(traj_hist, axis= 0)
        
        traj_fut_gt = tv_gt_future[:,2:4]
        traj_fut_gt = traj_fut_gt[0:-1:p.FPS_DIV]
        traj_fut_gt_len = traj_fut_gt.shape[0]
        fut_len = min(p.TGT_SEQ_LEN, traj_fut_gt_len)
        traj_fut_gt = traj_fut_gt[:fut_len]
        
        # Plot future modes
        n_mode = tv_future.shape[0]
        for i in range(n_mode):
            if self.cart == False:
                tv_future[i,:,0] += traj_hist[0,0]
                tv_future[i,:,1] += traj_hist[0,1]
        tv_future = tv_future[:,:fut_len]
   
        if n_mode>1:
            best_mode = np.argmax(tv_mode_prob)
            # TODO: complete mm metric
        else:
           mse = np.sum((tv_future[0]-traj_fut_gt)**2)
        return mse, fut_len
    
    def plot_frame(self, tv_id, frame_data, tv_hist, tv_gt_future, tv_future,
                    tv_gt_future_from_model, tv_mode_prob, metric):
        frame = frame_data[0,1]
        
        # Initialise the image
        image = np.copy(self.background_image)
        # Plot Vehicles
        if self.cart:
            correct_y = lambda y: y
        else:
            correct_y = lambda y: self.image_height-y
        corner_x = lambda itr: int(frame_data[itr, 2]*p.WIDTH_X-
                                frame_data[itr, 6]*p.WIDTH_X/2)
        corner_y = lambda itr: int(
            correct_y(frame_data[itr, 3]*p.HEIGHT_X+p.HEIGHT_B)
            -frame_data[itr, 5]*p.HEIGHT_X/2)
        if self.cart == False:
            for veh_itr, veh_id in enumerate(frame_data[:,0]):
                if veh_id == tv_id:
                    image = pf.draw_vehicle(image, 
                                    corner_x(veh_itr),
                                    corner_y(veh_itr), 
                                    int(frame_data[veh_itr, 6]*p.WIDTH_X), 
                                    int(frame_data[veh_itr, 5]*p.HEIGHT_X), 
                                    p.COLOR_CODES['TV'])
                else:
                    image = pf.draw_vehicle(image, 
                                    corner_x(veh_itr),
                                    corner_y(veh_itr), 
                                    int(frame_data[veh_itr, 6]*p.WIDTH_X), 
                                    int(frame_data[veh_itr, 5]*p.HEIGHT_X), 
                                    p.COLOR_CODES['SV'])
                                                                                                                                                                                                             
        
        # Plot history
        traj_hist = tv_hist[:,2:4]
        traj_hist = np.flip(traj_hist, axis= 0)
        traj_hist = traj_hist[0::p.FPS_DIV]
        traj_hist[:,0] *= p.WIDTH_X
        traj_hist[:,1] = traj_hist[:,1]*p.HEIGHT_X+ p.HEIGHT_B
        if self.cart == False:
            traj_hist[:,1] = self.image_height-traj_hist[:,1]
        traj_hist = traj_hist.astype(int)
        traj_hist_len = traj_hist.shape[0]
        traj_hist = traj_hist[:min(p.MAX_OBS_LEN, traj_hist_len)]
        #pdb.set_trace()
        image = pf.draw_line(image,traj_hist - self.BIAS, p.COLOR_CODES['GT_TRAJ'])
        # Plot GT Fut Traj
        traj_fut_gt = tv_gt_future[:,2:4]
        
        #exit()
        traj_fut_gt = traj_fut_gt[0::p.FPS_DIV]
        traj_fut_gt[:,0] *= p.WIDTH_X
        traj_fut_gt[:,1] = traj_fut_gt[:,1]*p.HEIGHT_X+ p.HEIGHT_B
        if self.cart == False:
            traj_fut_gt[:,1] = self.image_height-traj_fut_gt[:,1]
        traj_fut_gt = traj_fut_gt.astype(int)
        traj_fut_gt_len = traj_fut_gt.shape[0]
        fut_len = min(p.TGT_SEQ_LEN, traj_fut_gt_len)
        traj_fut_gt = traj_fut_gt[:fut_len]
        
        image = pf.draw_line(image,traj_fut_gt - self.BIAS, p.COLOR_CODES['GT_FUT_TRAJ'])
        # Plot future modes
        n_mode = tv_future.shape[0]
        for i in range(n_mode):
            traj_fut = tv_future[i]
            if self.cart:
                traj_fut[:,0] = traj_fut[:,0]*p.WIDTH_X
                traj_fut[:,1] = traj_fut[:,1]*p.HEIGHT_X
            else:
                traj_fut[:,0] = traj_fut[:,0]*p.WIDTH_X + traj_hist[0,0]
                traj_fut[:,1] = -1*traj_fut[:,1]*p.HEIGHT_X + traj_hist[0,1]
            traj_fut = traj_fut.astype(int)
            traj_fut = traj_fut[:fut_len]
            
            image = pf.draw_line(image,traj_fut - self.BIAS, 
                                p.COLOR_CODES['PR_TRAJ'][i])
        '''
        tv_gt_future_from_model = tv_gt_future_from_model[0]
        tv_gt_future_from_model[:,0] = tv_gt_future_from_model[:,0]*p.WIDTH_X + traj_hist[0,0]
        tv_gt_future_from_model[:,1] = -1*tv_gt_future_from_model[:,1]*p.HEIGHT_X + traj_hist[0,1]
        tv_gt_future_from_model = tv_gt_future_from_model.astype(int)
        tv_gt_future_from_model = tv_gt_future_from_model[:fut_len]
        
        image = pf.draw_line(image,tv_gt_future_from_model, 
                                p.COLOR_CODES['PR_TRAJ'][-1])
        '''
        # Write probabilities 
        #Save
        
        file_fname = 'File{}'\
            .format(self.file_id)
        
        tv_fname = 'TV{}'\
            .format(tv_id)
        image_name = 'FRAME{}_RMSE{}.png'\
            .format(frame, metric)
        
        f_list = [file_fname, tv_fname]
        file_dir = p.SAVE_DIR
        if os.path.exists(file_dir) == False:
                os.mkdir(file_dir)
        for fold in f_list:
            file_dir = os.path.join(file_dir,fold)
            if os.path.exists(file_dir) == False:
                os.mkdir(file_dir)
        file_dir = os.path.join(file_dir, image_name)
        if not cv2.imwrite(file_dir, image):
            raise Exception("Could not write image: " + file_dir)
        
        
        

if __name__ =="__main__":
        
    bev_plotter = BEVPlotter( 
        data_dir = p.DATASET_DIR,
        prediction_dir = p.PREDICTION_DIR,
        map_dir = p.MAP_DIR,
        file_id = 39)
    bev_plotter.plot_predictions()
    