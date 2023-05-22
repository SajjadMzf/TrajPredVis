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
        cart = True):
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
        print(min([min(-1*lane['l'][:,1]) for lane in lane_markings]))        
        self.lanes = [lane_markings[0]['l']]
        for lane_marking in lane_markings:
            self.lanes.append(lane_marking['r'])
            self.lanes.append(lane_marking['l'])
        
        self.background_image = np.ones((self.image_height, self.image_width,3),\
                         dtype=np.uint8)*p.COLOR_CODES['BACKGROUND']
        self.background_image = self.background_image.astype(np.uint8)
        
        for lane in self.lanes:
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
        # Traj Data
        
        data_df = pd.read_csv(data_dir)
       
        if cart:
            data_df = data_df[['id','frame','x', 'y', 'laneId', 'height', 'width']]
        else:
            data_df = data_df[['id','frame','s', 'd', 'laneId', 'height', 'width']]
        
        data_df = data_df.sort_values(by=['id', 'frame'])
        self.data = data_df.to_numpy()
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
        print(self.prediction_ids[0,:100])
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
                        np.array(list(map(float,prediction[i,7].split(';')))) 
                    self.pred_trajs[i,0,:,1] = \
                        np.array(list(map(float,prediction[i,8].split(';')))) 
                else:
                    self.pred_trajs[i,0,:,0] = \
                        np.array(list(map(float,prediction[i,3].split(';')))) 
                    self.pred_trajs[i,0,:,1] = \
                        np.array(list(map(float,prediction[i,4].split(';')))) 
        
        # for i in range(prediction.shape[0]):
        #     self.gt_trajs[i,0,:,0] = \
        #                 np.array(list(map(float,prediction[i,-2].split(';')))) 
        #     self.gt_trajs[i,0,:,1] = \
        #         np.array(list(map(float,prediction[i,-1].split(';'))))
                
        
    def plot_predictions(self):
        tv_ids = np.unique(self.prediction_ids[:,2]) 
        n_plot = 0
        rmse = 0
        n_sample = 0
        n_collision = np.zeros((p.TGT_SEQ_LEN))
        collision_time = np.zeros((p.TGT_SEQ_LEN))
        for tv_id in tv_ids:
            tv_data = self.data[self.data[:,0]==tv_id]
            frames = self.prediction_ids[self.prediction_ids[:,2]==tv_id,3]
            if n_plot<p.N_PLOT:
                print('TV:{}, frames:{}'.format(tv_id,frames))
            else:
                print('TV:{}'.format(tv_id))
            indexes = self.prediction_ids[self.prediction_ids[:,2]==tv_id,0]
            for itr,frame in enumerate(frames):
                ind = indexes[itr]
                assert(self.prediction_ids[ind, 3] == frame)
                frame_data = self.data[self.data[:,1]== frame]
                tv_hist = tv_data[tv_data[:,1]<=frame]
                tv_gt_future = tv_data[tv_data[:,1]>=frame]
                tv_future = self.pred_trajs[ind]
                tv_gt_from_model = self.gt_trajs[ind]
                tv_mode_prob = self.mode_probs[ind]
                
                n_col, col_time = self.check_violation(tv_id, frame, tv_hist, tv_gt_future, 
                     tv_future, tv_gt_from_model,tv_mode_prob)
                n_collision += n_col
                collision_time += col_time
                if p.MM == False:#TODO: complete code for multimodal
                    mse, fut_len = self.calc_mse(tv_id,frame_data, tv_hist, tv_gt_future,\
                                    tv_future, tv_gt_from_model, tv_mode_prob)
                    rmse +=mse
                else:
                    mse = 0
                    fut_len = 1
                n_sample += fut_len
                if n_plot<p.N_PLOT:
                    if p.ONE_PER_TRACK and itr>0:
                        break
                    n_plot +=1
                    self.plot_frame(tv_id,frame_data, tv_hist, tv_gt_future,\
                                 tv_future, tv_gt_from_model, tv_mode_prob, metric = np.sqrt(mse/fut_len))
            if p.MM and n_plot>p.N_PLOT:
                break   
               
            
        rmse = np.sqrt(rmse/n_sample)
        
        print('RMSE: {}, n_samples:{}'.format(rmse, n_sample)) 
        print('N Collisions: {}'.format(n_collision))       
    def check_violation(self, tv_id, frame, tv_hist, tv_gt_future, 
                     tv_future, tv_gt_future_from_model,tv_mode_prob):
        n_collision = np.zeros((p.TGT_SEQ_LEN))
        collision_time = np.zeros((p.TGT_SEQ_LEN))
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
        
        for i in range(n_mode):
            first_col_seq = -1
            collision_flag = False
            for seq in range(fut_len):
                frame_data = self.data[self.data[:,1]== frame+seq*p.FPS_DIV]
                for veh_itr, veh_id in enumerate(frame_data[:,0]):
                    if veh_id != tv_id:
                        if pf.check_collision(tv_future[i,seq], 
                                              [frame_data[veh_itr, 2], 
                                               frame_data[veh_itr, 3]], 
                                               p.LONG_COL,
                                               p.LAT_COL):
                            collision_time[seq] +=1
                            collision_flag = True
                            if first_col_seq<0:
                                first_col_seq = seq
                            if seq<5:
                                print('Collision ID-FRAME: {}-{}, Seq: {}'\
                                      .format(tv_id, frame, seq))
            n_collision[first_col_seq] += collision_flag
        return n_collision, collision_time                       
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
        '''
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
        '''                                                                                                                                                                                                     
        
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
                traj_fut[:,1] = traj_fut[:,1]*p.HEIGHT_X + traj_hist[0,1]
            traj_fut = traj_fut.astype(int)
            traj_fut = traj_fut[:fut_len]
            
            image = pf.draw_line(image,traj_fut - self.BIAS, 
                                p.COLOR_CODES['PR_TRAJ'][i])
        
        tv_gt_future_from_model = tv_gt_future_from_model[0]
        tv_gt_future_from_model[:,0] = tv_gt_future_from_model[:,0]*p.WIDTH_X + traj_hist[0,0]
        tv_gt_future_from_model[:,1] = -1*tv_gt_future_from_model[:,1]*p.HEIGHT_X + traj_hist[0,1]
        tv_gt_future_from_model = tv_gt_future_from_model.astype(int)
        tv_gt_future_from_model = tv_gt_future_from_model[:fut_len]

        #image = pf.draw_line(image,tv_gt_future_from_model, 
        #                        p.COLOR_CODES['PR_TRAJ'][-1])

        # Write probabilities 
        #Save
        file_name = 'File{}_TV{}_FRAME{}_RMSE{}.png'\
            .format(self.file_id, tv_id, frame, metric)
        file_dir = os.path.join(p.SAVE_DIR, file_name)
        if not cv2.imwrite(file_dir, image):
            raise Exception("Could not write image: " + file_dir)
        











    def plot(self, file_id_pairs = None, remove_ids_list = None):
        
        plot_ids = file_id_pairs if file_id_pairs is not None else self.plot_ids
        if len(plot_ids)>p.MAX_PLOTS:
            plot_ids = [plot_ids[i] for i in range(p.MAX_PLOTS)]
        self.remove_ids_list = remove_ids_list
        for i, plot_id in enumerate(plot_ids):
            self.plot_one_scenario(plot_id)
    
    
    def plot_one_scenario(self,plot_id):
        
        if plot_id not in self.plot_ids:
            print('file tv pair {}-{} cannot be found!'.format(plot_id[0], plot_id[1]))
            return 
        else:
            scenario_itr = self.plot_ids.index(plot_id)
        

        tv_id = self.sorted_scenarios[scenario_itr]['tv']
        data_file = self.sorted_scenarios[scenario_itr]['data_file']
        traj_min = self.sorted_scenarios[scenario_itr]['traj_min']
        traj_max = self.sorted_scenarios[scenario_itr]['traj_max']
        with open(p.map_paths[data_file], 'rb') as handle:
            map_data = pickle.load(handle)
        track_path = p.track_paths[data_file]
        print(track_path)
        print(p.map_paths[data_file])
        pickle_path = p.frame_pickle_paths[data_file]
        frames_data = rc.read_track_csv(track_path, pickle_path, group_by = 'frames', reload = False, fr_div = p.fr_div)
        
        driving_dir = map_data['driving_dir']
        
        print('FILE-TV: {}-{}, List of Available Frames: {}, dd:{}'.format(
            plot_id[0], 
            plot_id[1], 
            self.sorted_scenarios[scenario_itr]['times'], 
            driving_dir))
        
        np.set_printoptions(precision=2, suppress=True)
        
        # for each time-step
        images = []
        
        for j,time in enumerate(self.sorted_scenarios[scenario_itr]['times']):
            
            if p.PLOT_MAN== False:
                man_preds = []
                man_labels = []
            else:
                man_preds = self.sorted_scenarios[scenario_itr]['man_preds'][j]
                man_labels = self.sorted_scenarios[scenario_itr]['man_labels'][j]
            mode_prob = self.sorted_scenarios[scenario_itr]['mode_prob'][j]
            traj_labels = self.sorted_scenarios[scenario_itr]['traj_labels'][j]
            
            traj_preds = self.sorted_scenarios[scenario_itr]['traj_dist_preds'][j][:,:, :2]
            frames = self.sorted_scenarios[scenario_itr]['frames'][j]
            if plot_id[0]==44 and plot_id[1] == 290:
                pdb.set_trace()
            scenario_tuple = (traj_min, traj_max, man_labels, man_preds, 
                              mode_prob, traj_labels, traj_preds, frames, 
                              frames_data, map_data)
            image = self.plot_one_frame(scenario_itr, tv_id, scenario_tuple, j)
            images.append(image)
        images = np.array(images)
        scenario_id = 'File{}_TV{}_SN{}_F{}'.format(data_file, tv_id, 
                                                    scenario_itr, frames[0])
        pf.save_image_sequence(p.model_name, images, self.traj_vis_dir, 
                               scenario_id, self.remove_ids_list is not None)              

    
    def plot_one_frame(self, scenario_itr, tv_id, scenario_tuple, time):
        summary_image = False
        (traj_min, traj_max, man_labels, man_preds, mode_prob, traj_labels, traj_preds, frames, frames_data, map_data) = scenario_tuple
        driving_dir = map_data['driving_dir']
        image_height = int(map_data['image_height']*p.Y_IMAGE_SCALE)
        image_width = int(map_data['image_width']*p.X_IMAGE_SCALE)
        lane_markings = map_data['lane_nodes_frenet']

        in_seq_len = self.in_seq_len
        tgt_seq_len = self.tgt_seq_len
        frame = frames[in_seq_len-1]
        #print(frames.shape)
        frame_list = [frame_data[rc.FRAME][0] for frame_data in frames_data]
        frame_data = frames_data[frame_list.index(frame)]
        
        
        traj_labels = traj_labels*(traj_max-traj_min)+traj_min
        traj_labels = np.cumsum(traj_labels, axis = 0)
        traj_preds =  traj_preds*(traj_max-traj_min)+traj_min
        traj_preds = np.cumsum(traj_preds, axis = 1)
        #print(traj_labels.shape)
        #pdb.set_trace()  
        
        image = pf.plot_frame(
            lane_markings,
            frame_data,
            tv_id, 
            driving_dir,
            frame,
            man_labels,
            man_preds,
            mode_prob,
            traj_labels,
            traj_preds,
            image_width,
            image_height)            
        return image
        
        

if __name__ =="__main__":
        
    bev_plotter = BEVPlotter( 
        data_dir = p.DATASET_DIR,
        prediction_dir = p.PREDICTION_DIR,
        map_dir = p.MAP_DIR,
        file_id = 39)
    bev_plotter.plot_predictions()
    