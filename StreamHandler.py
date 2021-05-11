"""
deeplabcut wrapper class to process live streams in real time

Created by Nicholas Thomas
Last Edited 4/5/2021

Notes:

Function List:
init(CamIdx, dropFrames=False)
beginCapture(maxFrames = None, labelVideo = True, shuffle = 0, vidOut = False, fps = 30,
             resizeFactor = None, outputDir = None, capStreamFps = False)
getMetaData()
getPoseData()

"""

import yaml
import os
import sys
import cv2
import time
import pandas as pd
import numpy as np
import os.path
import argparse
import tensorflow as tf
import pdb

from datetime import datetime
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
from pathlib import Path
from tqdm import tqdm
from deeplabcut.utils import auxiliaryfunctions
from skimage.util import img_as_ubyte
from threading import Thread
from queue import Queue
from collections import OrderedDict as ordDict



class StreamHandler:
    def __init__(self, camIdx, dropFrames = False):
        """Short summary

        Initializes stream handler object

        Parameters
        ----------
        CamIdx : type = list of strings, list of integers, or integer
            The camera or video reference(s) from which the streams will be pulled for labeling. On
            a laptop, the webcam will generally be able to be referenced by passing in a 1 or 0
            depending on the laptop. For videos, pass in a list of the full file path for each video
            that you want to simultaneously analyze.
        dropFrames : type = boolean
            dropFrames is a flag that when set to true, the stream handler will not generate a
            backlog of frames. It will pull the most recent frame available to process and label,
            skipping any frames that were pulled while the previous was being processed. It will
            essentially subsample the video stream at based on the processing rate if that rate is
            below the framerate of the stream.
        """

        #Storage structure for frames, Thinking can sequentially add frames for multiple cameras
        self.queue = Queue()    #Temporary frame storage, only used for retrieval
        self.stopped = False    #Boolean stop for threaded frame retrieval
        self.curFrames = []     #Current Frame or Batch of Frames pointer
        self.dropFrames = dropFrames    #Flag to skip frames if processing speed is lower than
                                        #stream speed
        self.frameCounter = 0   #Number of total frames that have been retrieved from the stream
        self.curIdx = 0         #The index of the current frame to process
        #We have two references for indexing: frameCounter and curIdx. When Dropframes is true,
        #frameCounter=curIdx as the current frame will be the most recently pulled frame. When
        #dropFrames is false, frameCounter >= curIdx as there can be a backlog of frames to process
        #and the current frame to process is not the most recently pulled frame
        self.width = None
        self.height = None
        self.meta = ordDict({
            'Frame Idx':[],
            'Frame Read':[],    #Whether or not the frame was actually read, this can only be false
                                #if dropFrames flag is set to True
            'Time Retrieved':[],
            'Time Read':[],
            'Time Processed':[],
            'Time Displayed':[],
            'Processing Time':[],
            'Displaying Time':[],
            'Total Time':[]
        })

        #If input for stream is an int, will convert into a list, might build in funciton to check
        #all input types. ie if one string is passed in
        if isinstance(camIdx, int):
            camIdx = [camIdx]

        #Curframe will be set back to it initalization if a frame is read or replaced by the
        #following frame if it had not yet been read; self.frame will only be replaced by the next read frame
        if isinstance(camIdx, list):
            self.streams = [] #input streams
            self.poseData = []
            self.frame = [] #most recently retrieved frame
            self.streamCount = len(camIdx) #number of streams
            self.multiple = self.streamCount > 1
            self.camIdx = camIdx

            #self.updateMeta()

        elif camIdx == None:
            print("Initalizing empty Stream Handler")
        else:
            raise Exception('Invalid input for camIdx')

    def beginCapture(self, config, maxFrames = None, labelVideo = True, shuffle = 0, vidOut = False,
                    fps = 30, resizeFactor = None, outputDir = None, capStreamFps = False,
                    printFPS = False, savePose=False, saveMeta=False, scorer = None, filetype = 'h5',
                    display=True, threshold = .1, num_outputs=None, predictionToPlot = None):

        """Short Summary
        Parameters
        ----------
        maxFrames : type = int
            The maximum number of frames you want to collect and analyze
        labelVideo : type = boolean
            A flag which when true, will analyze each frame using deeplabcut and plot the labels
            over the frames
        shuffle : type = int
            The Shuffle for which network iteration of deeplabcut to use
        vidOut : type = boolean
            A flag which when true, will save a video
        fps : type = int
            The framerate of the output video stream. Also if capStreamFps is True, the livestream
            will be set to the framerate
        resizeFactor : type = double
            A double ranging from 0 to 1 which determines by what factor the frames are rescaled
        outputDir : type = string
            The filepath for which the video will be saved
        capStreamFps : type = boolean
            A flag which when true, will synthetically enforce a framerate on the frame retrieval
            speed
        printFPS : type = boolean
            A flag which when true will continually print the framerate of the processing
        savePose : boolean
            A flag which when true will automatically save the pose data
        saveMeta : boolean
            A flag which when true will automatically save the meta data
        scorer : string
            Name of individual running the tests
        filetype : string
            The filetype of the data. Can be either 'h5' or 'csv'
        """

        if not predictionToPlot == None and predictionToPlot > num_outputs:
            raise Error("Not valid prediction selections")

        self.num_outputs = num_outputs

        if labelVideo:
            sess, inputs, outputs, dlc_cfg = self.initializeDLC(config=config, shuffle=shuffle)


        for i, stream in enumerate(self.camIdx):
            if not isinstance(stream, (int, str)):
                raise Exception('Each individual stream must be an int or string')
            self.streams.append(cv2.VideoCapture(stream))
            self.poseData.append([]) #initalizes empty lists for all pose data
            (self.grabbed, frame) = self.streams[i].read()
            self.frame.append(frame)
            if self.height == None:
                self.height = np.shape(frame)[0]
            if self.width == None:
                self.width = np.shape(frame)[1]

        if labelVideo:
            poses = self.analyzeFrames(self.frame, sess, inputs, outputs, dlc_cfg)
            for i in range(self.streamCount):
                self.poseData[i].append(poses[i])

        self.curFrame = self.frame

        '''
        if maxFrames == None:
            self.maxFrames = self.streams[0].get(cv2.CAP_PROP_FRAME_COUNT) - 1
        else:
            self.maxFrames = maxFrames
            '''
        if maxFrames != None:
            self.maxFrames = maxFrames
        else:
            self.maxFrames =None

        #Ensures they are both ints
        if not resizeFactor == None:
            self.height = int(self.height * resizeFactor)
            self.width = int(self.width * resizeFactor)
        self.startTime = time.time() #time reference

        if vidOut:
            os.chdir(outputDir)
            if(outputDir == None):
                raise Exception('Tried to write output videos but no output directory was specified')
            videoWriters = []
            date = datetime.now()
            date = str(date).split('.')[0]
            for i in range(self.streamCount):
                videoWriters.append(cv2.VideoWriter('cam{}-{}.avi'.format(i,date), cv2.VideoWriter_fourcc(*"MJPG"),
                                    fps=fps, frameSize=(self.width, self.height)))

        if capStreamFps:
            self.fps = fps
        else:
            self.fps = None

        self.startThread()
        loopcount = 0
        timeArr = np.zeros((100))
        self.fpsArr = []

        while True:

            if (not self.maxFrames == None and self.curIdx >= self.maxFrames) or  cv2.waitKey(1) == 27:
                print("###STOPPING###")
                self.stopped = True
                self.t.join() #I think this stops the thread properly. Need to look into more but it works
                for stream in self.streams:
                    stream.release()
                if vidOut:
                    for writer in videoWriters:
                        writer.release()
                cv2.destroyAllWindows()
                break

            timeIdx = loopcount % 100 #Make a constant so it is clear this is used in fps calculations
            #READING
            curIdx = self.curIdx #ensures all idx are consistent for current loop iterations
            frame = self.read()
            #print(self.frameCounter, curIdx, len(self.meta['Frame Read']))
            self.meta['Frame Read'][curIdx] = True #might need to move meta update into main while loop

            if not resizeFactor == None:
                for i in range(self.streamCount):
                    frame[i] = cv2.resize(frame[i], None,fx=resizeFactor,fy=resizeFactor)

            if labelVideo:
                self.meta['Time Read'][curIdx] = time.time() - self.startTime

                #PROCESSING
                pose = self.analyzeFrames(frame, sess, inputs, outputs, dlc_cfg, num_outputs=num_outputs)
                for i in range(self.streamCount):
                    self.poseData[i].append(poses[i])

                procTime =  time.time() - self.startTime
                self.meta['Time Processed'][curIdx] = procTime

                #FRAMERATE CALCULATIONS
                timeArr[timeIdx] = procTime
                elapsedTime = timeArr[timeIdx] - timeArr[(timeIdx+1)%100] #%100 should account for edge case of 99
                #TODO Explain Time calculation
                fps = 100 / elapsedTime
                self.fpsArr.append(fps)
                #LABELING and DISPLAYING
                labeledFrames = self.labelFrames(frame, pose, threshold=threshold, predictionToPlot = predictionToPlot)
                for i in range(self.streamCount):
                    if display:
                        cv2.imshow("Stream {}".format(i), labeledFrames[i])
                    if vidOut:
                        videoWriters[i].write(labeledFrames[i])

            else:
                for i in range(self.streamCount):
                    cv2.imshow("Stream {}".format(i), frame[i])
                    if vidOut:
                        videoWriters[i].write(frame[i])


            self.meta['Time Displayed'][curIdx] = time.time() - self.startTime

            if printFPS:
                sys.stdout.write('\r')
                sys.stdout.write(str(fps))
                sys.stdout.flush()
            loopcount  += 1

        if saveMeta:
            self.metaDataDF = self.getMetaData(outputDir = outputDir, filetype = filetype)
        if savePose:
            self.poseDataDF = self.getPoseData(outputDir = outputDir, filetype = filetype, scorer=scorer)

        return

    def getMetaData(self, outputDir = None, filetype = 'h5'):
        """Short Summary
        Returns the meta data for the stream handler object and the processed video stream. The meta
        data consists of time stamps and elapsed time for the different steps in the pipeline:
        Retrieval, Reading, Processing, Displaying. Note that total elapsed time is measured using
        the reading time stamp not the retrieval time stamp

        Note: There appears to be a bug that if you call this method twice the second call will
        sometimes have an additional row of nan values.

        Returns
        -------
        metaData : type = pandas Dataframe
            A dataframe containing all the time data associated with the pipeline
        """
        timeproc = np.array(self.meta['Time Processed'])
        timedisp = np.array(self.meta['Time Displayed'])
        timeread = np.array(self.meta['Time Read'])
        self.meta['Processing Time'] = timeproc - timeread
        self.meta['Displaying Time'] = timedisp - timeproc
        self.meta['Total Time'] = timedisp - timeread
        metaData = pd.DataFrame.from_dict(self.meta)
        if not outputDir == None:
            if filetype == 'h5':
                fileName = outputDir + 'metaData_{}.h5'.format(str(datetime.today().now())[:-7])
                metaData.to_hdf(fileName, key='metaData')
            elif filetype == 'csv':
                fileName = outputDir + 'metaData_{}.csv'.format(str(datetime.today().now())[:-7])
                metaData.to_csv(fileName, key='metaData')
        return metaData

    def getPoseData(self, outputDir = None, filetype='h5', scorer=None):
        """Short Summary
        Returns the pose estimations for all the processed frames. If dropFrames is false, that will
        every frame in the stream. Note that the method will always return a list even if there is
        only 1 stream

        Returns
        -------
        poseDataFrames : type = list pandas Dataframe
            A list of size N (number of streams) with a dataframe for each stream containing all of
            the pose estimation data. Each label in each frame has an X and Y coordinate value
            (based on the rescaled frame size) and a likelihood value (between 0 and 1).
        """

        poseDataFrames = []
        bodyparts = np.repeat(self.bodyparts,3)
        columnNames = ['X', 'Y', 'likelihood'] * len(self.bodyparts)
        if scorer == None:
            scorer = datetime.now().strftime("%B %d, %Y")

        scorer = [scorer] * len(bodyparts)
        for i in range(self.streamCount):
            rows, cols = np.shape(self.poseData)[1:3]
            df = pd.DataFrame((np.reshape(self.poseData[i], (rows, cols*3 ))))
            cols = list(zip(scorer, bodyparts, columnNames))
            df.columns = pd.MultiIndex.from_tuples(cols, names = ["scorer", "bodypart", "coord"])
            poseDataFrames.append(df)
            if not outputDir == None:
                if filetype == 'h5':
                    fileName = outputDir + 'Cam{}_PoseEstimationData_{}.h5'.format(i, str(datetime.today().now())[:-7])
                    df.to_hdf(fileName, key='df')
                elif filetype == 'csv':
                    fileName = outputDir + 'Cam{}_PoseEstimationData_{}.csv'.format(i, str(datetime.today().now())[:-7])
                    df.to_csv(fileName, key='df')
        return poseDataFrames

#HELPER METHODS
####################################################################################################

    #Helper method that starts thread to read frames. Executes independently and continously outside
    #of the main code body
    def startThread(self):
        self.t = Thread(target=self.update)
        self.t.start()
        return self

    #Helper method to stop the threaded operations
    def stop(self):
        self.stopped = True

    #Helper method that is executed on a single thread. It will pull frames from the stream(s)
    #and batch and enque the frames (FIFO).
    def update(self):
        start = time.time() #Reference time against which
        while True:  #If thread is stopped, the loop will stop
            if not self.maxFrames == None:
                if self.frameCounter > (self.maxFrames):
                    break
            if self.stopped:
                print("Stopping")
                break

            self.frameCounter += 1
            frames = []

            #grabs one frame from every stream and batches them in a list
            for stream in self.streams:
                (grabbed, frame) = stream.read()
                frames.append(frame)

            #curIdx is incremented here when dropFrames flag is true as it retrieves the most recent frame
            if (self.dropFrames):
                self.curIdx += 1
                self.curFrames = frames

            end = time.time()

            #This will impose an artifical framerate on frame pulling if an fps is designated in the
            #beginCapture method. This is can help better approximate a livestream from a video for
            #testing and evaluation purposes.
            if not self.fps == None and grabbed:
                elapsedTime = end-start
                if elapsedTime < (1/self.fps):
                    time.sleep(1/self.fps - elapsedTime)
                start = time.time()

            self.queue.put_nowait(frames)
            self.updateMeta()

        return self

    #Helper method which will pull frames from the queue and return that list of frame(s)
    def read(self):
        #only returns once a frame or list of frames (1 frame from each stream) is available
        while True:
            if self.dropFrames:
                if len(self.curFrames) > 0: #curFrame will be an empty array if it has been
                                            #retrieved and there is not a new frame available
                    frame = self.curFrames
                    self.curFrames = []
                    return frame
            #curIdx for when dropFrames flag is False is incremented here as there can be a backlog
            #of frames
            elif not self.queue.empty():
                self.meta['Frame Read'][self.curIdx] = True
                self.curIdx += 1
                frame = self.queue.get_nowait()
                return frame

    #Helper method that adds another row to the meta data, initalizing the values. Note that the
    #meta data is the time stamps and elapsed time for each of the steps in the pipeline
    def updateMeta(self):
        self.meta['Frame Idx'].append(self.frameCounter) #frame idx in progress
        self.meta['Frame Read'].append(False) #frames have not been read yet
        self.meta['Time Retrieved'].append(time.time() - self.startTime) #time pulled from stream
        self.meta['Time Read'].append(np.nan) #time frame is read
        self.meta['Time Processed'].append(np.nan) #time after deeplabcut processing
        self.meta['Time Displayed'].append(np.nan) #time frame is displayed with opencv
        self.meta['Processing Time'].append(np.nan) #time elapsed to process
        self.meta['Displaying Time'].append(np.nan) #time elapsed to dispaly
        self.meta['Total Time'].append(np.nan) #total time from reading to displaying the frame
        return

    #Uses Deeplabcut to analyze a single frame and outputes the pose estimation data
    def analyzeFrames(self, frames, sess, inputs, outputs, dlc_cfg, num_outputs=None):
        frames = [img_as_ubyte(f) for f in frames]
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        if not num_outputs == None and not num_outputs == dlc_cfg.num_outputs:
            dlc_cfg.num_outputs = num_outputs
        pose  = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)
        return pose

    #Adds label markers to multiple frames for each detected label
    def labelFrames(self, frames, poses, threshold = .1, resizeFactor = 1, predictionToPlot = 1):
        labeledFrames = []
        numBodyparts = len(self.bodyparts)
        self.pose = poses
        if not self.num_outputs == None:
            tempIdx = np.zeros((1, self.num_outputs), dtype = bool)
            tempIdx[:, predictionToPlot-1] = 1
            poseIdx = np.repeat(np.array([np.repeat(tempIdx, 3)]), numBodyparts, axis=0).flatten()
            self.poseIdx = poseIdx
        for i, pose in enumerate(poses):
            if not self.num_outputs == None:
                pose = pose[poseIdx]
            poseBool = pose[2::3] > threshold
            xCoords = pose[0::3][poseBool]
            yCoords = pose[1::3][poseBool]
            curBodyparts = self.bodyparts[poseBool]

            frame = frames[i]
            for j, bodypart in enumerate(curBodyparts):
            #cv2.circle(frame, (int(points[i,0]), int(points[i,1])), 2, color = (255,255,255), thickness =2)
                cv2.putText(frame, bodypart, ( int(xCoords[j]), int(yCoords[j]) ), cv2.FONT_HERSHEY_SIMPLEX, .69, (255,255,255), thickness = 2, lineType=cv2.LINE_AA)
            labeledFrames.append(frame)
        return labeledFrames

#Adapted Directly from DeepLabcut: This is the network initalization step
####################################################################################################
    def initializeDLC(self, config, videotype='avi', shuffle=0, trainingsetindex=0, gputouse=0,
                    save_as_csv=False, destFolder=None, crop=None, TFGPUinference=True,
                    dynamic=(False, .5, 10)):

        batchsize=self.streamCount;

        #Temporary hardcoded file path
        # config = '/home/nickt/DeepLabCut/Trial 8-Nick_T-2020-04-08/config.yaml'

        #########################################################################################
        #Taken from Deeplabcut

        if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
            del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

        tf.reset_default_graph()
        start_path=os.getcwd() #record cwd to return to this directory in the end
        cfg = auxiliaryfunctions.read_config(config)
        trainFraction = cfg['TrainingFraction'][trainingsetindex]

        modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
        path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
        try:
            dlc_cfg = load_config(str(path_test_config))
        except FileNotFoundError:
            raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))
        # Check which snapshots are available and sort them by # iterations
        try:
          Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(modelfolder , 'train'))if "index" in fn])
        except FileNotFoundError:
          raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(shuffle,shuffle))

        if cfg['snapshotindex'] == 'all':
            print("Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
            snapshotindex = -1
        else:
            snapshotindex=cfg['snapshotindex']

        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]

        print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

        ##################################################
        # Load and setup CNN part detector
        ##################################################

        # Check if data already was generated:
        dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
        trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]
        # Update number of output and batchsize
        dlc_cfg['num_outputs'] = cfg.get('num_outputs', dlc_cfg.get('num_outputs', 1))
        dlc_cfg['batch_size']=batchsize

        if dynamic[0]: #state=true
            #(state,detectiontreshold,margin)=dynamic
            print("Starting analysis in dynamic cropping mode with parameters:", dynamic)
            dlc_cfg['num_outputs']=1
            TFGPUinference=False
            dlc_cfg['batch_size']=1
            print("Switching batchsize to 1, num_outputs (per animal) to 1 and TFGPUinference to False (all these features are not supported in this mode).")

        # Name for scorer:
        if dlc_cfg['num_outputs']>1:
            if  TFGPUinference:
                print("Switching to numpy-based keypoint extraction code, as multiple point extraction is not supported by TF code currently.")
                TFGPUinference=False
            print("Extracting ", dlc_cfg['num_outputs'], "instances per bodypart")
            xyz_labs_orig = ['x', 'y', 'likelihood']
            suffix = [str(s+1) for s in range(dlc_cfg['num_outputs'])]
            suffix[0] = '' # first one has empty suffix for backwards compatibility
            xyz_labs = [x+s for s in suffix for x in xyz_labs_orig]
        else:
            xyz_labs = ['x', 'y', 'likelihood']

        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
        DLCscorer = 'nickt'
        pdindex = pd.MultiIndex.from_product([dlc_cfg['all_joints_names'],
                                              xyz_labs],
                                             names=['bodyparts', 'coords'])
        #####################################################################################
        tmpcfg = yaml.load(config)
        stream = open(config, 'r')
        tmpcfg = yaml.load(stream)
        bodyparts = tmpcfg['bodyparts']
        self.bodyparts = np.array(bodyparts)
        self.sess = sess #redundant
        return sess, inputs, outputs, dlc_cfg
