clear
close all
%% Setup
%fullVideoName = 'D:\MaqFACS\For Analysis\Lisa Parr - G_10012_X1.wmv';
%fullVideoName = 'D:\Uri\MirrorExpLappe\040411\040411_1204Cam1.avi';
fullVideoName = 'D:\Raviv\Face videos\x1_181_3534.avi';

%outputFolder = 'D:\Results\FaceTracking\Lisa';
%fullOutputVidFileName = fullfile(outputFolder,'G_10012_X1_foreGroundDetect');

%% Import Video and Initialize Foreground Detector

foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 100);

videoReader = vision.VideoFileReader(fullVideoName);
for i = 1:100
    frame = step(videoReader); % read the next video frame
    foreground = step(foregroundDetector, frame);
end

figure; imshow(frame); title('Video Frame');
figure; imshow(foreground); title('Foreground');

se = strel('square', 2);
filteredForeground = imopen(foreground, se);
figure; imshow(filteredForeground); title('Clean Foreground');

