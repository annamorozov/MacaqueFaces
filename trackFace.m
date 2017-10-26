%% Setup
inputFolder = 'D:\MaqFACS\For Analysis\Lisa Parr - G_10012_X1\18345_FramesCropped\TrainingData';
outputFolder = 'D:\Results\FaceTracking\Lisa';
fullOutputVidFileName = fullfile(outputFolder,'Lisa Parr - G_10012_X1');

% Create the face detector object.
Threshold=7; % Threshold of face cascade object detector, set low to increase sensitivity, high to reduce number of false positives
%faceDetector = vision.CascadeObjectDetector(); 
modelPath=fullfile(cd,'XMLFiles','MacaqueFrontalFaceModel.xml');% assumes models are located in xmlfiles directory; please change if this is not the case
faceDetector=vision.CascadeObjectDetector(modelPath,'MergeThreshold',Threshold); 

% Create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Create video writer object
vw = VideoWriter(fullOutputVidFileName);
open(vw);

%% Get images from dir
extension = 'png';

currentFolder = pwd;
cd(inputFolder);

% Get list of all [extension] files in this directory
pattern = sprintf('*.%s',extension);
imagefiles = dir(pattern);
numimgs = length(imagefiles);  

cd(currentFolder);

%% Detection and Tracking

%runLoop = true;
numPts = 0;
frameCount = 0;

while frameCount < 400 && frameCount < numimgs
    % Get the next frame.
    imgname = imagefiles(frameCount+1).name;
    imgFullFileName = fullfile(inputFolder,imgname);
    %imgFileName = sprintf('Frame%6.6d.png', frameCount + 1);
    %imgFullFileName = fullfile(inputFolder, imgFileName);
    videoFrame = imread(imgFullFileName);

    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    
    if numPts < 10
        % Detection mode.
        fprintf('Frame %d: detection mode.\n', frameCount);
        
        bbox = faceDetector.step(videoFrameGray);
        
        if ~isempty(bbox)
            if frameCount == 1
                % the rectangle represented as [x, y, w, h]
                %TODO: handle this in appropriate way
                firstBboxArea =  bbox(1,3)* bbox(1,4); 
            end
        
            iCorrectFace = 1; 
            nofaces=size(bbox,1);
            if(nofaces ~= 1)
                fprintf('More than 1 face in frame %d\n', frameCount);
                
                bboxAreaArray = zeros(1,nofaces);
                for i=1:nofaces
                    bboxAreaArray(i) =  bbox(i,3)* bbox(i,4); 
                end
                % find the closest to the desired area
                [~,iCorrectFace] = min(abs(bboxAreaArray-firstBboxArea));
            end
            
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(iCorrectFace, :));

            % Re-initialize the point tracker.
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % Save a copy of the points.
            oldPoints = xyPoints;

            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPoints = bbox2points(bbox(iCorrectFace, :));

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display detected corners.
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
            %writeVideo(vw,videoFrame);
        end
    else
        % Tracking mode.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints, 1);

        if numPts >= 10
            % Estimate the geometric transformation between the old points
            % and the new points.
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

            % Apply the transformation to the bounding box.
            bboxPoints = transformPointsForward(xform, bboxPoints);

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the face being tracked.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display tracked points.
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

            % Reset the points.
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end
    end
    writeVideo(vw,videoFrame);
end

%% Clean up
release(pointTracker);
release(faceDetector);
close(vw);


