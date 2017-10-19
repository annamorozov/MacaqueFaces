%% Setup
inputFolder = 'D:\MaqFACS\For Analysis\Lisa Parr - G_10012_X1';
outputFolder = 'D:\Results\FaceTracking\Lisa';
fullOutputVidFileName = fullfile(outputFolder,'Lisa Parr - G_10012_X1.wmv');

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

%% Detection and Tracking

%runLoop = true;
numPts = 0;
frameCount = 0;

while frameCount < 400
    % Get the next frame.
    imgFileName = sprintf('Frame%6.6d.png', frameCount + 1);
    imgFullFileName = fullfile(inputFolder, imgFileName);
    videoFrame = imread(imgFullFileName);

    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    
    %TODO: uncomment this!!!
    %if numPts < 10
        % Detection mode.
        bbox = faceDetector.step(videoFrameGray);
        
        if ~isempty(bbox)
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));

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
            bboxPoints = bbox2points(bbox(1, :));

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display detected corners.
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
            writeVideo(vw,videoFrame);
        end
    %end
    
end

%% Clean up
release(pointTracker);
release(faceDetector);
close(vw);


