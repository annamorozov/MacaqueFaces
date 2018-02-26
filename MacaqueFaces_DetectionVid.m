function [] = MacaqueFaces_DetectionVid(video_file,output_dir)
%% Run face detection on a single video file.
% Aimed at videos of rhesus macaque monkeys.
% Produces facial images and stills showing detected faces (as jpegs) and
% CSV file containing locations of detected faces
% video_file: filepath and filename of video
% output_dir: directory to save results to.

%% Parameters
flag=2; % set flag to 1 to run face only detection, set to 2 to run face + eyes/nose detection
Threshold=7; % Threshold of face cascade object detector, set low to increase sensitivity, high to reduce number of false positives
Frames=0; % Set number of frames per second to process, setting to 0 will cause all frames to be processed
%Output_Format='jpg'; % outputs images as jpegs; change to 'png' if desired
%outputDir = 'D:\Raviv\Face videos\070218_DON_shapes_near_far\Results';

[~,inputName,~] = fileparts(video_file);
fullOutputName = fullfile(output_dir, sprintf('%s_detected_face_parts_rect_thresh_7', inputName));

%% Open Detection Models
filepath=fullfile(cd,'MacaqueFaces\XMLFiles','MacaqueFrontalFaceModel.xml');% assumes models are located in xmlfiles directory; please change if this is not the case
FaceDetector=vision.CascadeObjectDetector(filepath,'MergeThreshold',Threshold); 
if flag==2
    filepath=fullfile(cd,'MacaqueFaces\XMLFiles','MacaqueSingleEyeModel.xml');
    EyeDetector=vision.CascadeObjectDetector(filepath,'MergeThreshold',1);
    filepath=fullfile(cd,'MacaqueFaces\XMLFiles','MacaqueNoseModel.xml');
    NoseDetector=vision.CascadeObjectDetector(filepath,'MergeThreshold',1);
end


%% Check Input Arguments
if nargin==1
    output_dir=fullfile(outputDir,'Detection_Outputs');
    mkdir(output_dir)
else
    if ~exist(output_dir,'dir')
        try
            mkdir(output_dir)
        catch
            errordlg('Invalid Output Directory');
        end
    end
end


%% Open Video
try
    video_input=VideoReader(video_file);
catch
    errordlg('Invalid Video File');
end
[~,vidname]=fileparts(video_file);
totalF=floor(video_input.Duration.*video_input.FrameRate); % calculate total number of frames

%% Calculate Interval between Frames
if Frames==0
    FrameInt=1;
else
    FrameInt=round(video_input.FrameRate/Frames);
end

%% Setup variable for face detection information (frame, position of face, coordinates of eyes and nose - if flag is set to 2)
if flag==2
    detection_info=zeros(10000,6);
else
    detection_info=zeros(10000,12);
end
totalfaces=0;

%% Run Face Detection
frameno=0;
wb=waitbar(0,'Detecting Faces','CreateCancelBtn','setappdata(gcbf,''canceling'',1)'); % set up wait bar to track progress and include cancel function
setappdata(wb,'canceling',0);

vw = VideoWriter(fullOutputName);
open(vw);

while hasFrame(video_input)
    
    I=readFrame(video_input); % read in frame
    frameno=frameno+1;
    
    if getappdata(wb,'canceling')
        break
    end
    waitbar(frameno/totalF);
    
    if rem(frameno,FrameInt)==0
        facebox=step(FaceDetector,I); % run face detection
        nofaces=size(facebox,1);
        NewI=I;
        goodfaces=0;
        
        for p=1:nofaces
            
            if nofaces > 1
                fprintf('Frame %d: nofaces = %d\n', frameno, nofaces);
            end
            
            CropI=imcrop(I,facebox(p,:)); % Isolate facial image
            CropI=imresize(CropI,[100,100]); % Resize facial image to 100x100 pixels
            if flag==1
                isgood=1;
            else
                isgood=1;
                % rect_for_crop =  [xmin ymin width height]
                reye=step(EyeDetector,imcrop(CropI,[1,1,50,50])); % run eye detection on upper left quadrant of image (for right eye)
                leye=step(EyeDetector,imcrop(CropI,[51,1,50,50])); % run eye detection on upper right quadrant of image (for left eye)
                nose=step(NoseDetector,imcrop(CropI,[26,1,50,100])); % run nose detection on central column
                if isempty(reye)||isempty(leye)||isempty(nose)
                    isgood=0; % if either of the eyes or the nose is not detected classify this as a bad image
                end
            end
            
            if isgood
                totalfaces=totalfaces+1;
                detection_info(totalfaces,1)=frameno;
                detection_info(totalfaces,2)=p;
                detection_info(totalfaces,3)=facebox(p,1)+(0.5*facebox(p,3));   % Face_x = leftest_x 0.5*Face_Width
                detection_info(totalfaces,4)=facebox(p,2)+(0.5*facebox(p,4));   % Face_y = upper_y + 0.5*Face_Height
                detection_info(totalfaces,5)=facebox(p,3);                      % Face_Width
                detection_info(totalfaces,6)=facebox(p,4);                      % Face_Height
                
                if flag==2
                    % convert bounding boxes from eyes and nose detection to x,y coordinates in original frame
                    
                    % right eye
                    reye_x=reye(1,1)+(reye(3)*0.5);
                    detection_info(totalfaces,7)=(reye_x/100)*facebox(p,3)+facebox(p,1);
                    reye_y=reye(1,2)+(reye(4)*0.5);
                    detection_info(totalfaces,8)=(reye_y/100)*facebox(p,4)+facebox(p,2);
                    
                    reye_rect_x      = (reye(1,1)/100)*facebox(p,3)+facebox(p,1);
                    reye_rect_y      = (reye(1,2)/100)*facebox(p,4)+facebox(p,2);
                    reye_rect_width  = (reye(3)/100)*facebox(p,3);
                    reye_rect_height = (reye(4)/100)*facebox(p,4);
                    reye_rect = [reye_rect_x reye_rect_y reye_rect_width reye_rect_height];
                    
                    % left eye
                    leye_x=leye(1,1)+50+(leye(3)*0.5);
                    detection_info(totalfaces,9)=(leye_x/100)*facebox(p,3)+facebox(p,1);
                    leye_y=leye(1,2)+(leye(4)*0.5);
                    detection_info(totalfaces,10)=(leye_y/100)*facebox(p,4)+facebox(p,2);
                    
                    leye_rect_x      = ((leye(1,1)+50)/100)*facebox(p,3)+facebox(p,1);
                    leye_rect_y      = (leye(1,2)/100)*facebox(p,4)+facebox(p,2);
                    leye_rect_width  = (leye(3)/100)*facebox(p,3);
                    leye_rect_height = (leye(4)/100)*facebox(p,4);
                    leye_rect = [leye_rect_x leye_rect_y leye_rect_width leye_rect_height];
                    
                    % nose
                    nose_x=nose(1,1)+25+(nose(3)*0.5);
                    detection_info(totalfaces,11)=(nose_x/100)*facebox(p,3)+facebox(p,1);
                    nose_y=nose(1,2)+(nose(4)*0.5);
                    detection_info(totalfaces,12)=(nose_y/100)*facebox(p,4)+facebox(p,2);
                    
                    nose_rect_x      = ((nose(1,1)+25)/100)*facebox(p,3)+facebox(p,1);
                    nose_rect_y      = (nose(1,2)/100)*facebox(p,4)+facebox(p,2);
                    nose_rect_width  = (nose(3)/100)*facebox(p,3);
                    nose_rect_height = (nose(4)/100)*facebox(p,4);
                    nose_rect = [nose_rect_x nose_rect_y nose_rect_width nose_rect_height];
                    
                    % insert eye and nose markers into frame
                    NewI=insertMarker(NewI,detection_info(totalfaces,7:8),'Color','red','Size',6);  % right eye
                    NewI=insertMarker(NewI,detection_info(totalfaces,9:10),'Color','green','Size',6);   % left eye
                    NewI=insertMarker(NewI,detection_info(totalfaces,11:12),'Color','cyan','Size',6);   % nose
                    
                    NewI=insertShape(NewI,'Rectangle',reye_rect(p,:),'LineWidth',2,'Color','red');  % right eye
                    NewI=insertShape(NewI,'Rectangle',leye_rect(p,:),'LineWidth',2,'Color','green');   % left eye
                    NewI=insertShape(NewI,'Rectangle',nose_rect(p,:),'LineWidth',2,'Color','cyan');   % nose
                end
                
                goodfaces=goodfaces+1;
                %fname=fullfile(output_dir,[vidname,'_',num2str(frameno),'_',num2str(p),'.',Output_Format]);
                %imwrite(CropI,fname); % write facial image to jpeg file
                %writeVideo(vw,CropI);
                NewI=insertShape(NewI,'Rectangle',facebox(p,:),'LineWidth',4); % insert bounding box into frame to show detected face
            end
        end
        if goodfaces>0
            %fname=fullfile(output_dir,[vidname,'_',num2str(frameno),'_detections.',Output_Format]);
            %imwrite(NewI,fname);  % write processed frame to jpeg file
            writeVideo(vw,NewI);
        else
            writeVideo(vw,I);
        end
    end
end
delete(wb);
close(vw);



%% Write List of Detected Faces and Coordinates to CSV File
detection_info=detection_info(1:totalfaces,:);
if flag==1
    T=array2table(detection_info,'VariableNames',{'FrameNumber','ImageNumber','Face_x','Face_y','Face_Width','Face_Height'});% convert output to table
else
    T=array2table(detection_info,'VariableNames',{'FrameNumber','ImageNumber','Face_x','Face_y','Face_Width','Face_Height',...
        'RightEye_x','RightEye_y','LeftEye_x','LeftEye_y','Nose_x','Nose_y'});% convert output to table
end
fname=fullfile(output_dir,[vidname,'_detection_results.csv']);
writetable(T,fname); % save output to CSV file
