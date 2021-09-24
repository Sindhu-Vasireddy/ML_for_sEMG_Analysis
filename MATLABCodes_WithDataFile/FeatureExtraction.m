function [feat,feat_out]=FeatureExtraction(D,featFunc)
    % D=load('Datafile1.mat'); 
    
    if nargin<2
        featFunc='getrmsfeat';
    end
% 
%     switch D
%         case 'D_0'
%             D_0.emg=D_0.emgMov;
%             D_0.force=D_0.force4_1;
%         otherwise
%             fprintf("We have data from Ninapro");
%     end
%          
    
%     emg=D.emgMov;
   
    
    emg=D.emg;% s-EMG input
    force=D.force(:,:);% index finger movement output
    row=find(D.stimulus==44);
    emg_index=zeros([length(row),12]);
    force_index=zeros([length(row),6]);
    for num=1:length(row)
        emg_index(num,:)=emg(row(num),:);
        force_index(num,:)=force(row(num),:);
        
    end

    
    winsize=800;
    wininc=20;
    if exist('ker'),
        parpool(ker,'IdleTimeout', 30);
    end
    
    
    numwin = size(emg_index,1);
    nSignals = size(emg_index,2);
    
    %-------------------------------------------------------------
    % allocate memory
    
    feat = zeros(numwin,nSignals,'single');
    
    %-------------------------------------------------------------
    parfor winInd = 1:numwin-winsize,
        
        if mod(winInd-1,wininc)==0,
            disp(['Feature Extraction Progress: ' num2str(round(winInd*10000/numwin)/100) '%'])
            windowed_sig = emg_index(winInd:winInd+winsize-1,:); 
            feat(winInd,:) = getrmsfeat(windowed_sig,winsize,wininc);   
        end
            %-------------------------------------------------------------
     end
    
    feat=feat(any(feat,2),:);
    %-------------------------------------------------------------
    
    %-------------------------------------------------------------
%     force=D.force4_1(:,4);
  

    winsize=800;
    wininc=20;
    if exist('ker'),
        parpool(ker,'IdleTimeout', 30);
    end
    
    numwin = size(force_index,1);
    nSignals = size(force_index,2);
    
    %-------------------------------------------------------------
    % allocate memory
    
    feat_out = zeros(numwin,nSignals,'single');
    %-------------------------------------------------------------
    
    parfor winInd = 1:numwin-winsize,
        if mod(winInd-1,wininc)==0,
            disp(['Feature Extraction Progress: ' num2str(round(winInd*10000/numwin)/100) '%'])
            windowed_sig = force_index(winInd:winInd+winsize-1,:); 
            feat_out(winInd,:) = getrmsfeat(windowed_sig,winsize,wininc);      
        end
           
     end
    feat_out=feat_out(any(feat_out,2),:);
end