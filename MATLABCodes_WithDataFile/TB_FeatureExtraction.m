D_0=load('Datafile1.mat');
D_0.emg=D_0.emgMov;
D_0.force=D_0.force4_1;
     emg=D_0.emg;% s-EMG input
    force=D_0.force(:,:);% index finger movement output
    winsize=800;
    wininc=20;
    if exist('ker'),
        parpool(ker,'IdleTimeout', 30);
    end
    
    
    numwin = size(emg,1);
    nSignals = size(emg,2);
    
    %-------------------------------------------------------------
    % allocate memory
    
    feat = zeros(numwin,nSignals,'single');
    
    %-------------------------------------------------------------
    parfor winInd = 1:numwin-winsize,
        
        if mod(winInd-1,wininc)==0,
            disp(['Feature Extraction Progress: ' num2str(round(winInd*10000/numwin)/100) '%'])
            windowed_sig = emg(winInd:winInd+winsize-1,:); 
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
    
    numwin = size(force,1);
    nSignals = size(force,2);
    
    %-------------------------------------------------------------
    % allocate memory
    
    feat_out = zeros(numwin,nSignals,'single');
    %-------------------------------------------------------------
    
    parfor winInd = 1:numwin-winsize,
        if mod(winInd-1,wininc)==0,
            disp(['Feature Extraction Progress: ' num2str(round(winInd*10000/numwin)/100) '%'])
            windowed_sig = force(winInd:winInd+winsize-1,:); 
            feat_out(winInd,:) = getrmsfeat(windowed_sig,winsize,wininc);      
        end
           
     end
    feat_out=feat_out(any(feat_out,2),:);
