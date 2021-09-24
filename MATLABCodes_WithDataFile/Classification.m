
clc
close all
clear all;
% D=load('S1_E3_A1.mat'); 
D=load('C:\Users\sindh\OneDrive\Documents\ACS6200_TB\DB2_s2\DB2_s2\S2_E3_A1.mat');
%% Filtering using Feature Extraction

featFunc='getrmsfeat';

emg=D.emg;

winsize=400;
wininc=20;
if exist('ker'),
    parpool(ker,'IdleTimeout', 30);
end


numwin = size(emg,1);
nSignals = size(emg,2);

%-------------------------------------------------------------
% allocate memory

feat = zeros(numwin,nSignals,'single');

featStim=zeros(numwin,1);
featRep=zeros(numwin,1);
checkStimRep=zeros(numwin,1);
%-------------------------------------------------------------
parfor winInd = 1:numwin-winsize,
    
    if mod(winInd-1,wininc)==0,
        disp(['Feature Extraction Progress: ' num2str(round(winInd*10000/numwin)/100) '%'])
        curStimWin=D.stimulus(winInd:winInd+winsize,:);
        curRepWin=D.repetition(winInd:winInd+winsize,:);
                
        if size(unique(curStimWin))==1 & size(unique(curRepWin))==1,
        
            checkStimRep(winInd,1)=1;
            featStim(winInd,1)=curStimWin(1);
            featRep(winInd,1)=curRepWin(1);
            windowed_sig = emg(winInd:winInd+winsize-1,:); 
            feat(winInd,:) = getrmsfeat(windowed_sig,winsize,wininc);   
         end
    end     %-------------------------------------------------------------
 end

%feat=feat(any(feat,2),:);
%-------------------------------------------------------------
%Remove features that correspond to windows without unique stimulus and repetition
z=find(checkStimRep==0);
feat(z,:)=[];
featStim(z,:)=[];
featRep(z,:)=[];

%--------------------------------------------------------------------
[ACC]=classify_knn(feat,featRep,featStim)
[ACC]=classify_lstm(feat,featRep,featStim)

%---------------------------------------------------------------------




















%----------------------------------------------------------------------------
%-------------------------------------------------------------
% %force=D.force4_1(:,4);
% force=D.force(:,4);
% % force=force(:,4);

% winsize=400;
% wininc=20;
% if exist('ker'),
%     parpool(ker,'IdleTimeout', 30);
% end
% 
% numwin = size(force,1);
% nSignals = size(force,2);
% 
% %-------------------------------------------------------------
% % allocate memory
% 
% feat_out = zeros(numwin,nSignals,'single');
% featStim_out=zeros(numwin,1);
% featRep_out=zeros(numwin,1);
% checkStimRep_out=zeros(numwin,1);
%-------------------------------------------------------------
% 
% parfor winInd = 1:numwin-winsize,
%     if mod(winInd-1,wininc)==0,
%         disp(['Feature Extraction Progress: ' num2str(round(winInd*10000/numwin)/100) '%'])
%         curStimWin_out=D.stimulus(winInd:winInd+winsize,:);
%         curRepWin_out=D.repetition(winInd:winInd+winsize,:);
%                 
%         if size(unique(curStimWin_out))==1 & size(unique(curRepWin_out))==1,
%         
%             checkStimRep_out(winInd,1)=1;
%             featStim_out(winInd,1)=curStimWin_out(1);
%             featRep_out(winInd,1)=curRepWin_out(1);
%             windowed_sig = force(winInd:winInd+winsize-1,:); 
%             feat_out(winInd,:) = getrmsfeat(windowed_sig,winsize,wininc);      
%         end
%     end    
%  end
% %feat_out=feat_out(any(feat_out,2),:);
% %-------------------------------------------------------------------
% %Remove features that correspond to windows without unique stimulus and repetition
% z_out=find(checkStimRep_out==0);
% feat_out(z_out,:)=[];
% featStim_out(z_out,:)=[];
% featRep_out(z_out,:)=[];






