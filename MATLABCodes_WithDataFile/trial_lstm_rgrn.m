
%% LSTM Regression for Generator
maxEpochs = 12000;
miniBatchSize = 324;%1945/6
tr=feat_out(:,3:4);

numResponses = 2;
numHiddenUnits = 200;
numFeatures = 12;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(150)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
%% Training on feat_1 and tr_1:
feat_cell={};
% o={};
i=1;
r=1;
for j=1:length(feat)
    if mod(j,1)==0
        l=feat(i:j,:);
        l_o=tr(i:j,:);
        feat_cell(r,:)={l'};
        feat_cell_o(r,:)={l_o'};
        i=i+1;
        r=r+1;
    end
end

%         figure
%         plot(cell2mat(feat_cell),'g')
%        
%         figure
%         plot(cell2mat(feat_cell_o),'g')


net = trainNetwork(feat_cell,feat_cell_o,layers,options);
%% Testing on feat_2:
feat_cell_in_test={};
% o={};
i=1;
r=1;
for j=1:length(feat_2)
    if mod(j,1)==0
        l_=feat_2(i:j,:);
        l_o_=feat_out_2(i:j,3:4);
        feat_cell_in_test(r,:)={l_'};
        feat_cell_o_test(r,:)={l_o_'};

        i=i+1;
        r=r+1;
    end
end
%         figure
%         plot(cell2mat(feat_cell_in_test),'r')
%         figure
%         plot(cell2mat(feat_cell_o_test),'g')
%         
        feat_pred=predict(net,feat_cell_in_test);
% Plot comparing feat_pred and feat_out_2:
    i=1;
    k=cell2mat(feat_pred);
    k_tr=cell2mat(feat_cell_o_test);
    r=1;
    for j=1:length(k)/2
        
            feat_pred_unrolled_(i:i+1-1,:)=k(r:r+1,:)';
            feat_cell_o_test_unrolled_(i:i+1-1,:)=k_tr(r:r+1,:)';
            i=i+1;
            r=r+2;
           
    end
% figure,plot(cell2mat(feat_pred),'r'), hold on, plot(cell2mat(feat_cell_o_test),'g')
figure,plot(feat_pred_unrolled_,'r'), hold on, plot(feat_cell_o_test_unrolled_,'g')