function [ACC]=classify_lstm(feat,featRep,featStim)
    %% LSTM Regression for Generator
    maxEpochs = 200;
    miniBatchSize = 1948*3;
    
    switch size(feat,2)
            case 12
                numFeatures = 12;  
                feat_=feat;
            case 10
               numFeatures = 10; 
               feat_=[feat(:,1:8) feat(:,11:12)];
            case 2
               numFeatures = 2;
               feat_=feat(:,11:12);
        end
    
    numClasses = 9;
    numHiddenUnits = 100;
    % numFeatures = 12;
    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits,'OutputMode','last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    options = trainingOptions('adam', ...
        'ExecutionEnvironment','gpu', ...
        'GradientThreshold',1, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest', ...
        'Shuffle','never', ...
        'Verbose',0, ...
        'Plots','training-progress');
    %% Prepare Test and Train sets:
    DB=[feat_ featRep];
    % Test Data- second and fifth repetitions
    row_2= find(DB(:,size(DB,2)) == 2);
    row_5= find(DB(:,size(DB,2)) == 5);
    
    TestData_2=[];
    TestData_5=[];
    class_test_2=[];
    class_test_5=[];
    
    for i=1:length(row_2)
     TestData_2(i,:)= DB(row_2(i),1:size(feat_,2));
     class_test_2(i,:)= featStim(row_2(i),:);
    end
    
    for i=1:length(row_5)
     TestData_5(i,:)= DB(row_5(i),1:size(feat_,2));
     class_test_5(i,:)= featStim(row_5(i),:);
    end
    TestData=[TestData_2;TestData_5];
    TestClass=[class_test_2;class_test_5];
    
    % Train Data- first, third, fourth, sixth repetetions.
    DB_train= DB(:,1:size(feat_,2));
    row_cl=[row_2;row_5];
    DB_train(row_cl,:)=[];
    
    
    featStim_train=featStim;
    featStim_train(row_cl,:)=[];
    
    TrainData=DB_train;
    TrainClass=featStim_train;
    %% Training
    z=find(TrainClass==0);
    TrainClass(z,:)=[];%1948 minibatch could be used.
    TrainData(z,:)=[];

%     x=[TrainData TrainClass];
%     random_x = x(randperm(size(x, 1)), :);
%     TrainData=random_x(:,1:end-1);
    

    Train_in={};
    Train_label={};
    % o={};
    i=1;
    r=1;
    for j=1:length(TrainData)
        if mod(j,1)==0
            l=TrainData(i:j,:);
%             l_o=num2str(TrainClass(i:j,:));
            Train_in(r,:)={l'};
%             Train_label(r,:)={l_o'};
            i=i+1;
            r=r+1;
        end
    end
%     [Train_Class_labels,idx]=sort(TrainClass);
%     Train_Class_labels;
%     Train_in=Train_in(idx);
%     Train_label_categorical_prep=TrainClass(idx);
    


% 
%     Train_label_categorical=categorical(random_x(:,size(random_x,2)));


    
    
    Train_label_categorical=categorical(TrainClass);
    net = trainNetwork(Train_in,Train_label_categorical,layers,options);
    %% Validating/Testing
    z=find(TestClass==0);
    TestClass(z,:)=[];%1948 minibatch could be used.
    TestData(z,:)=[];
% 
%     x_=[TestData TestClass];
%     random_x_ = x_(randperm(size(x_, 1)), :);
%     TestData=random_x_(:,1:end-1);

    Test_in={};
    % o={};
    i=1;
    r=1;
    for j=1:length(TestData)
        if mod(j,1)==0
            l_=TestData(i:j,:);
%             l_o_=TestClass(i:j,:);
            Test_in(r,:)={l_'};
%             Test_label(r,:)={l_o_'};
            i=i+1;
            r=r+1;
        end
    end

%       [Test_Class_labels,idx]=sort(TestClass);
%       Test_Class_labels;
%       Test_in= Test_in(idx);
%       Test_label_categorical_prep= TestClass(idx);
% 
%       z=find(Test_label_categorical_prep==0);
%       Test_label_categorical_prep(z,:)=[];%1948 minibatch could be used.
%       Test_in(z,:)=[];
%     
%       x_=[Test_in Test_label_categorical_prep];
%       random_x_ = x_(randperm(size(x_, 1)), :);
%       Test_label_categorical=categorical(random_x_(:,size(random_x_,2)));
%       Test_label_categorical=categorical(Test_label_categorical_prep);
      

%     z_=find(Test_label_categorical==categorical(0));
%     Test_label_categorical(z_,:)=[];
%     Test_in(z_,:)=[];
    Test_label_categorical=categorical(TestClass);
%     label=predict(net,Test_in);
    %% Classify

    Predicted_label = classify(net,Test_in, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest');
    
    %% Calculate Accuracy of the model:
    Nc = length(find(Predicted_label==Test_label_categorical));
    Na=size(Predicted_label,1);
    ACC=100*(Nc/Na);
%     acc_lstm = sum(YPred == dum(179:278))./numel(dum(179:278))
    fprintf("The Classification Model Accuracy using LSTM is: %f%%\n",ACC)
end