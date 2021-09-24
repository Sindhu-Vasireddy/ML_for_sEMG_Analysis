function [ACC]=classify_knn(feat,featRep,featStim)
    %% Classification into 1 of 10 classes.
    switch size(feat,2)
            case 12
                feat_=feat; 
            case 10
               
               feat_=[feat(:,1:8) feat(:,11:12)];
            case 2
               
               feat_=feat(:,11:12);
    end
    
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
    
%     K=3
    % [TrainErr,TestErr,TrainPredict,TestPredict] = knnclassify(TrainData,TestData,TrainClass,TestClass,K)
    
    Mdl=fitcknn(TrainData,TrainClass,'NumNeighbors',10,'Standardize',1)
    label=predict(Mdl,TestData);
    Nc = length(find(label==TestClass));
    Na=size(label,1);
    ACC=100*(Nc/Na);
    
    fprintf("The Classification Model Accuracy using k-nn algorithm with fitcknn is: %f%%\n",ACC)
end
% [Idx,D]=knnsearch(TrainData,TestData,'k',3,'Distance','euclidean');
% for index=1:length(Idx), labels(index)=TrainClass(Idx(index));end
% labels=labels';
% Nc_ = length(find(labels==TestClass));
% ACC_=100*(Nc_/length(labels));
% 
% fprintf("The Classification Model Accuracy using k-nn algorithm with knnsearch is: %f%%\n",ACC_)
% 
