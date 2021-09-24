function [fit_lstm,RMSE_lstm]=lstm_rgrn(train_in,train_out,test_in,test_out)
    %% LSTM Regression for Generator
    maxEpochs = 12000;
    miniBatchSize = 324;
    
    
    
    numHiddenUnits = 200;
    switch size(train_in,2)
        case 12
            switch size(train_out,2)
                case 1
                    numResponses = 1;
                    numFeatures = 12;
                case 2
                    numResponses = 2;
                    numFeatures = 12;    
            end
        case 10
           switch size(train_out,2)
                case 1
                    numResponses = 1;
                    numFeatures = 10;
                case 2
                    numResponses = 2;
                    numFeatures = 10;    
            end

        case 2
            switch size(train_out,2)
                case 1
                    numResponses = 1;
                    numFeatures = 2;
                case 2
                    numResponses = 2;
                    numFeatures = 2;    
           end
    end

    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits,'OutputMode','sequence')
        fullyConnectedLayer(150)
        dropoutLayer(0.5)
        fullyConnectedLayer(numResponses)
        regressionLayer];
    
    options = trainingOptions('adam', ...
        'ExecutionEnvironment','cpu', ...
        'GradientThreshold',1, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest', ...
        'Shuffle','never', ...
        'Verbose',0, ...
        'Plots','training-progress');
    %% Training on feat_1 and feat_out_1:
    feat_cell={};
    % o={};
    i=1;
    r=1;
    for j=1:length(train_in)
        if mod(j,1)==0
            l=train_in(i:j,:);
            l_o=train_out(i:j,:);
            feat_cell(r,:)={l'};
            feat_cell_o(r,:)={l_o'};
            i=i+1;
            r=r+1;
        end
    end
    net = trainNetwork(feat_cell,feat_cell_o,layers,options);

    %% Testing on feat_2:
    feat_cell_in_test={};
    % o={};
    i=1;
    r=1;
    for j=1:length(test_in)
        if mod(j,1)==0
            l_=test_in(i:j,:);
            l_o_=test_out(i:j,:);
            feat_cell_in_test(r,:)={l_'};
            feat_cell_o_test(r,:)={l_o_'};
            i=i+1;
            r=r+1;
        end
    end
    feat_pred=predict(net,feat_cell_in_test);
    
    %% Plot comparing feat_pred and feat_out_2:
    switch size(test_out,2)
        case 1
            figure,plot(cell2mat(feat_pred),'r'), hold on, plot(cell2mat(feat_cell_o_test),'g')

            % R-Squared Calculation:
            y_pred=cell2mat(feat_pred);
            y_true=cell2mat(feat_cell_o_test);
            % err=immse(Predicted_Out,feat_out)
            RMSE_lstm=sqrt(mean((y_pred-y_true).^2));
            Mean=mean(y_true);
            
            SSR_lstm=sum((y_pred-Mean).^2,1);
            SST_lstm=sum((y_true-Mean).^2,1);
            
            R_squared_lstm= SSR_lstm./SST_lstm;
            fit_lstm=R_squared_lstm.*100;
            
            fprintf('Goodness of fit for LSTM Regression: %f%%, %f%% \n',fit_lstm)
        case 2
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
            figure,plot(feat_pred_unrolled_,'r'), hold on, plot(feat_cell_o_test_unrolled_,'g')

            % R-Squared Calculation:
            y_pred=feat_pred_unrolled_;
            y_true=feat_cell_o_test_unrolled_;
            % err=immse(Predicted_Out,feat_out)
            RMSE_lstm=sqrt(mean((y_pred-y_true).^2));
            Mean=mean(y_true);
            SSR_lstm=sum((y_pred-Mean).^2,1);
            SST_lstm=sum((y_true-Mean).^2,1);
            
            R_squared_lstm= SSR_lstm./SST_lstm;
            fit_lstm=R_squared_lstm.*100;
            
            fprintf('Goodness of fit for LSTM Regression: %f%%, %f%% \n',fit_lstm)
            fprintf('RMSE for LSTM Regression: %f, %f \n',RMSE_lstm)
    end

end
